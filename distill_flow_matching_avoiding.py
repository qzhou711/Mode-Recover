"""Distill an Avoiding Flow Matching policy into a shortcut/few-step student."""

import argparse
import copy
import json
import math
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra import compose, initialize
from tqdm import trange

from agents.models.diffusion.ema import ExponentialMovingAverage


def make_agent(
    weights_dir: Path, batch_size: int, solver_steps: int,
    teacher_layers: int = 4, teacher_embed_dim: int = 72, teacher_heads: int = 4,
):
    with initialize(config_path="configs"):
        cfg = compose(
            config_name="avoiding_config",
            overrides=[
                "agents=flow_matching_transformer_agent",
                "window_size=5",
                "epoch=1",
                f"train_batch_size={batch_size}",
                f"n_layer={teacher_layers}",
                f"n_embd={teacher_embed_dim}",
                f"n_head={teacher_heads}",
                "simulation.render=False",
            ],
        )
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(str(weights_dir), sv_name="eval_best_flow.pth")
    agent.model.solver_steps = solver_steps
    return agent


def make_student(agent, solver_steps, layers, embed_dim, heads, initialization="auto"):
    use_teacher_init = initialization == "teacher" or (
        initialization == "auto" and layers == 0 and embed_dim == 0 and heads == 0
    )
    if use_teacher_init:
        student = copy.deepcopy(agent.model)
    else:
        layers = layers or 4
        embed_dim = embed_dim or 72
        heads = heads or 4
        if embed_dim % heads:
            raise ValueError("student embed dimension must be divisible by student heads")
        with initialize(config_path="configs"):
            cfg = compose(
                config_name="avoiding_config",
                overrides=[
                    "agents=flow_matching_transformer_agent",
                    "window_size=5",
                    "epoch=1",
                    f"n_layer={layers}",
                    f"n_embd={embed_dim}",
                    f"n_head={heads}",
                    "simulation.render=False",
                ],
            )
        student_agent = hydra.utils.instantiate(cfg.agents)
        student = student_agent.model
        student.min_action = agent.model.min_action.detach().clone()
        student.max_action = agent.model.max_action.detach().clone()
    student.solver_steps = solver_steps
    return student


def normalized_pairwise_geometry(student_endpoint, teacher_endpoint):
    if student_endpoint.shape[0] < 2:
        return student_endpoint.sum() * 0.0
    student_distances = torch.pdist(student_endpoint.flatten(1))
    teacher_distances = torch.pdist(teacher_endpoint.flatten(1))
    student_distances = student_distances / student_distances.mean().clamp_min(1e-6)
    teacher_distances = teacher_distances / teacher_distances.mean().clamp_min(1e-6)
    return F.mse_loss(student_distances, teacher_distances)


def conditional_pairwise_geometry(student_endpoint, teacher_endpoint, samples_per_state):
    if samples_per_state <= 1:
        return normalized_pairwise_geometry(student_endpoint, teacher_endpoint)
    if student_endpoint.shape[0] % samples_per_state:
        raise ValueError("batch size must be divisible by conditional samples")
    n_states = student_endpoint.shape[0] // samples_per_state
    student_grouped = student_endpoint.flatten(1).reshape(n_states, samples_per_state, -1)
    teacher_grouped = teacher_endpoint.flatten(1).reshape(n_states, samples_per_state, -1)
    student_distances = torch.cdist(student_grouped, student_grouped)
    teacher_distances = torch.cdist(teacher_grouped, teacher_grouped)
    mask = ~torch.eye(samples_per_state, dtype=torch.bool, device=student_endpoint.device)
    return F.smooth_l1_loss(student_distances[:, mask], teacher_distances[:, mask])


def repeat_conditions(state, action, samples_per_state):
    if samples_per_state <= 1:
        return state, action
    n_states = state.shape[0] // samples_per_state
    if n_states == 0:
        raise ValueError("batch size must be at least conditional samples")
    state = state[:n_states].repeat_interleave(samples_per_state, dim=0)
    action = action[:n_states].repeat_interleave(samples_per_state, dim=0)
    return state, action



def conditional_centered_gram(student_endpoint, teacher_endpoint, samples_per_state):
    """Match same-state noise-to-endpoint geometry without cross-state leakage."""
    if samples_per_state <= 1:
        return student_endpoint.sum() * 0.0
    if student_endpoint.shape[0] % samples_per_state:
        raise ValueError("batch size must be divisible by conditional samples")
    n_states = student_endpoint.shape[0] // samples_per_state
    student = student_endpoint.flatten(1).reshape(n_states, samples_per_state, -1)
    teacher = teacher_endpoint.flatten(1).reshape(n_states, samples_per_state, -1)
    student = student - student.mean(dim=1, keepdim=True)
    teacher = teacher - teacher.mean(dim=1, keepdim=True)
    feature_dim = student.shape[-1]
    student_gram = student @ student.transpose(1, 2) / feature_dim
    teacher_gram = teacher @ teacher.transpose(1, 2) / feature_dim
    return F.smooth_l1_loss(student_gram, teacher_gram)


def structured_teacher_state_dict(teacher_state, student_state):
    """Deterministically slice a 4x72 teacher into a 2x36 student."""
    embed_index = torch.tensor(
        [index for head in range(3) for index in range(head * 18, head * 18 + 12)]
    )
    mlp_index = torch.cat([embed_index + offset for offset in (0, 72, 144, 216)])
    time_index = torch.cat([embed_index, embed_index + 72])
    block_map = {0: 0, 1: 3}
    mapped = {}

    for student_key, student_tensor in student_state.items():
        teacher_key = student_key
        if ".blocks." in student_key:
            prefix, remainder = student_key.split(".blocks.", 1)
            student_block, suffix = remainder.split(".", 1)
            teacher_key = f"{prefix}.blocks.{block_map[int(student_block)]}.{suffix}"
        source = teacher_state[teacher_key]
        embed_index_device = embed_index.to(source.device)
        mlp_index_device = mlp_index.to(source.device)
        time_index_device = time_index.to(source.device)

        if source.shape == student_tensor.shape:
            value = source
        elif student_key.endswith("pos_emb"):
            value = source.index_select(2, embed_index_device)
        elif source.ndim == 1:
            index = (
                mlp_index_device if source.shape[0] == 288
                else time_index_device if source.shape[0] == 144
                else embed_index_device
            )
            value = source.index_select(0, index)
        elif source.ndim == 2:
            row_index = None
            column_index = None
            if source.shape[0] == 288:
                row_index = mlp_index_device
            elif source.shape[0] == 144:
                row_index = time_index_device
            elif source.shape[0] == 72 and student_tensor.shape[0] == 36:
                row_index = embed_index_device
            if source.shape[1] == 288:
                column_index = mlp_index_device
            elif source.shape[1] == 144:
                column_index = time_index_device
            elif source.shape[1] == 72 and student_tensor.shape[1] == 36:
                column_index = embed_index_device
            value = source
            if row_index is not None:
                value = value.index_select(0, row_index)
            if column_index is not None:
                value = value.index_select(1, column_index)
        else:
            raise ValueError(f"unsupported structured mapping for {student_key}")

        if value.shape != student_tensor.shape:
            raise ValueError(
                f"structured mapping shape mismatch for {student_key}: "
                f"{tuple(source.shape)} -> {tuple(value.shape)}, "
                f"expected {tuple(student_tensor.shape)}"
            )
        mapped[student_key] = value.detach().clone()
    return mapped

def save_ema(student, ema, path):
    ema.store(student.get_params())
    ema.copy_to(student.get_params())
    torch.save(student.state_dict(), path)
    ema.restore(student.get_params())


@torch.no_grad()
def teacher_shortcut_target(teacher, noise, state, start_time, teacher_steps):
    if start_time == 0.0:
        x_t = noise
    else:
        prefix_steps = max(1, int(round(teacher_steps * start_time)))
        x_t = teacher.integrate(noise, state, start_time=0.0, end_time=start_time, steps=prefix_steps)
    suffix_steps = max(1, int(round(teacher_steps * (1.0 - start_time))))
    endpoint = teacher.integrate(x_t, state, start_time=start_time, end_time=1.0, steps=suffix_steps)
    target_velocity = (endpoint - x_t) / (1.0 - start_time)
    return x_t, target_velocity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-steps", type=int, default=16)
    parser.add_argument("--teacher-layers", type=int, default=4)
    parser.add_argument("--teacher-embed-dim", type=int, default=72)
    parser.add_argument("--teacher-heads", type=int, default=4)
    parser.add_argument("--student-steps", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--flow-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--anchor-dir", type=Path, default=None)
    parser.add_argument("--anchor-steps", type=int, default=16)
    parser.add_argument("--geometry-weight", type=float, default=0.0)
    parser.add_argument("--distribution-weight", type=float, default=0.0)
    parser.add_argument("--conditional-samples", type=int, default=1)
    parser.add_argument("--student-layers", type=int, default=0)
    parser.add_argument("--student-embed-dim", type=int, default=0)
    parser.add_argument("--student-heads", type=int, default=0)
    parser.add_argument(
        "--student-init",
        choices=["auto", "teacher", "random", "checkpoint", "structured_teacher"],
        default="auto",
        help="Student initialization. Teacher initialization requires matching architectures.",
    )
    parser.add_argument(
        "--student-init-dir",
        type=Path,
        default=None,
        help="Directory containing eval_best_flow.pth for checkpoint initialization.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-epochs",
        type=int,
        nargs="*",
        default=[],
        help="One-based epochs whose EMA weights are saved under checkpoints/epoch_NNNN.",
    )
    args = parser.parse_args()
    if args.student_steps < 1 or args.student_steps >= args.teacher_steps:
        raise ValueError("student_steps must be in [1, teacher_steps)")
    if args.conditional_samples < 1:
        raise ValueError("conditional_samples must be positive")
    if any(epoch < 1 or epoch > args.epochs for epoch in args.save_epochs):
        raise ValueError("save epochs must be in [1, epochs]")
    if args.student_init == "teacher":
        requested_architecture = (
            args.student_layers or args.teacher_layers,
            args.student_embed_dim or args.teacher_embed_dim,
            args.student_heads or args.teacher_heads,
        )
        teacher_architecture = (
            args.teacher_layers,
            args.teacher_embed_dim,
            args.teacher_heads,
        )
        if requested_architecture != teacher_architecture:
            raise ValueError(
                "teacher initialization requires identical teacher/student architectures"
            )
    if args.student_init == "checkpoint" and args.student_init_dir is None:
        raise ValueError("checkpoint initialization requires --student-init-dir")
    if args.student_init != "checkpoint" and args.student_init_dir is not None:
        raise ValueError("--student-init-dir is only valid with checkpoint initialization")
    if args.anchor_weight < 0:
        raise ValueError("anchor_weight must be non-negative")
    if args.distribution_weight < 0:
        raise ValueError("distribution_weight must be non-negative")
    if (args.anchor_weight > 0) != (args.anchor_dir is not None):
        raise ValueError("positive --anchor-weight requires --anchor-dir, and vice versa")
    if args.anchor_steps < 1:
        raise ValueError("anchor_steps must be positive")
    save_epochs = set(args.save_epochs)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(mode="disabled")

    agent = make_agent(
        args.teacher_dir, args.batch_size, args.teacher_steps,
        args.teacher_layers, args.teacher_embed_dim, args.teacher_heads,
    )
    teacher = agent.model.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    anchor = None
    if args.anchor_dir is not None:
        anchor_agent = make_agent(
            args.anchor_dir, args.batch_size, args.anchor_steps,
            args.student_layers or 4, args.student_embed_dim or 72, args.student_heads or 4,
        )
        anchor = anchor_agent.model.to(agent.device).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
    student = make_student(
        agent,
        args.student_steps,
        args.student_layers,
        args.student_embed_dim,
        args.student_heads,
        args.student_init,
    ).to(agent.device).train()
    initialization_checkpoint = None
    if args.student_init == "checkpoint":
        initialization_checkpoint = args.student_init_dir / "eval_best_flow.pth"
        initialization_state = torch.load(
            initialization_checkpoint, map_location=agent.device
        )
        student.load_state_dict(initialization_state, strict=True)
        loaded_state = student.state_dict()
        if loaded_state.keys() != initialization_state.keys():
            raise RuntimeError("student checkpoint keys changed after strict loading")
        if not all(
            torch.equal(loaded_state[key], initialization_state[key])
            for key in loaded_state
        ):
            raise RuntimeError("student does not exactly match initialization checkpoint")
    elif args.student_init == "structured_teacher":
        if (args.student_layers, args.student_embed_dim, args.student_heads) != (2, 36, 3):
            raise ValueError(
                "structured_teacher currently supports only a 4x72 teacher and 2x36 student"
            )
        structured_state = structured_teacher_state_dict(
            teacher.state_dict(), student.state_dict()
        )
        student.load_state_dict(structured_state, strict=True)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    initialization_max_abs_diff = None
    if args.student_init == "teacher":
        initialization_max_abs_diff = max(
            float((student_parameter - teacher_parameter).abs().max().item())
            for student_parameter, teacher_parameter in zip(
                student.get_params(), teacher.get_params()
            )
        )
        if initialization_max_abs_diff != 0.0:
            raise RuntimeError("teacher-initialized student does not exactly match teacher")
    elif args.student_init in {"checkpoint", "structured_teacher"}:
        initialization_max_abs_diff = 0.0

    optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(student.get_params(), args.ema_decay, agent.device)

    validation_state, validation_action, _ = next(iter(agent.test_dataloader))
    validation_state = agent.scaler.scale_input(validation_state).float()
    validation_action = agent.scaler.scale_output(validation_action).float()
    validation_state, validation_action = repeat_conditions(
        validation_state, validation_action, args.conditional_samples
    )
    validation_noise = torch.randn_like(validation_action)
    validation_targets = []
    validation_anchor_targets = []
    with torch.no_grad():
        for grid_index in range(args.student_steps):
            start_time = grid_index / args.student_steps
            x_t, target_velocity = teacher_shortcut_target(
                teacher, validation_noise, validation_state, start_time, args.teacher_steps
            )
            validation_targets.append((start_time, x_t, target_velocity))
            if anchor is not None:
                _, anchor_velocity = teacher_shortcut_target(
                    anchor, validation_noise, validation_state, start_time, args.anchor_steps
                )
                validation_anchor_targets.append(anchor_velocity)
        validation_flow_t = torch.linspace(
            0.05, 0.95, validation_action.shape[0], device=validation_action.device
        )
        validation_flow_x = (
            (1.0 - validation_flow_t[:, None, None]) * validation_noise
            + validation_flow_t[:, None, None] * validation_action
        )
        validation_flow_target = validation_action - validation_noise

    best_loss, best_epoch, history = math.inf, -1, []
    for epoch in trange(args.epochs, desc=f"FM shortcut {args.teacher_steps}-to-{args.student_steps}"):
        totals, shortcut_losses, flow_losses, geometry_losses = [], [], [], []
        anchor_losses, distribution_losses = [], []
        for batch_index, (state, action, _) in enumerate(agent.train_dataloader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = agent.scaler.scale_input(state).float()
            action = agent.scaler.scale_output(action).float()
            state, action = repeat_conditions(state, action, args.conditional_samples)
            noise = torch.randn_like(action)
            grid_index = int(torch.randint(0, args.student_steps, (1,)).item())
            start_time = grid_index / args.student_steps
            x_t, target_velocity = teacher_shortcut_target(
                teacher, noise, state, start_time, args.teacher_steps
            )
            t = torch.full((action.shape[0],), start_time, device=action.device, dtype=action.dtype)
            predicted_velocity = student.velocity(x_t, t, state)
            shortcut_loss = F.mse_loss(predicted_velocity, target_velocity)
            flow_loss = student.loss(action, state)
            student_endpoint = x_t + (1.0 - start_time) * predicted_velocity
            teacher_endpoint = x_t + (1.0 - start_time) * target_velocity
            geometry_loss = conditional_pairwise_geometry(
                student_endpoint, teacher_endpoint, args.conditional_samples
            )
            distribution_loss = conditional_centered_gram(
                student_endpoint, teacher_endpoint, args.conditional_samples
            )
            if anchor is None:
                anchor_loss = predicted_velocity.sum() * 0.0
            else:
                with torch.no_grad():
                    _, anchor_velocity = teacher_shortcut_target(
                        anchor, noise, state, start_time, args.anchor_steps
                    )
                anchor_endpoint = x_t + (1.0 - start_time) * anchor_velocity
                anchor_loss = F.mse_loss(student_endpoint, anchor_endpoint)
            loss = (
                shortcut_loss
                + args.flow_weight * flow_loss
                + args.geometry_weight * geometry_loss
                + args.distribution_weight * distribution_loss
                + args.anchor_weight * anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())
            totals.append(loss.detach().item())
            shortcut_losses.append(shortcut_loss.detach().item())
            flow_losses.append(flow_loss.detach().item())
            geometry_losses.append(geometry_loss.detach().item())
            distribution_losses.append(distribution_loss.detach().item())
            anchor_losses.append(anchor_loss.detach().item())
        scheduler.step()
        student.eval()
        validation_shortcuts, validation_geometries = [], []
        validation_distributions, validation_anchors = [], []
        with torch.no_grad():
            for start_time, x_t, target_velocity in validation_targets:
                t = torch.full(
                    (validation_action.shape[0],), start_time,
                    device=validation_action.device, dtype=validation_action.dtype,
                )
                predicted_velocity = student.velocity(x_t, t, validation_state)
                validation_shortcuts.append(F.mse_loss(predicted_velocity, target_velocity))
                student_endpoint = x_t + (1.0 - start_time) * predicted_velocity
                validation_geometries.append(
                    conditional_pairwise_geometry(
                        student_endpoint,
                        x_t + (1.0 - start_time) * target_velocity,
                        args.conditional_samples,
                    )
                )
                validation_distributions.append(
                    conditional_centered_gram(
                        student_endpoint,
                        x_t + (1.0 - start_time) * target_velocity,
                        args.conditional_samples,
                    )
                )
                if anchor is not None:
                    validation_anchors.append(
                        F.mse_loss(
                            student_endpoint,
                            x_t + (1.0 - start_time) * validation_anchor_targets[len(validation_anchors)],
                        )
                    )
            validation_flow = F.mse_loss(
                student.velocity(validation_flow_x, validation_flow_t, validation_state),
                validation_flow_target,
            )
            validation_loss = (
                torch.stack(validation_shortcuts).mean()
                + args.flow_weight * validation_flow
                + args.geometry_weight * torch.stack(validation_geometries).mean()
                + args.distribution_weight * torch.stack(validation_distributions).mean()
                + args.anchor_weight * (
                    torch.stack(validation_anchors).mean()
                    if validation_anchors else validation_flow * 0.0
                )
            ).item()
        student.train()
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "shortcut_loss": float(np.mean(shortcut_losses)),
            "flow_loss": float(np.mean(flow_losses)),
            "geometry_loss": float(np.mean(geometry_losses)),
            "distribution_loss": float(np.mean(distribution_losses)),
            "anchor_loss": float(np.mean(anchor_losses)),
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if record["validation_loss"] < best_loss:
            best_loss, best_epoch = record["validation_loss"], epoch
            save_ema(student, ema, args.output_dir / "eval_best_flow.pth")
        completed_epoch = epoch + 1
        if completed_epoch in save_epochs:
            checkpoint_dir = args.output_dir / "checkpoints" / f"epoch_{completed_epoch:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_ema(student, ema, checkpoint_dir / "eval_best_flow.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)
    save_ema(student, ema, args.output_dir / "last_flow.pth")
    summary = {
        "teacher_checkpoint": str(args.teacher_dir / "eval_best_flow.pth"),
        "teacher_steps": args.teacher_steps,
        "teacher_architecture": {
            "layers": args.teacher_layers,
            "embed_dim": args.teacher_embed_dim,
            "heads": args.teacher_heads,
        },
        "student_steps": args.student_steps,
        "student_initialization": args.student_init,
        "student_initialization_checkpoint": (
            str(initialization_checkpoint) if initialization_checkpoint else None
        ),
        "initialization_max_abs_diff": initialization_max_abs_diff,
        "epochs": args.epochs,
        "saved_epochs": sorted(save_epochs),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "flow_weight": args.flow_weight,
        "anchor_weight": args.anchor_weight,
        "anchor_checkpoint": (
            str(args.anchor_dir / "eval_best_flow.pth") if args.anchor_dir else None
        ),
        "anchor_steps": args.anchor_steps,
        "geometry_weight": args.geometry_weight,
        "distribution_weight": args.distribution_weight,
        "conditional_samples": args.conditional_samples,
        "student_architecture": {
            "layers": (
                args.teacher_layers
                if args.student_layers == args.student_embed_dim == args.student_heads == 0
                else args.student_layers or 4
            ),
            "embed_dim": (
                args.teacher_embed_dim
                if args.student_layers == args.student_embed_dim == args.student_heads == 0
                else args.student_embed_dim or 72
            ),
            "heads": (
                args.teacher_heads
                if args.student_layers == args.student_embed_dim == args.student_heads == 0
                else args.student_heads or 4
            ),
        },
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "history": history,
    }
    (args.output_dir / "distillation_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
