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


def make_student(agent, solver_steps, layers, embed_dim, heads):
    if layers == 0 and embed_dim == 0 and heads == 0:
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
    parser.add_argument("--geometry-weight", type=float, default=0.0)
    parser.add_argument("--conditional-samples", type=int, default=1)
    parser.add_argument("--student-layers", type=int, default=0)
    parser.add_argument("--student-embed-dim", type=int, default=0)
    parser.add_argument("--student-heads", type=int, default=0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--max-batches-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.student_steps < 1 or args.student_steps >= args.teacher_steps:
        raise ValueError("student_steps must be in [1, teacher_steps)")
    if args.conditional_samples < 1:
        raise ValueError("conditional_samples must be positive")
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
    student = make_student(
        agent, args.student_steps, args.student_layers, args.student_embed_dim, args.student_heads
    ).to(agent.device).train()
    for parameter in student.parameters():
        parameter.requires_grad_(True)

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
    with torch.no_grad():
        for grid_index in range(args.student_steps):
            start_time = grid_index / args.student_steps
            x_t, target_velocity = teacher_shortcut_target(
                teacher, validation_noise, validation_state, start_time, args.teacher_steps
            )
            validation_targets.append((start_time, x_t, target_velocity))
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
            loss = (
                shortcut_loss
                + args.flow_weight * flow_loss
                + args.geometry_weight * geometry_loss
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
        scheduler.step()
        student.eval()
        validation_shortcuts, validation_geometries = [], []
        with torch.no_grad():
            for start_time, x_t, target_velocity in validation_targets:
                t = torch.full(
                    (validation_action.shape[0],), start_time,
                    device=validation_action.device, dtype=validation_action.dtype,
                )
                predicted_velocity = student.velocity(x_t, t, validation_state)
                validation_shortcuts.append(F.mse_loss(predicted_velocity, target_velocity))
                validation_geometries.append(
                    conditional_pairwise_geometry(
                        x_t + (1.0 - start_time) * predicted_velocity,
                        x_t + (1.0 - start_time) * target_velocity,
                        args.conditional_samples,
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
            ).item()
        student.train()
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "shortcut_loss": float(np.mean(shortcut_losses)),
            "flow_loss": float(np.mean(flow_losses)),
            "geometry_loss": float(np.mean(geometry_losses)),
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if record["validation_loss"] < best_loss:
            best_loss, best_epoch = record["validation_loss"], epoch
            save_ema(student, ema, args.output_dir / "eval_best_flow.pth")
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
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "flow_weight": args.flow_weight,
        "geometry_weight": args.geometry_weight,
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
