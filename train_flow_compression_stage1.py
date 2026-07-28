"""Stage-1 teacher-only compression from FM-4x72-16 to FM-2x36-16."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import trange

from agents.models.diffusion.ema import ExponentialMovingAverage
from distill_flow_matching_avoiding import make_agent, make_student, save_ema


def block_diagonal_basis(basis, repeats):
    rows, columns = basis.shape
    result = basis.new_zeros(rows * repeats, columns * repeats)
    for index in range(repeats):
        result[index * rows:(index + 1) * rows, index * columns:(index + 1) * columns] = basis
    return result


@torch.no_grad()
def calibration_activations(teacher, dataloader, scaler, batches):
    captured = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())

    handle = teacher.model.blocks[0].register_forward_pre_hook(capture)
    for batch_index, (state, action, _) in enumerate(dataloader):
        if batch_index >= batches:
            break
        state = scaler.scale_input(state).float()
        action = scaler.scale_output(action).float()
        noise = torch.randn_like(action)
        time = torch.rand(action.shape[0], device=action.device)
        time_view = time[:, None, None]
        x_t = (1.0 - time_view) * noise + time_view * action
        teacher.velocity(x_t, time, state)
    handle.remove()
    return torch.cat(captured, dim=0)


def activation_basis(activations):
    energy = activations.square().mean(dim=0)
    head_energy = energy.reshape(4, 18).mean(dim=1)
    selected_heads = torch.topk(head_energy, k=3).indices.sort().values
    selected = []
    for head in selected_heads.tolist():
        local = torch.topk(energy[head * 18:(head + 1) * 18], k=12).indices
        selected.extend((local + head * 18).sort().values.tolist())
    basis = torch.zeros(72, 36)
    basis[selected, torch.arange(36)] = 1.0
    return basis, {"selected_heads": selected_heads.tolist(), "selected_channels": selected}


def pca_basis(activations):
    centered = activations - activations.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:36]
    basis = eigenvectors[:, order]
    signs = torch.sign(basis[torch.argmax(basis.abs(), dim=0), torch.arange(36)])
    basis = basis * signs.clamp(min=-1.0, max=1.0)
    explained = eigenvalues[order].sum() / eigenvalues.clamp_min(0).sum().clamp_min(1e-12)
    return basis, {"explained_variance": float(explained)}


def projected_tensor(source, target, key, basis, basis2, basis4):
    if source.shape == target.shape:
        return source.detach().clone()
    if key.endswith("pos_emb"):
        return (source @ basis).detach().clone()
    bases = {72: basis, 144: basis2, 288: basis4}
    if source.ndim == 1:
        if ".ln" in key or key.endswith("ln_f.weight"):
            return torch.ones_like(target)
        output_basis = bases.get(source.shape[0])
        if output_basis is None:
            raise ValueError(f"no output basis for {key}: {tuple(source.shape)}")
        return (output_basis.T @ source).detach().clone()
    if source.ndim == 2:
        output_basis = bases.get(source.shape[0])
        input_basis = bases.get(source.shape[1])
        value = source
        if output_basis is not None:
            value = output_basis.T @ value
        if input_basis is not None:
            value = value @ input_basis
        if value.shape != target.shape:
            raise ValueError(
                f"projection mismatch for {key}: {tuple(source.shape)} -> "
                f"{tuple(value.shape)}, expected {tuple(target.shape)}"
            )
        return value.detach().clone()
    raise ValueError(f"unsupported tensor for {key}: {tuple(source.shape)}")


def compressed_state_dict(teacher_state, student_state, basis, merge_layers):
    device = next(iter(teacher_state.values())).device
    basis = basis.to(device)
    basis2 = block_diagonal_basis(basis, 2)
    basis4 = block_diagonal_basis(basis, 4)
    mapped = {}
    layer_sources = {0: [0, 1] if merge_layers else [0], 1: [2, 3] if merge_layers else [3]}

    for student_key, target in student_state.items():
        teacher_keys = [student_key]
        if ".blocks." in student_key:
            prefix, remainder = student_key.split(".blocks.", 1)
            student_block, suffix = remainder.split(".", 1)
            teacher_keys = [
                f"{prefix}.blocks.{teacher_block}.{suffix}"
                for teacher_block in layer_sources[int(student_block)]
            ]
        values = [
            projected_tensor(
                teacher_state[teacher_key], target, student_key, basis, basis2, basis4
            )
            for teacher_key in teacher_keys
        ]
        value = torch.stack(values).mean(dim=0) if len(values) > 1 else values[0]
        if value.shape != target.shape:
            raise ValueError(f"final mapping mismatch for {student_key}")
        mapped[student_key] = value
    return mapped


def pairwise_correlation(student_endpoint, teacher_endpoint):
    student_dist = torch.pdist(student_endpoint.flatten(1)).double()
    teacher_dist = torch.pdist(teacher_endpoint.flatten(1)).double()
    student_dist -= student_dist.mean()
    teacher_dist -= teacher_dist.mean()
    denominator = student_dist.norm() * teacher_dist.norm()
    return float((student_dist @ teacher_dist / denominator.clamp_min(1e-12)).item())


def centered_gram_error(student_endpoint, teacher_endpoint):
    student = student_endpoint.flatten(1)
    teacher = teacher_endpoint.flatten(1)
    student -= student.mean(dim=0, keepdim=True)
    teacher -= teacher.mean(dim=0, keepdim=True)
    student_gram = student @ student.T / student.shape[1]
    teacher_gram = teacher @ teacher.T / teacher.shape[1]
    return float(F.mse_loss(student_gram, teacher_gram).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method", choices=["activation", "pca", "merge", "random"], required=True
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches-per-epoch", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--flow-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(mode="disabled")
    agent = make_agent(args.teacher_dir, args.batch_size, 16, 4, 72, 4)
    teacher = agent.model.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = make_student(
        agent, 16, 2, 36, 3, initialization="random"
    ).to(agent.device).train()

    initialization_metadata = {"method": args.method}
    if args.method != "random":
        activations = calibration_activations(
            teacher, agent.train_dataloader, agent.scaler, args.calibration_batches
        )
        if args.method == "pca":
            basis, metadata = pca_basis(activations)
            merge_layers = False
        else:
            basis, metadata = activation_basis(activations)
            merge_layers = args.method == "merge"
        initialization_metadata.update(metadata)
        state = compressed_state_dict(
            teacher.state_dict(), student.state_dict(), basis, merge_layers
        )
        student.load_state_dict(state, strict=True)
        max_diff = max(
            float((student.state_dict()[key] - value).abs().max().item())
            for key, value in state.items()
        )
        if max_diff != 0.0:
            raise RuntimeError("compressed initialization changed after strict loading")
        initialization_metadata["load_max_abs_diff"] = max_diff

    optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    ema = ExponentialMovingAverage(student.get_params(), args.ema_decay, agent.device)

    validation_state, validation_action, _ = next(iter(agent.test_dataloader))
    validation_state = agent.scaler.scale_input(validation_state).float()
    validation_action = agent.scaler.scale_output(validation_action).float()
    validation_noise = torch.randn_like(validation_action)
    validation_time = torch.linspace(
        0.01, 0.99, validation_action.shape[0], device=validation_action.device
    )
    validation_x = (
        (1.0 - validation_time[:, None, None]) * validation_noise
        + validation_time[:, None, None] * validation_action
    )
    with torch.no_grad():
        validation_target = teacher.velocity(
            validation_x, validation_time, validation_state
        )

    best_loss, best_epoch, history = math.inf, -1, []
    for epoch in trange(args.epochs, desc=f"FM compression {args.method}"):
        totals, functional_losses, flow_losses = [], [], []
        for batch_index, (state, action, _) in enumerate(agent.train_dataloader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = agent.scaler.scale_input(state).float()
            action = agent.scaler.scale_output(action).float()
            noise = torch.randn_like(action)
            time = torch.rand(action.shape[0], device=action.device)
            time_view = time[:, None, None]
            x_t = (1.0 - time_view) * noise + time_view * action
            with torch.no_grad():
                target_velocity = teacher.velocity(x_t, time, state)
            prediction = student.velocity(x_t, time, state)
            functional_loss = F.mse_loss(prediction, target_velocity)
            flow_loss = student.loss(action, state)
            loss = functional_loss + args.flow_weight * flow_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())
            totals.append(float(loss.detach().item()))
            functional_losses.append(float(functional_loss.detach().item()))
            flow_losses.append(float(flow_loss.detach().item()))
        scheduler.step()
        student.eval()
        with torch.no_grad():
            validation_functional = F.mse_loss(
                student.velocity(validation_x, validation_time, validation_state),
                validation_target,
            ).item()
        student.train()
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "functional_loss": float(np.mean(functional_losses)),
            "flow_loss": float(np.mean(flow_losses)),
            "validation_functional_loss": validation_functional,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if validation_functional < best_loss:
            best_loss, best_epoch = validation_functional, epoch
            save_ema(student, ema, args.output_dir / "eval_best_flow.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)

    save_ema(student, ema, args.output_dir / "last_flow.pth")
    best_state = torch.load(
        args.output_dir / "eval_best_flow.pth", map_location=agent.device
    )
    student.load_state_dict(best_state, strict=True)
    student.eval()
    with torch.no_grad():
        teacher_endpoint = teacher.integrate(
            validation_noise, validation_state, steps=16
        )
        student_endpoint = student.integrate(
            validation_noise, validation_state, steps=16
        )
    summary = {
        "method": args.method,
        "teacher_checkpoint": str(args.teacher_dir / "eval_best_flow.pth"),
        "teacher_steps": 16,
        "student_steps": 16,
        "student_architecture": {"layers": 2, "embed_dim": 36, "heads": 3},
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_functional_loss": best_loss,
        "flow_weight": args.flow_weight,
        "initialization": initialization_metadata,
        "open_loop": {
            "endpoint_mse": float(F.mse_loss(student_endpoint, teacher_endpoint).item()),
            "pairwise_correlation": pairwise_correlation(
                student_endpoint, teacher_endpoint
            ),
            "centered_gram_mse": centered_gram_error(
                student_endpoint, teacher_endpoint
            ),
        },
        "history": history,
    }
    (args.output_dir / "compression_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
