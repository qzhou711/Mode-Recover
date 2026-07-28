"""Probe progressive architecture compression with mode-preserving objectives."""

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
from train_flow_compression_stage1 import (
    block_diagonal_basis,
    calibration_activations,
    centered_gram_error,
    pairwise_correlation,
    projected_tensor,
)


def selection_basis(activations, student_layers, student_heads):
    energy = activations.square().mean(dim=0)
    teacher_head_energy = energy.reshape(4, 18).mean(dim=1)
    if student_heads == 4:
        heads = torch.arange(4)
        channels_per_head = 12
    else:
        heads = torch.topk(teacher_head_energy, k=3).indices.sort().values
        channels_per_head = 16
    selected = []
    for head in heads.tolist():
        local = torch.topk(
            energy[head * 18:(head + 1) * 18], k=channels_per_head
        ).indices
        selected.extend((local + head * 18).sort().values.tolist())
    basis = torch.zeros(72, 48)
    basis[selected, torch.arange(48)] = 1.0
    return basis, {
        "teacher_layers": [0, 1, 2, 3] if student_layers == 4 else [0, 2, 3],
        "selected_heads": heads.tolist(),
        "selected_channels": selected,
    }


def initialize_student(teacher, student, basis, layer_map):
    device = next(iter(teacher.state_dict().values())).device
    basis = basis.to(device)
    basis2 = block_diagonal_basis(basis, 2)
    basis4 = block_diagonal_basis(basis, 4)
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    mapped = {}
    for student_key, target in student_state.items():
        teacher_key = student_key
        if ".blocks." in student_key:
            prefix, remainder = student_key.split(".blocks.", 1)
            student_block, suffix = remainder.split(".", 1)
            teacher_key = f"{prefix}.blocks.{layer_map[int(student_block)]}.{suffix}"
        mapped[student_key] = projected_tensor(
            teacher_state[teacher_key], target, student_key, basis, basis2, basis4
        )
    student.load_state_dict(mapped, strict=True)
    max_diff = max(
        float((student.state_dict()[key] - value).abs().max().item())
        for key, value in mapped.items()
    )
    if max_diff != 0.0:
        raise RuntimeError("teacher-derived initialization changed after strict load")
    return max_diff


def repeat_group(tensor, samples):
    return tensor.repeat_interleave(samples, dim=0)


def inverse_cluster_weights(teacher_values, samples, clusters=4):
    batch = teacher_values.shape[0] // samples
    points = teacher_values.flatten(1).reshape(batch, samples, -1).detach()
    clusters = min(clusters, samples)
    initial = torch.linspace(
        0, samples - 1, clusters, device=points.device
    ).round().long()
    centers = points[:, initial].clone()
    for _ in range(3):
        assignments = torch.cdist(points, centers).argmin(dim=-1)
        new_centers = []
        for cluster in range(clusters):
            mask = (assignments == cluster).unsqueeze(-1)
            count = mask.sum(dim=1).clamp_min(1)
            center = (points * mask).sum(dim=1) / count
            empty = mask.sum(dim=1).squeeze(-1) == 0
            center[empty] = centers[:, cluster][empty]
            new_centers.append(center)
        centers = torch.stack(new_centers, dim=1)
    counts = torch.stack(
        [(assignments == cluster).sum(dim=1) for cluster in range(clusters)], dim=1
    ).clamp_min(1)
    weights = torch.gather(1.0 / counts.float(), 1, assignments)
    return (weights / weights.mean(dim=1, keepdim=True)).reshape(-1)


def sinkhorn_loss(student_values, teacher_values, samples, epsilon=0.05, iterations=12):
    batch = student_values.shape[0] // samples
    student = student_values.flatten(1).reshape(batch, samples, -1)
    teacher = teacher_values.flatten(1).reshape(batch, samples, -1)
    cost = torch.cdist(student, teacher).square() / student.shape[-1]
    log_kernel = -cost / epsilon
    log_marginal = -math.log(samples)
    log_u = torch.zeros_like(cost[:, :, 0])
    log_v = torch.zeros_like(cost[:, 0, :])
    for _ in range(iterations):
        log_u = log_marginal - torch.logsumexp(
            log_kernel + log_v[:, None, :], dim=2
        )
        log_v = log_marginal - torch.logsumexp(
            log_kernel + log_u[:, :, None], dim=1
        )
    transport = torch.exp(log_u[:, :, None] + log_kernel + log_v[:, None, :])
    return (transport * cost).sum(dim=(1, 2)).mean()


def grouped_objective(prediction, target, samples, use_balancing, sinkhorn_weight):
    element_error = (prediction - target).square().flatten(1).mean(dim=1)
    if use_balancing:
        weights = inverse_cluster_weights(target, samples)
        pointwise = (weights * element_error).mean()
        distribution = sinkhorn_loss(prediction, target, samples)
    else:
        pointwise = element_error.mean()
        distribution = pointwise * 0.0
    return pointwise + sinkhorn_weight * distribution, pointwise, distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method", choices=["width", "intermediate", "balanced", "ddil"], required=True
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches-per-epoch", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--conditional-samples", type=int, default=8)
    parser.add_argument("--sinkhorn-weight", type=float, default=0.1)
    parser.add_argument("--flow-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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

    student_layers = 4 if args.method == "width" else 3
    student_heads = 4 if args.method == "width" else 3
    student = make_student(
        agent, 16, student_layers, 48, student_heads, initialization="random"
    ).to(agent.device).train()
    activations = calibration_activations(
        teacher, agent.train_dataloader, agent.scaler, args.calibration_batches
    )
    basis, initialization = selection_basis(
        activations, student_layers, student_heads
    )
    initialization["load_max_abs_diff"] = initialize_student(
        teacher, student, basis, initialization["teacher_layers"]
    )

    optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    ema = ExponentialMovingAverage(student.get_params(), args.ema_decay, agent.device)
    use_balancing = args.method in {"balanced", "ddil"}
    samples = args.conditional_samples if use_balancing else 1

    validation_state, validation_action, _ = next(iter(agent.test_dataloader))
    validation_state = agent.scaler.scale_input(validation_state).float()
    validation_action = agent.scaler.scale_output(validation_action).float()
    validation_state = repeat_group(validation_state[: max(1, 64 // samples)], samples)
    validation_action = repeat_group(validation_action[: max(1, 64 // samples)], samples)
    validation_noise = torch.randn_like(validation_action)
    validation_time_base = torch.linspace(
        0.05, 0.95, validation_state.shape[0] // samples,
        device=validation_state.device,
    )
    validation_time = repeat_group(validation_time_base, samples)
    validation_x = (
        (1.0 - validation_time[:, None, None]) * validation_noise
        + validation_time[:, None, None] * validation_action
    )
    with torch.no_grad():
        validation_target = teacher.velocity(
            validation_x, validation_time, validation_state
        )

    best_score, best_epoch, history = math.inf, -1, []
    for epoch in trange(args.epochs, desc=f"FM progressive {args.method}"):
        totals, pointwise_values, distribution_values, ddil_values = [], [], [], []
        for batch_index, (state, action, _) in enumerate(agent.train_dataloader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = agent.scaler.scale_input(state).float()
            action = agent.scaler.scale_output(action).float()
            base_count = max(1, state.shape[0] // samples)
            state = repeat_group(state[:base_count], samples)
            action = repeat_group(action[:base_count], samples)
            noise = torch.randn_like(action)
            time_base = torch.rand(base_count, device=action.device)
            time = repeat_group(time_base, samples)
            x_data = (
                (1.0 - time[:, None, None]) * noise
                + time[:, None, None] * action
            )
            with torch.no_grad():
                target = teacher.velocity(x_data, time, state)
            prediction = student.velocity(x_data, time, state)
            data_loss, pointwise, distribution = grouped_objective(
                prediction, target, samples, use_balancing, args.sinkhorn_weight
            )
            ddil_loss = data_loss * 0.0
            if args.method == "ddil":
                scalar_time = float(time_base.mean().item())
                if scalar_time > 1e-4:
                    with torch.no_grad():
                        x_student = student.integrate(
                            noise, state, start_time=0.0, end_time=scalar_time,
                            steps=max(1, int(round(8 * scalar_time))),
                        )
                        ddil_target = teacher.velocity(
                            x_student,
                            torch.full_like(time, scalar_time),
                            state,
                        )
                    ddil_prediction = student.velocity(
                        x_student, torch.full_like(time, scalar_time), state
                    )
                    ddil_loss, _, _ = grouped_objective(
                        ddil_prediction, ddil_target, samples, True,
                        args.sinkhorn_weight,
                    )
            flow_loss = student.loss(action, state)
            loss = data_loss + 0.5 * ddil_loss + args.flow_weight * flow_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())
            totals.append(float(loss.detach().item()))
            pointwise_values.append(float(pointwise.detach().item()))
            distribution_values.append(float(distribution.detach().item()))
            ddil_values.append(float(ddil_loss.detach().item()))
        scheduler.step()
        student.eval()
        with torch.no_grad():
            validation_prediction = student.velocity(
                validation_x, validation_time, validation_state
            )
            validation_score, _, validation_distribution = grouped_objective(
                validation_prediction, validation_target, samples, use_balancing,
                args.sinkhorn_weight,
            )
        student.train()
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "pointwise_loss": float(np.mean(pointwise_values)),
            "sinkhorn_loss": float(np.mean(distribution_values)),
            "ddil_loss": float(np.mean(ddil_values)),
            "validation_score": float(validation_score.item()),
            "validation_sinkhorn": float(validation_distribution.item()),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if record["validation_score"] < best_score:
            best_score, best_epoch = record["validation_score"], epoch
            save_ema(student, ema, args.output_dir / "eval_best_flow.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)

    save_ema(student, ema, args.output_dir / "last_flow.pth")
    student.load_state_dict(
        torch.load(args.output_dir / "eval_best_flow.pth", map_location=agent.device),
        strict=True,
    )
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
        "student_architecture": {
            "layers": student_layers, "embed_dim": 48, "heads": student_heads
        },
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "steps": 16,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "conditional_samples": samples,
        "sinkhorn_weight": args.sinkhorn_weight if use_balancing else 0.0,
        "uses_student_induced_states": args.method == "ddil",
        "initialization": initialization,
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
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
