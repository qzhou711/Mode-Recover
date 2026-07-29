"""Repair a structurally compressed Flow policy before step distillation."""

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
from distill_flow_matching_avoiding import (
    conditional_pairwise_geometry,
    make_agent,
    make_student,
    repeat_conditions,
    save_ema,
)
from train_flow_compression_stage1 import (
    calibration_activations,
    centered_gram_error,
    pairwise_correlation,
)
from train_flow_progressive_compression import initialize_student, selection_basis


def teacher_derived_basis(activations, layers, embed_dim, heads, method="activation"):
    if (layers, embed_dim, heads) == (3, 48, 3):
        if method in {"activation", "early"}:
            basis, metadata = selection_basis(activations, layers, heads)
            if method == "early":
                metadata["teacher_layers"] = [0, 1, 2]
            metadata["method"] = method
            return basis, metadata
        if method == "structured":
            basis = torch.zeros(72, 48)
            basis[torch.arange(48), torch.arange(48)] = 1.0
            return basis, {
                "method": method,
                "teacher_layers": [0, 2, 3],
                "selected_channels": list(range(48)),
            }
        if method == "pca":
            centered = activations - activations.mean(dim=0, keepdim=True)
            _, singular_values, right_vectors = torch.linalg.svd(
                centered, full_matrices=False
            )
            basis = right_vectors[:48].T.contiguous()
            signs = torch.sign(
                basis[torch.argmax(basis.abs(), dim=0), torch.arange(48)]
            )
            basis = basis * torch.where(signs == 0, torch.ones_like(signs), signs)
            explained = singular_values[:48].square().sum() / singular_values.square().sum()
            return basis, {
                "method": method,
                "teacher_layers": [0, 2, 3],
                "explained_variance": float(explained),
            }
        raise ValueError(f"unsupported 3x48 initialization method: {method}")
    if (layers, embed_dim, heads) != (4, 54, 3):
        raise ValueError("supported students are 3x48x3 and 4x54x3")
    energy = activations.square().mean(dim=0)
    head_energy = energy.reshape(4, 18).mean(dim=1)
    selected_heads = torch.topk(head_energy, k=3).indices.sort().values
    selected = [
        channel
        for head in selected_heads.tolist()
        for channel in range(head * 18, (head + 1) * 18)
    ]
    basis = torch.zeros(72, 54)
    basis[selected, torch.arange(54)] = 1.0
    return basis, {
        "teacher_layers": [0, 1, 2, 3],
        "selected_heads": selected_heads.tolist(),
        "selected_channels": selected,
        "preserves_complete_attention_heads": True,
        "teacher_head_dim": 18,
        "student_head_dim": 18,
    }


def differentiable_heun(model, noise, state, steps):
    x = noise
    dt = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (x.shape[0],), index * dt, device=x.device, dtype=x.dtype
        )
        velocity = model.velocity(x, time, state)
        if index + 1 == steps:
            x = x + dt * velocity
        else:
            predictor = x + dt * velocity
            next_time = torch.full_like(time, (index + 1) * dt)
            next_velocity = model.velocity(predictor, next_time, state)
            x = x + 0.5 * dt * (velocity + next_velocity)
    return x


def feature_alignment(teacher, student, basis, x_t, time, state, active_layers):
    teacher_features, student_features, handles = [], [], []
    for block in teacher.model.blocks:
        handles.append(block.register_forward_hook(
            lambda _module, _inputs, output: teacher_features.append(output.detach())
        ))
    for block in student.model.blocks:
        handles.append(block.register_forward_hook(
            lambda _module, _inputs, output: student_features.append(output)
        ))
    with torch.no_grad():
        teacher_velocity = teacher.velocity(x_t, time, state)
    student_velocity = student.velocity(x_t, time, state)
    for handle in handles:
        handle.remove()
    losses = [
        F.smooth_l1_loss(student_features[index], teacher_features[index] @ basis)
        for index in range(active_layers)
    ]
    return teacher_velocity, student_velocity, torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-layers", type=int, required=True)
    parser.add_argument("--student-embed-dim", type=int, required=True)
    parser.add_argument("--student-heads", type=int, required=True)
    parser.add_argument(
        "--init-method",
        choices=["structured", "activation", "pca", "early"],
        default="activation",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches-per-epoch", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--conditional-samples", type=int, default=4)
    parser.add_argument("--repair-steps", type=int, default=16)
    parser.add_argument("--feature-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--flow-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(mode="disabled")
    agent = make_agent(args.teacher_dir, args.batch_size, 16)
    teacher = agent.model.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = make_student(
        agent, 16, args.student_layers, args.student_embed_dim,
        args.student_heads, "random",
    ).to(agent.device).train()
    activations = calibration_activations(
        teacher, agent.train_dataloader, agent.scaler, args.calibration_batches
    )
    basis, initialization = teacher_derived_basis(
        activations, args.student_layers, args.student_embed_dim,
        args.student_heads, args.init_method
    )
    initialization["load_max_abs_diff"] = initialize_student(
        teacher, student, basis, initialization["teacher_layers"]
    )
    basis = basis.to(agent.device)
    torch.save(student.state_dict(), args.output_dir / "initial_flow.pth")

    optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    ema = ExponentialMovingAverage(student.get_params(), args.ema_decay, agent.device)
    validation_state, validation_action, _ = next(iter(agent.test_dataloader))
    validation_state = agent.scaler.scale_input(validation_state).float()[:64]
    validation_action = agent.scaler.scale_output(validation_action).float()[:64]
    validation_noise = torch.randn_like(validation_action)
    with torch.no_grad():
        teacher_validation_endpoint = teacher.integrate(
            validation_noise, validation_state, steps=16
        )

    best_score, best_epoch, history = math.inf, -1, []
    for epoch in trange(args.epochs, desc=f"Repair {args.student_layers}x{args.student_embed_dim}"):
        totals, velocities, endpoints, geometries, features, flows = [], [], [], [], [], []
        active_layers = min(
            args.student_layers,
            1 + epoch * args.student_layers // max(1, args.epochs),
        )
        for batch_index, (state, action, _) in enumerate(agent.train_dataloader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = agent.scaler.scale_input(state).float()
            action = agent.scaler.scale_output(action).float()
            state, action = repeat_conditions(
                state, action, args.conditional_samples
            )
            noise = torch.randn_like(action)
            time = torch.rand(action.shape[0], device=action.device)
            x_data = (
                (1.0 - time[:, None, None]) * noise
                + time[:, None, None] * action
            )
            teacher_velocity, student_velocity, feature_loss = feature_alignment(
                teacher, student, basis, x_data, time, state, active_layers
            )
            with torch.no_grad():
                teacher_endpoint = teacher.integrate(noise, state, steps=16)
            student_endpoint = differentiable_heun(
                student, noise, state, args.repair_steps
            )
            velocity_loss = F.mse_loss(student_velocity, teacher_velocity)
            endpoint_loss = F.smooth_l1_loss(student_endpoint, teacher_endpoint)
            geometry_loss = conditional_pairwise_geometry(
                student_endpoint, teacher_endpoint, args.conditional_samples
            )
            flow_loss = student.loss(action, state)
            loss = (
                velocity_loss
                + args.endpoint_weight * endpoint_loss
                + args.geometry_weight * geometry_loss
                + args.feature_weight * feature_loss
                + args.flow_weight * flow_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())
            totals.append(float(loss.detach()))
            velocities.append(float(velocity_loss.detach()))
            endpoints.append(float(endpoint_loss.detach()))
            geometries.append(float(geometry_loss.detach()))
            features.append(float(feature_loss.detach()))
            flows.append(float(flow_loss.detach()))
        scheduler.step()

        ema.store(student.get_params())
        ema.copy_to(student.get_params())
        student.eval()
        with torch.no_grad():
            validation_endpoint = student.integrate(
                validation_noise, validation_state, steps=16
            )
            validation_mse = float(
                F.mse_loss(validation_endpoint, teacher_validation_endpoint).item()
            )
            validation_gram = centered_gram_error(
                validation_endpoint, teacher_validation_endpoint
            )
            validation_correlation = pairwise_correlation(
                validation_endpoint, teacher_validation_endpoint
            )
        student.train()
        ema.restore(student.get_params())
        validation_score = validation_mse + 0.1 * validation_gram
        record = {
            "epoch": epoch,
            "loss": float(np.mean(totals)),
            "velocity_loss": float(np.mean(velocities)),
            "endpoint_loss": float(np.mean(endpoints)),
            "geometry_loss": float(np.mean(geometries)),
            "feature_loss": float(np.mean(features)),
            "active_feature_layers": active_layers,
            "flow_loss": float(np.mean(flows)),
            "validation_endpoint_mse": validation_mse,
            "validation_pairwise_correlation": validation_correlation,
            "validation_centered_gram_mse": validation_gram,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if validation_score < best_score:
            best_score, best_epoch = validation_score, epoch
            save_ema(student, ema, args.output_dir / "eval_best_flow.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)
        if epoch + 1 in {10, 50, 100, args.epochs}:
            checkpoint = args.output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            save_ema(student, ema, checkpoint / "eval_best_flow.pth")

    save_ema(student, ema, args.output_dir / "last_flow.pth")
    summary = {
        "method": "teacher_aligned_repair",
        "teacher_checkpoint": str(args.teacher_dir / "eval_best_flow.pth"),
        "student_architecture": {
            "layers": args.student_layers,
            "embed_dim": args.student_embed_dim,
            "heads": args.student_heads,
            "head_dim": args.student_embed_dim // args.student_heads,
        },
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "steps": 16,
        "epochs": args.epochs,
        "repair_steps": args.repair_steps,
        "conditional_samples": args.conditional_samples,
        "feature_weight": args.feature_weight,
        "initialization": initialization,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "history": history,
    }
    (args.output_dir / "repair_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
