"""Causal audit of attention-head repartition versus width compression.

All lanes use the same demonstration-free FM-3x72-16 teacher rollout buffer and
the same repair objective.  Only the student head layout and projection basis
change.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.models.diffusion.ema import ExponentialMovingAverage
from teacher_flow_deployment import build_flow
from train_flow_progressive_compression import initialize_student, selection_basis
from train_recoverable_width_compression import canonical_signs
from train_teacher_generated_flow_v2 import activation_matrix, differentiable_integrate, save_ema


METHODS = ("head_only_3", "per_head_coordinate_4", "per_head_pca_4", "global_pca_3")


def _pca_basis(values: torch.Tensor, output_dim: int) -> tuple[torch.Tensor, float]:
    centered = values - values.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:output_dim]
    basis = canonical_signs(eigenvectors[:, order])
    explained = float(
        eigenvalues[order].clamp_min(0).sum()
        / eigenvalues.clamp_min(0).sum().clamp_min(1e-12)
    )
    return basis, explained


def make_student_and_basis(method, activations, device):
    if method == "head_only_3":
        student = build_flow(3, 72, 3, device, 16).train()
        return student, torch.eye(72), {
            "method": method,
            "teacher_heads": 4,
            "student_heads": 3,
            "projection": "identity; attention tensor reshape only",
        }
    if method == "per_head_coordinate_4":
        student = build_flow(3, 48, 4, device, 16).train()
        basis, selection = selection_basis(activations, 3, 4)
        return student, basis, {
            "method": method,
            "teacher_heads": 4,
            "student_heads": 4,
            "projection": "independent top-energy coordinates per teacher head, 18->12",
            **selection,
        }
    if method == "per_head_pca_4":
        student = build_flow(3, 48, 4, device, 16).train()
        basis = torch.zeros(72, 48)
        fractions = []
        for head in range(4):
            local, fraction = _pca_basis(activations[:, head * 18:(head + 1) * 18], 12)
            basis[head * 18:(head + 1) * 18, head * 12:(head + 1) * 12] = local
            fractions.append(fraction)
        return student, basis, {
            "method": method,
            "teacher_heads": 4,
            "student_heads": 4,
            "projection": "block-diagonal PCA within each teacher head, 18->12",
            "per_head_explained_fraction": fractions,
        }
    if method == "global_pca_3":
        student = build_flow(3, 48, 3, device, 16).train()
        basis, fraction = _pca_basis(activations, 48)
        return student, basis, {
            "method": method,
            "teacher_heads": 4,
            "student_heads": 3,
            "projection": "global PCA, 72->48; matched joint-compression control",
            "explained_fraction": fraction,
        }
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--endpoint-weight", type=float, default=0.03)
    parser.add_argument("--save-epochs", default="50,100,250,500")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metadata = torch.load(args.bundle_dir / "deployment_metadata.pt", map_location="cpu")
    teacher = build_flow(3, 72, 4, "cuda", 16)
    checkpoint = args.teacher if args.teacher.is_file() else args.teacher / "eval_best_flow.pth"
    teacher.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    teacher.min_action = metadata["y_bounds_tensor"][0].cuda()
    teacher.max_action = metadata["y_bounds_tensor"][1].cuda()
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    data = torch.load(args.buffer, map_location="cpu")
    assert data["metadata"]["uses_original_demonstrations"] is False
    assert data["metadata"]["uses_expert_actions"] is False
    episode_ids = data["episode_ids"].long()
    unique_ids = torch.unique(episode_ids, sorted=True)
    if len(unique_ids) != len(data["successes"]):
        raise ValueError("episode-level successes do not align with sample episode ids")
    dense_ids = torch.searchsorted(unique_ids, episode_ids)
    if not torch.equal(unique_ids[dense_ids], episode_ids):
        raise ValueError("global episode id mapping is inconsistent")
    keep = data["successes"].bool()[dense_ids]
    states = data["states"].float()[keep]
    noises = data["noises"].float()[keep]
    endpoints = data["teacher_endpoints"].float()[keep]
    loader = DataLoader(
        TensorDataset(states, noises, endpoints),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    activations = activation_matrix(teacher, states, noises, 8, args.batch_size)
    student, basis, basis_metadata = make_student_and_basis(args.method, activations, "cuda")
    orthogonality_error = float((basis.T @ basis - torch.eye(basis.shape[1])).abs().max())
    if orthogonality_error > 1e-4:
        raise RuntimeError(f"non-orthonormal basis: {orthogonality_error}")
    initialize_student(teacher, student, basis, [0, 1, 2])
    initial_state = student.state_dict()
    torch.save(initial_state, args.output_dir / "initial_flow.pth")
    if args.method == "head_only_3":
        max_copy_diff = max(
            float((initial_state[key].detach().cpu() - value.detach().cpu()).abs().max())
            for key, value in teacher.state_dict().items()
        )
        if max_copy_diff != 0.0:
            raise RuntimeError(f"head-only lane must copy every tensor exactly: {max_copy_diff}")
    else:
        max_copy_diff = None

    optimizer = torch.optim.AdamW(student.get_params(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(student.get_params(), 0.995, "cuda")
    milestones = {int(value) for value in args.save_epochs.split(",")}
    history, best = [], math.inf
    for epoch in range(1, args.epochs + 1):
        velocity_values, endpoint_values = [], []
        for batch_index, (state, noise, endpoint) in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            state, noise, endpoint = state.cuda(), noise.cuda(), endpoint.cuda()
            time = float(torch.rand(()).clamp(0.02, 0.98))
            time_vector = torch.full((len(state),), time, device="cuda")
            with torch.no_grad():
                x_t = teacher.integrate(
                    noise, state, start_time=0.0, end_time=time,
                    steps=max(1, round(16 * time)),
                )
                target_velocity = teacher.velocity(x_t, time_vector, state)
            velocity_loss = F.mse_loss(student.velocity(x_t, time_vector, state), target_velocity)
            endpoint_loss = F.mse_loss(
                differentiable_integrate(student, noise, state, steps=16), endpoint
            )
            loss = velocity_loss + args.endpoint_weight * endpoint_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())
            velocity_values.append(float(velocity_loss.detach()))
            endpoint_values.append(float(endpoint_loss.detach()))
        scheduler.step()
        score = float(np.mean(velocity_values) + args.endpoint_weight * np.mean(endpoint_values))
        row = {
            "epoch": epoch,
            "selection_loss": score,
            "velocity_loss": float(np.mean(velocity_values)),
            "endpoint_loss": float(np.mean(endpoint_values)),
        }
        history.append(row)
        if score < best:
            best = score
            save_ema(student, ema, args.output_dir / "structure_best_flow.pth")
        if epoch in milestones:
            save_ema(student, ema, args.output_dir / f"pretrain_epoch_{epoch:04d}.pth")
        if epoch % 25 == 0:
            print(json.dumps(row), flush=True)

    (args.output_dir / "metrics.json").write_text(json.dumps({
        "experiment": "head-width causal mechanism audit",
        "teacher": "FM-3x72-16 keep013, 4 heads",
        "student_embed_dim": 72 if args.method == "head_only_3" else 48,
        "student_heads": 3 if args.method in {"head_only_3", "global_pca_3"} else 4,
        "method": args.method,
        "basis": basis_metadata,
        "orthogonality_error": orthogonality_error,
        "exact_tensor_copy_max_diff": max_copy_diff,
        "teacher_buffer_samples": len(states),
        "uses_original_demonstrations": False,
        "uses_expert_actions": False,
        "history": history,
    }, indent=2))


if __name__ == "__main__":
    main()
