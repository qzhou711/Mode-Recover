"""Demo-free recoverability training for exact depth subsets of FM-4x72-16."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.models.diffusion.ema import ExponentialMovingAverage
from teacher_flow_deployment import build_flow, load_deployed_teacher
from train_flow_progressive_compression import initialize_student
from train_teacher_generated_flow_v2 import differentiable_integrate, save_ema


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layer-map", required=True, help="Comma-separated teacher layers, e.g. 0,1,2")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--endpoint-weight", type=float, default=0.03)
    p.add_argument("--save-epochs", default="50,100,250,500")
    p.add_argument("--all-rollouts", action="store_true",
                   help="Do not filter teacher rollouts using environment success labels.")
    p.add_argument("--holdout-residue", type=int, default=-1,
                   help="Exclude episode_id %% 10 == residue from repair training.")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    layer_map = [int(x) for x in a.layer_map.split(",")]
    if not 0 < len(layer_map) < 4 or len(set(layer_map)) != len(layer_map) or min(layer_map) < 0 or max(layer_map) > 3:
        p.error("--layer-map must contain 1-3 distinct values from 0,1,2,3")
    if layer_map != sorted(layer_map):
        p.error("--layer-map must preserve teacher depth order")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    data = torch.load(a.buffer, map_location="cpu")
    assert data["metadata"]["uses_original_demonstrations"] is False
    assert data["metadata"]["uses_expert_actions"] is False
    episode_ids = data["episode_ids"].long()
    keep = torch.ones(len(episode_ids), dtype=torch.bool)
    if not a.all_rollouts:
        keep &= data["successes"].bool()[episode_ids]
    if a.holdout_residue >= 0:
        keep &= (episode_ids % 10) != a.holdout_residue
    states = data["states"].float()[keep]
    noises = data["noises"].float()[keep]
    endpoints = data["teacher_endpoints"].float()[keep]
    loader = DataLoader(
        TensorDataset(states, noises, endpoints), batch_size=a.batch_size,
        shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(a.seed),
    )

    teacher, _, meta = load_deployed_teacher(a.bundle_dir)
    assert (meta["teacher_layers"], meta["teacher_embed_dim"], meta["teacher_heads"], meta["teacher_steps"]) == (4, 72, 4, 16)
    student = build_flow(len(layer_map), 72, 4, "cuda", 16).train()
    identity = torch.eye(72)
    load_diff = initialize_student(teacher, student, identity, layer_map)
    torch.save(student.state_dict(), a.output_dir / "initial_flow.pth")

    # Exact-copy audit: every non-block tensor and each selected block must match bitwise.
    ts, ss = teacher.state_dict(), student.state_dict()
    copy_diffs = []
    for sk, sv in ss.items():
        tk = sk
        if ".blocks." in sk:
            prefix, remainder = sk.split(".blocks.", 1)
            block, suffix = remainder.split(".", 1)
            tk = f"{prefix}.blocks.{layer_map[int(block)]}.{suffix}"
        copy_diffs.append(float((sv.detach().cpu() - ts[tk].detach().cpu()).abs().max()))
    exact_copy_max_diff = max(copy_diffs)
    if exact_copy_max_diff != 0.0 or load_diff != 0.0:
        raise RuntimeError(f"depth subset initialization is not exact: {exact_copy_max_diff}")

    optimizer = torch.optim.AdamW(student.get_params(), lr=a.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(student.get_params(), 0.995, "cuda")
    milestones = {int(x) for x in a.save_epochs.split(",")}
    history, best = [], math.inf
    for epoch in range(1, a.epochs + 1):
        velocity_values, endpoint_values = [], []
        for bi, (state, noise, endpoint) in enumerate(loader):
            if bi >= a.max_batches:
                break
            state, noise, endpoint = state.cuda(), noise.cuda(), endpoint.cuda()
            t = float(torch.rand(()).clamp(0.02, 0.98))
            tv = torch.full((len(state),), t, device="cuda")
            with torch.no_grad():
                x_t = teacher.integrate(noise, state, start_time=0.0, end_time=t, steps=max(1, round(16 * t)))
                target_velocity = teacher.velocity(x_t, tv, state)
            velocity_loss = F.mse_loss(student.velocity(x_t, tv, state), target_velocity)
            student_endpoint = differentiable_integrate(student, noise, state, steps=16)
            endpoint_loss = F.mse_loss(student_endpoint, endpoint)
            loss = velocity_loss + a.endpoint_weight * endpoint_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())
            velocity_values.append(float(velocity_loss.detach()))
            endpoint_values.append(float(endpoint_loss.detach()))
        scheduler.step()
        score = float(np.mean(velocity_values) + a.endpoint_weight * np.mean(endpoint_values))
        row = {"epoch": epoch, "selection_loss": score,
               "velocity_loss": float(np.mean(velocity_values)),
               "endpoint_loss": float(np.mean(endpoint_values))}
        history.append(row)
        if score < best:
            best = score
            save_ema(student, ema, a.output_dir / "structure_best_flow.pth")
        if epoch in milestones:
            save_ema(student, ema, a.output_dir / f"pretrain_epoch_{epoch:04d}.pth")
        if epoch % 25 == 0:
            print(json.dumps(row), flush=True)

    summary = {
        "experiment": "TinySR-inspired recoverability-guided exact depth subset",
        "teacher_architecture": "FM-4x72-16", "student_architecture": f"FM-{len(layer_map)}x72-16",
        "layer_map": layer_map, "exact_copy_max_diff": exact_copy_max_diff,
        "teacher_buffer_samples": len(states), "endpoint_weight": a.endpoint_weight,
        "uses_environment_success_filter": not a.all_rollouts,
        "holdout_residue_modulo_10": a.holdout_residue,
        "uses_original_demonstrations": False, "uses_expert_actions": False,
        "history": history,
    }
    (a.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
