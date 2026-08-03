"""Demo-free 16-step structural repair on global Teacher and Student-induced states."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from agents.models.diffusion.ema import ExponentialMovingAverage
from teacher_flow_deployment import build_flow, load_deployed_teacher
from train_teacher_generated_flow_v2 import differentiable_integrate, save_ema


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--base-buffer", type=Path, required=True)
    p.add_argument("--induced-buffer", type=Path, required=True)
    p.add_argument("--initial-student", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--induced-ratio", type=float, required=True)
    p.add_argument("--base-latents", type=Path)
    p.add_argument("--balance-base-latents", action="store_true")
    p.add_argument("--induced-recovery-only", action="store_true")
    p.add_argument("--induced-success-only", action="store_true",
                   help="Use induced samples only from successful assisted episodes.")
    p.add_argument("--endpoint-weight", type=float, default=0.03)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--save-epochs", default="50,100,250")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not 0 < a.induced_ratio < 1:
        p.error("--induced-ratio must be between zero and one")
    if a.balance_base_latents and a.base_latents is None:
        p.error("--balance-base-latents requires --base-latents")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    base = torch.load(a.base_buffer, map_location="cpu")
    induced = torch.load(a.induced_buffer, map_location="cpu")
    for data in (base, induced):
        assert not data["metadata"]["uses_original_demonstrations"]
        assert not data["metadata"]["uses_expert_actions"]
    base_keep = base["successes"].bool()[base["episode_ids"].long()]
    base_states = base["states"].float()[base_keep]
    base_noises = base["noises"].float()[base_keep]
    base_endpoints = base["teacher_endpoints"].float()[base_keep]
    induced_keep = induced["teacher_control"].bool() if a.induced_recovery_only else torch.ones(len(induced["states"]), dtype=torch.bool)
    if a.induced_success_only:
        episode_offset = int(induced["episode_ids"].min())
        episode_index = induced["episode_ids"].long() - episode_offset
        if int(episode_index.max()) >= len(induced["successes"]):
            raise ValueError("induced episode ids do not align with successes")
        induced_keep &= induced["successes"].bool()[episode_index]
    induced_states = induced["states"].float()[induced_keep]
    induced_noises = induced["noises"].float()[induced_keep]
    induced_endpoints = induced["teacher_corrections"].float()[induced_keep]
    states = torch.cat((base_states, induced_states))
    noises = torch.cat((base_noises, induced_noises))
    endpoints = torch.cat((base_endpoints, induced_endpoints))
    source = torch.cat((torch.zeros(len(base_states)), torch.ones(len(induced_states))))
    dataset = TensorDataset(states, noises, endpoints, source)

    weights = torch.empty(len(dataset), dtype=torch.double)
    if a.balance_base_latents:
        latent_data = np.load(a.base_latents)
        labels = torch.from_numpy(latent_data["sample_latents"]).long()[base_keep]
        unique, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
        weights[: len(base_states)] = ((1 - a.induced_ratio) / len(unique) / counts[inverse]).double()
    else:
        weights[: len(base_states)] = (1 - a.induced_ratio) / len(base_states)
    weights[len(base_states) :] = a.induced_ratio / len(induced_states)
    sampler = WeightedRandomSampler(
        weights,
        a.batch_size * a.max_batches,
        replacement=True,
        generator=torch.Generator().manual_seed(a.seed),
    )
    loader = DataLoader(dataset, batch_size=a.batch_size, sampler=sampler, drop_last=True)

    teacher, _, meta = load_deployed_teacher(a.bundle_dir)
    student = build_flow(3, 48, 3, "cuda", 16)
    checkpoint = a.initial_student if a.initial_student.is_file() else a.initial_student / "eval_best_flow.pth"
    student.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    student.train()
    opt = torch.optim.AdamW(student.get_params(), lr=a.learning_rate, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(student.get_params(), 0.995, "cuda")
    milestones = {int(value) for value in a.save_epochs.split(",")}
    history = []
    best = math.inf
    for epoch in range(1, a.epochs + 1):
        velocity_values, endpoint_values, induced_fractions = [], [], []
        for batch_index, (state, noise, endpoint, is_induced) in enumerate(loader):
            if batch_index >= a.max_batches:
                break
            state, noise, endpoint = state.cuda(), noise.cuda(), endpoint.cuda()
            t = float(torch.rand(()).clamp(0.02, 0.98))
            tv = torch.full((len(state),), t, device="cuda")
            with torch.no_grad():
                x = teacher.integrate(noise, state, start_time=0.0, end_time=t, steps=max(1, round(16 * t)))
                target_velocity = teacher.velocity(x, tv, state)
            velocity_loss = F.mse_loss(student.velocity(x, tv, state), target_velocity)
            student_endpoint = differentiable_integrate(student, noise, state, steps=16)
            endpoint_loss = F.mse_loss(student_endpoint, endpoint)
            loss = velocity_loss + a.endpoint_weight * endpoint_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            opt.step()
            ema.update(student.get_params())
            velocity_values.append(float(velocity_loss.detach()))
            endpoint_values.append(float(endpoint_loss.detach()))
            induced_fractions.append(float(is_induced.mean()))
        sched.step()
        score = float(np.mean(velocity_values) + a.endpoint_weight * np.mean(endpoint_values))
        record = {
            "epoch": epoch,
            "selection_loss": score,
            "velocity_loss": float(np.mean(velocity_values)),
            "endpoint_loss": float(np.mean(endpoint_values)),
            "sampled_induced_fraction": float(np.mean(induced_fractions)),
        }
        history.append(record)
        if score < best:
            best = score
            save_ema(student, ema, a.output_dir / "structure_best_flow.pth")
        if epoch in milestones:
            save_ema(student, ema, a.output_dir / f"pretrain_epoch_{epoch:04d}.pth")
        if epoch % 25 == 0:
            print(json.dumps(record), flush=True)
    summary = {
        "experiment": "P6.3 demonstration-free Student-induced structural repair",
        "teacher_architecture": "FM-4x72-16",
        "student_architecture": "FM-3x48-16",
        "solver_steps": 16,
        "induced_ratio": a.induced_ratio,
        "balance_base_latents": a.balance_base_latents,
        "induced_recovery_only": a.induced_recovery_only,
        "induced_success_only": a.induced_success_only,
        "endpoint_weight": a.endpoint_weight,
        "base_samples": len(base_states),
        "induced_samples": len(induced_states),
        "uses_original_demonstrations": False,
        "uses_expert_actions": False,
        "history": history,
    }
    (a.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
