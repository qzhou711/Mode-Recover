"""Audit label-free proxies for post-repair depth-subset recoverability.

This script deliberately never reads D3IL mode labels or environment rewards.  It
only compares a compressed Flow policy with the frozen teacher on held-out
teacher/student-induced states.  Ground-truth closed-loop metrics are joined by a
separate analysis step, so they cannot leak into the proxy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from teacher_flow_deployment import build_flow, load_deployed_teacher


def integrate_to(model, noise, state, end, steps):
    return model.integrate(noise, state, start_time=0.0, end_time=end, steps=steps)


def cvar(values, fraction=0.2):
    values = values.flatten()
    count = max(1, int(np.ceil(len(values) * fraction)))
    return values.topk(count).values.mean()


def normalized_gram(x):
    x = x.flatten(2)
    x = x - x.mean(1, keepdim=True)
    x = F.normalize(x, dim=-1)
    return x @ x.transpose(1, 2)


def set_distances(teacher, student):
    teacher = teacher.flatten(2)
    student = student.flatten(2)
    scale = teacher.std(dim=1, keepdim=True).mean(dim=-1, keepdim=True).clamp_min(1e-4)
    distances = torch.cdist(teacher / scale, student / scale).square() / teacher.shape[-1]
    return distances.min(2).values.mean(), distances.min(1).values.mean()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--student-checkpoint", type=Path, required=True)
    p.add_argument("--student-layers", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--groups", type=int, default=48)
    p.add_argument("--samples-per-state", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260804)
    p.add_argument("--episode-residue", type=int, default=-1,
                   help="When nonnegative, audit only episode_id %% 10 == residue.")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    raw = torch.load(a.buffer, map_location="cpu")
    metadata = raw.get("metadata", {})
    if metadata.get("uses_original_demonstrations", False) or metadata.get("uses_expert_actions", False):
        raise RuntimeError("proxy audit must be demonstration-free")
    states = raw["states"].float()
    if a.episode_residue >= 0:
        if "episode_ids" not in raw:
            raise RuntimeError("episode-residue requires episode_ids in buffer")
        states = states[(raw["episode_ids"].long() % 10) == a.episode_residue]
    generator = torch.Generator().manual_seed(a.seed)
    indices = torch.randperm(len(states), generator=generator)[:min(a.samples, len(states))]
    states = states[indices]
    noise_shape = tuple(raw["noises"].shape[1:])

    teacher, _, teacher_meta = load_deployed_teacher(a.bundle_dir)
    student = build_flow(a.student_layers, 72, 4, "cuda", 16).eval()
    student.load_state_dict(torch.load(a.student_checkpoint, map_location="cuda"), strict=True)
    teacher.eval()

    endpoint_errors, velocity_errors = [], []
    for start in range(0, len(states), a.batch_size):
        state = states[start:start + a.batch_size].cuda()
        noise = torch.randn((len(state),) + noise_shape, device="cuda")
        teacher_endpoint = integrate_to(teacher, noise, state, 1.0, 16)
        student_endpoint = integrate_to(student, noise, state, 1.0, 16)
        endpoint_errors.append((student_endpoint - teacher_endpoint).flatten(1).square().mean(1).cpu())
        per_time = []
        for t in (0.25, 0.5, 0.75):
            x_t = integrate_to(teacher, noise, state, t, max(1, round(16 * t)))
            tv = torch.full((len(state),), t, device="cuda")
            per_time.append((student.velocity(x_t, tv, state) - teacher.velocity(x_t, tv, state))
                            .flatten(1).square().mean(1))
        velocity_errors.append(torch.stack(per_time).mean(0).cpu())
    endpoint_errors = torch.cat(endpoint_errors)
    velocity_errors = torch.cat(velocity_errors)

    group_indices = torch.randperm(len(states), generator=generator)[:min(a.groups, len(states))]
    grouped_states = states[group_indices].cuda()
    k = a.samples_per_state
    expanded_states = grouped_states[:, None].expand(-1, k, *grouped_states.shape[1:]).reshape(
        -1, *grouped_states.shape[1:])
    noises = torch.randn((len(expanded_states),) + noise_shape, device="cuda")
    trajectory_gram = []
    trajectory_paired = []
    final_teacher = final_student = None
    for t in (0.25, 0.5, 0.75, 1.0):
        steps = max(1, round(16 * t))
        teacher_x = integrate_to(teacher, noises, expanded_states, t, steps).reshape(
            len(grouped_states), k, *noise_shape)
        student_x = integrate_to(student, noises, expanded_states, t, steps).reshape(
            len(grouped_states), k, *noise_shape)
        trajectory_gram.append(F.mse_loss(normalized_gram(student_x), normalized_gram(teacher_x)))
        trajectory_paired.append((student_x - teacher_x).flatten(2).square().mean((1, 2)).mean())
        if t == 1.0:
            final_teacher, final_student = teacher_x, student_x
    coverage, precision = set_distances(final_teacher, final_student)

    result = {
        "protocol": "label-free teacher-relative recoverability proxy audit v1",
        "mask": a.mask, "stage": a.stage,
        "student_checkpoint": str(a.student_checkpoint),
        "buffer": str(a.buffer), "buffer_metadata": metadata,
        "uses_mode_labels": False, "uses_environment_rewards": False,
        "uses_original_demonstrations": False,
        "sample_count": len(states), "group_count": len(grouped_states), "samples_per_state": k,
        "endpoint_mse_mean": endpoint_errors.mean().item(),
        "endpoint_mse_median": endpoint_errors.median().item(),
        "endpoint_mse_cvar20": cvar(endpoint_errors).item(),
        "velocity_mse_mean": velocity_errors.mean().item(),
        "velocity_mse_cvar20": cvar(velocity_errors).item(),
        "multi_noise_trajectory_gram_mse": torch.stack(trajectory_gram).mean().item(),
        "multi_noise_trajectory_paired_mse": torch.stack(trajectory_paired).mean().item(),
        "teacher_to_student_set_coverage": coverage.item(),
        "student_to_teacher_set_precision": precision.item(),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
