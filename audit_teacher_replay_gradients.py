"""Audit mode coverage, replay errors, and gradient conflicts before training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from teacher_flow_deployment import build_flow


def huber_per_sample(prediction, target):
    value = torch.sqrt((prediction - target).square() + 0.01 ** 2) - 0.01
    return value.flatten(1).mean(1)


def gradients(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.cat([
        parameter.grad.flatten() if parameter.grad is not None
        else torch.zeros_like(parameter).flatten()
        for parameter in model.get_params()
    ])


def cosine(left, right):
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))


def predict(model, states, noises):
    zero = torch.zeros(len(states), device=states.device)
    return model.boundary_transition(noises, zero, torch.ones_like(zero), states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-buffer", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--student-buffer", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    teacher_data = torch.load(args.teacher_buffer, map_location="cpu")
    student_data = torch.load(args.student_buffer, map_location="cpu")
    assert not teacher_data["metadata"]["uses_original_demonstrations"]
    assert not teacher_data["metadata"]["uses_expert_actions"]
    assert not student_data["metadata"]["uses_original_demonstrations"]
    discovery = np.load(args.discovery)
    labels = torch.as_tensor(discovery["sample_latents"]).long()
    assert len(labels) == len(teacher_data["states"])
    success_sample = teacher_data["successes"][teacher_data["episode_ids"].long()].bool()
    replay_indices = torch.where(success_sample & (labels >= 0))[0]
    replay_counts = torch.bincount(labels[replay_indices], minlength=24)

    student_episode_ids = student_data["episode_ids"].long()
    unique_student = torch.unique(student_episode_ids)
    student_row = {int(episode): row for row, episode in enumerate(unique_student)}
    sample_rows = torch.tensor([student_row[int(episode)] for episode in student_episode_ids])
    student_success = student_data["student_successes"][sample_rows].bool()
    teacher_success = student_data["teacher_successes"][sample_rows].bool()
    anchor_indices = torch.where(student_success)[0]
    correction_indices = torch.where((~student_success) & teacher_success)[0]

    result = {
        "protocol": {
            "uses_original_demonstrations": False,
            "teacher_replay_samples": len(replay_indices),
            "teacher_replay_latent_counts": replay_counts.tolist(),
            "occupied_teacher_latents": int((replay_counts > 0).sum()),
            "student_success_anchor_samples": len(anchor_indices),
            "student_failure_teacher_success_correction_samples": len(correction_indices),
            "trials": args.trials,
        },
        "models": {},
    }
    for item in args.models:
        name, checkpoint = item.split("=", 1)
        model = build_flow(3, 48, 3, "cuda", 1)
        model.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
        model.train()
        with torch.no_grad():
            per_latent = []
            for latent in range(24):
                index = replay_indices[labels[replay_indices] == latent]
                if len(index) > 2048:
                    index = index[torch.randperm(len(index), generator=generator)[:2048]]
                prediction = predict(model, teacher_data["states"][index].float().cuda(),
                                     teacher_data["noises"][index].float().cuda())
                error = huber_per_sample(prediction,
                                         teacher_data["teacher_endpoints"][index].float().cuda())
                per_latent.append(float(error.mean()))
        trials = []
        for _ in range(args.trials):
            replay = replay_indices[torch.randint(len(replay_indices), (args.batch_size,), generator=generator)]
            anchor = anchor_indices[torch.randint(len(anchor_indices), (args.batch_size,), generator=generator)]
            correction = correction_indices[torch.randint(len(correction_indices), (args.batch_size,), generator=generator)]
            replay_loss = huber_per_sample(
                predict(model, teacher_data["states"][replay].float().cuda(), teacher_data["noises"][replay].float().cuda()),
                teacher_data["teacher_endpoints"][replay].float().cuda()).mean()
            replay_gradient = gradients(model, replay_loss)
            anchor_loss = huber_per_sample(
                predict(model, student_data["states"][anchor].float().cuda(), student_data["noises"][anchor].float().cuda()),
                student_data["student_endpoints"][anchor].float().cuda()).mean()
            anchor_gradient = gradients(model, anchor_loss)
            correction_loss = huber_per_sample(
                predict(model, student_data["states"][correction].float().cuda(), student_data["noises"][correction].float().cuda()),
                student_data["teacher_corrections"][correction].float().cuda()).mean()
            correction_gradient = gradients(model, correction_loss)
            trials.append({
                "losses": {"replay": float(replay_loss), "anchor": float(anchor_loss),
                           "correction": float(correction_loss)},
                "gradient_cosines": {
                    "replay_anchor": cosine(replay_gradient, anchor_gradient),
                    "replay_correction": cosine(replay_gradient, correction_gradient),
                    "anchor_correction": cosine(anchor_gradient, correction_gradient),
                },
                "gradient_norms": {"replay": float(replay_gradient.norm()),
                                   "anchor": float(anchor_gradient.norm()),
                                   "correction": float(correction_gradient.norm())},
            })
        result["models"][name] = {
            "teacher_replay_error_by_latent": per_latent,
            "teacher_replay_error_mean": float(np.mean(per_latent)),
            "teacher_replay_error_min": float(np.min(per_latent)),
            "teacher_replay_error_max": float(np.max(per_latent)),
            "mean_losses": {key: float(np.mean([trial["losses"][key] for trial in trials]))
                            for key in ("replay", "anchor", "correction")},
            "mean_gradient_cosines": {
                key: float(np.mean([trial["gradient_cosines"][key] for trial in trials]))
                for key in ("replay_anchor", "replay_correction", "anchor_correction")
            },
            "mean_gradient_norms": {
                key: float(np.mean([trial["gradient_norms"][key] for trial in trials]))
                for key in ("replay", "anchor", "correction")
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
