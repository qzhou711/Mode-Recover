"""Audit whether cross-episode trajectory relations survive on Student-induced states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from teacher_flow_deployment import build_flow


def load_policy(path, device="cuda"):
    model = build_flow(3, 48, 3, device, 1)
    checkpoint = path if path.is_file() else path / "eval_best_flow.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    return model


def predict_waypoints(model, states, noises, metadata):
    shape = noises.shape
    flat_states, flat_noises = states.flatten(0, 1), noises.flatten(0, 1)
    zero = torch.zeros(len(flat_states), device=states.device)
    output = model.boundary_transition(flat_noises, zero, torch.ones_like(zero), flat_states)
    current = flat_states[:, -1, :2] * metadata["x_std"][:2] + metadata["x_mean"][:2]
    waypoint = current + output[:, -1] * metadata["y_std"] + metadata["y_mean"]
    return waypoint.reshape(shape[0], shape[1], 2)


def upper_values(matrix):
    index = torch.triu_indices(len(matrix), len(matrix), 1, device=matrix.device)
    return matrix[index[0], index[1]], index


def relation_metrics(reference, prediction, labels):
    # Normalize away the global scale; preserve relative trajectory geometry.
    reference_distance = torch.cdist(reference.flatten(1), reference.flatten(1))
    prediction_distance = torch.cdist(prediction.flatten(1), prediction.flatten(1))
    rv, index = upper_values(reference_distance)
    pv, _ = upper_values(prediction_distance)
    rvn = rv / rv.mean().clamp_min(1e-8)
    pvn = pv / pv.mean().clamp_min(1e-8)
    correlation = torch.corrcoef(torch.stack((rv, pv)))[0, 1]
    same = labels[index[0]] == labels[index[1]]
    return {
        "distance_correlation": float(correlation),
        "normalized_relation_mae": float((rvn - pvn).abs().mean()),
        "reference_same_mode_distance": float(rv[same].mean()) if same.any() else None,
        "reference_cross_mode_distance": float(rv[~same].mean()),
        "prediction_same_mode_distance": float(pv[same].mean()) if same.any() else None,
        "prediction_cross_mode_distance": float(pv[~same].mean()),
        "prediction_cross_same_ratio": float(pv[~same].mean() / pv[same].mean())
        if same.any() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--seed", type=int, default=27182)
    args = parser.parse_args()
    data = torch.load(args.buffer, map_location="cpu")
    assert not data["metadata"]["uses_original_demonstrations"]
    metadata = torch.load(args.bundle_dir / "deployment_metadata.pt", map_location="cuda")
    episode_ids = data["episode_ids"].long()
    unique = torch.unique(episode_ids)
    row = {int(episode): index for index, episode in enumerate(unique)}
    successful = torch.tensor([int(episode) for episode in unique
                               if bool(data["teacher_successes"][row[int(episode)]])])
    generator = torch.Generator().manual_seed(args.seed)
    selected = successful[torch.randperm(len(successful), generator=generator)[:args.episodes]]
    states, noises, teacher_outputs, labels = [], [], [], []
    for episode in selected:
        indices = torch.where(episode_ids == episode)[0]
        indices = indices[torch.argsort(data["control_steps"][indices])]
        indices = indices[torch.linspace(0, len(indices) - 1, 32).round().long()]
        states.append(data["states"][indices])
        noises.append(data["noises"][indices])
        teacher_outputs.append(data["teacher_corrections"][indices])
        labels.append(data["teacher_latents"][row[int(episode)]])
    states = torch.stack(states).float().cuda()
    noises = torch.stack(noises).float().cuda()
    teacher_outputs = torch.stack(teacher_outputs).float().cuda()
    labels = torch.stack(labels).long().cuda()
    flat_states = states.flatten(0, 1)
    current = flat_states[:, -1, :2] * metadata["x_std"][:2] + metadata["x_mean"][:2]
    teacher_waypoint = current + teacher_outputs.flatten(0, 1)[:, -1] * metadata["y_std"] + metadata["y_mean"]
    teacher_waypoint = teacher_waypoint.reshape(len(selected), 32, 2)
    result = {"episodes": len(selected), "uses_original_demonstrations": False,
              "models": {}}
    for item in args.models:
        name, checkpoint = item.split("=", 1)
        model = load_policy(Path(checkpoint))
        with torch.no_grad():
            prediction = predict_waypoints(model, states, noises, metadata)
        result["models"][name] = {
            "absolute_waypoint": relation_metrics(teacher_waypoint, prediction, labels),
            "action_displacement": relation_metrics(
                teacher_waypoint - current.reshape(len(selected), 32, 2),
                prediction - current.reshape(len(selected), 32, 2), labels),
            "temporal_delta": relation_metrics(
                torch.diff(teacher_waypoint, dim=1),
                torch.diff(prediction, dim=1), labels),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
