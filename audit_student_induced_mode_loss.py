"""Audit frozen-mode shortcut and Teacher conditional diversity without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from teacher_flow_deployment import build_flow
from train_bmd_inference_model import InferenceModel


def load_policy(path, steps, device):
    model = build_flow(3, 48, 3, device, steps)
    checkpoint = path if path.is_file() else path / "eval_best_flow.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    return model


def pseudo_path(model, states, noises, meta):
    batch, length = noises.shape[:2]
    flat_states = states.flatten(0, 1)
    flat_noises = noises.flatten(0, 1)
    zero = torch.zeros(len(flat_states), device=states.device)
    one = torch.ones_like(zero)
    endpoint = model.boundary_transition(flat_noises, zero, one, flat_states)
    current = flat_states[:, -1, :2] * meta["x_std"][:2] + meta["x_mean"][:2]
    waypoint = current + endpoint[:, -1] * meta["y_std"] + meta["y_mean"]
    return waypoint.reshape(batch, length, 2)


def classifier_metrics(classifier, path, labels, q_mean, q_std):
    velocity = torch.diff(path, dim=1, prepend=path[:, :1])
    logits = classifier((torch.cat((path, velocity), 2) - q_mean) / q_std)
    probability = logits.softmax(1)
    return {
        "accuracy": float((logits.argmax(1) == labels).float().mean()),
        "target_probability": float(probability.gather(1, labels[:, None]).mean()),
        "cross_entropy": float(F.cross_entropy(logits, labels)),
        "predicted_classes": int(logits.argmax(1).unique().numel()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Entries formatted name=checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--states", type=int, default=128)
    parser.add_argument("--noises", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = torch.load(args.buffer, map_location="cpu")
    assert not data["metadata"]["uses_original_demonstrations"]
    metadata = torch.load(args.bundle_dir / "deployment_metadata.pt", map_location=device)
    q_checkpoint = torch.load(args.classifier, map_location="cpu")
    classifier = InferenceModel(q_checkpoint["model"], 24).to(device)
    classifier.load_state_dict(q_checkpoint["state_dict"])
    classifier.eval()
    q_mean = torch.as_tensor(q_checkpoint["mean"], device=device)
    q_std = torch.as_tensor(q_checkpoint["std"], device=device)

    episode_ids = data["episode_ids"].long()
    unique_episodes = torch.unique(episode_ids)
    episode_to_row = {int(episode): row for row, episode in enumerate(unique_episodes)}
    assert len(unique_episodes) == len(data["teacher_successes"])
    assert all(int(unique_episodes[row]) <= int(unique_episodes[row + 1])
               for row in range(len(unique_episodes) - 1))
    successful = [episode for episode in unique_episodes
                  if bool(data["teacher_successes"][episode_to_row[int(episode)]])]
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(successful), generator=generator)
    selected_episodes = [successful[index] for index in permutation[:args.episodes]]
    selected_states, selected_noises, selected_labels = [], [], []
    for episode in selected_episodes:
        indices = torch.where(episode_ids == episode)[0]
        indices = indices[torch.argsort(data["control_steps"][indices])]
        chosen = indices[torch.linspace(0, len(indices) - 1, 32).round().long()]
        selected_states.append(data["states"][chosen])
        selected_noises.append(data["noises"][chosen])
        selected_labels.append(data["teacher_latents"][episode_to_row[int(episode)]])
    states = torch.stack(selected_states).float().to(device)
    noises = torch.stack(selected_noises).float().to(device)
    labels = torch.stack(selected_labels).long().to(device)

    result = {
        "protocol": {
            "uses_original_demonstrations": False,
            "successful_teacher_episodes": len(selected_episodes),
            "pseudo_path_points": 32,
            "conditional_states": args.states,
            "noises_per_state": args.noises,
        },
        "pseudo_path_classifier": {},
    }
    for item in args.models:
        name, checkpoint = item.split("=", 1)
        model = load_policy(Path(checkpoint), 1, device)
        with torch.no_grad():
            path = pseudo_path(model, states, noises, metadata)
            result["pseudo_path_classifier"][name] = classifier_metrics(
                classifier, path, labels, q_mean, q_std)

    teacher = load_policy(args.teacher, 16, device)
    eligible_indices = torch.cat([torch.where(episode_ids == episode)[0]
                                  for episode in selected_episodes])
    chosen = eligible_indices[torch.randperm(len(eligible_indices), generator=generator)
                              [:args.states]]
    conditional_states = data["states"][chosen].float().to(device)
    repeated_states = conditional_states[:, None].expand(-1, args.noises, -1, -1)
    fresh_noise = torch.randn(
        args.states, args.noises, conditional_states.shape[1], 2, generator=generator
    ).to(device)
    flat_states = repeated_states.flatten(0, 1)
    flat_noise = fresh_noise.flatten(0, 1)
    zero = torch.zeros(len(flat_states), device=device)
    one = torch.ones_like(zero)
    with torch.no_grad():
        output = teacher.boundary_transition(flat_noise, zero, one, flat_states)
    physical = output[:, -1] * metadata["y_std"] + metadata["y_mean"]
    physical = physical.reshape(args.states, args.noises, 2)
    distance = torch.cdist(physical, physical)
    upper = torch.triu_indices(args.noises, args.noises, 1, device=device)
    pairwise = distance[:, upper[0], upper[1]]
    centered = physical - physical.mean(1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / max(args.noises - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    effective_rank = eigenvalues.sum(1).square() / eigenvalues.square().sum(1).clamp_min(1e-12)
    bins = torch.round(physical / 0.02).long()
    occupied = torch.tensor([len(torch.unique(row, dim=0)) for row in bins])
    result["teacher_same_state_multi_noise"] = {
        "pairwise_distance_mean": float(pairwise.mean()),
        "per_state_pairwise_median": float(pairwise.mean(1).median()),
        "per_state_pairwise_p10": float(pairwise.mean(1).quantile(0.1)),
        "per_state_pairwise_p90": float(pairwise.mean(1).quantile(0.9)),
        "effective_rank_mean": float(effective_rank.mean()),
        "effective_rank_median": float(effective_rank.median()),
        "occupied_2cm_bins_mean": float(occupied.float().mean()),
        "occupied_2cm_bins_median": float(occupied.float().median()),
        "states_with_more_than_one_2cm_bin": float((occupied > 1).float().mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
