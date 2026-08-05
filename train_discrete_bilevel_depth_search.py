"""Discrete, demonstration-free bilevel depth search.

Each outer round samples two *hard* deletion candidates.  Each candidate is
independently initialized from the frozen teacher, repaired for the same paired
inner schedule, and evaluated on paired held-out states/noises.  A categorical
architecture distribution is updated from the resulting preference.  There is
no shared supernet and no straight-through gradient through the subnet.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from teacher_flow_deployment import build_flow, load_deployed_teacher
from train_flow_progressive_compression import initialize_student
from train_teacher_generated_flow_v2 import differentiable_integrate


def cvar(values, fraction=0.2):
    n = max(1, int(math.ceil(values.numel() * fraction)))
    return values.flatten().topk(n).values.mean()


def set_distances(teacher_x, student_x):
    teacher_x, student_x = teacher_x.flatten(2), student_x.flatten(2)
    center = teacher_x.mean(1, keepdim=True)
    scale = (teacher_x - center).square().mean((1, 2), keepdim=True).sqrt().clamp_min(0.05)
    distance = torch.cdist((teacher_x - center) / scale, (student_x - center) / scale)
    distance = distance.square() / teacher_x.shape[-1]
    return distance.min(2).values.mean(), distance.min(1).values.mean()


def build_candidate(teacher, selected_layers):
    model = build_flow(len(selected_layers), 72, 4, "cuda", 16).train()
    initialize_student(teacher, model, torch.eye(72), selected_layers)
    return model


@torch.no_grad()
def inclusion_probabilities(logits, k, temperature, samples=4096):
    noise = -torch.log(-torch.log(torch.rand(samples, len(logits), device=logits.device).clamp_(1e-6, 1-1e-6)))
    selected = (logits[None] / temperature + noise).topk(k, dim=1).indices
    counts = torch.zeros_like(logits).scatter_add_(0, selected.flatten(), torch.ones_like(selected, dtype=logits.dtype).flatten())
    return counts / samples


def repair_candidate(model, teacher, data, schedules, learning_rate):
    optimizer = torch.optim.AdamW(model.get_params(), lr=learning_rate, weight_decay=1e-5)
    states, noises, endpoints = data
    for indices, t in schedules:
        state, noise, endpoint = states[indices].cuda(), noises[indices].cuda(), endpoints[indices].cuda()
        tv = torch.full((len(state),), t, device="cuda")
        with torch.no_grad():
            x_t = teacher.integrate(noise, state, start_time=0.0, end_time=t,
                                    steps=max(1, round(16 * t)))
            target_velocity = teacher.velocity(x_t, tv, state)
        velocity = F.mse_loss(model.velocity(x_t, tv, state), target_velocity)
        endpoint_loss = F.mse_loss(differentiable_integrate(model, noise, state, steps=16), endpoint)
        loss = velocity + 0.03 * endpoint_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_params(), 1.0)
        optimizer.step()
    return model.eval()


@torch.inference_mode()
def evaluate_candidate(model, teacher, state, noise, groups, k):
    teacher_endpoint = teacher.integrate(noise, state, steps=16)
    student_endpoint = model.integrate(noise, state, steps=16)
    errors = (student_endpoint - teacher_endpoint).flatten(1).square().mean(1)
    trajectory = []
    for t in (0.25, 0.5, 0.75):
        steps = max(1, round(16 * t))
        teacher_x = teacher.integrate(noise, state, start_time=0.0, end_time=t, steps=steps)
        student_x = model.integrate(noise, state, start_time=0.0, end_time=t, steps=steps)
        trajectory.append(F.mse_loss(student_x, teacher_x))
    shape = (groups, k) + tuple(teacher_endpoint.shape[1:])
    coverage, precision = set_distances(teacher_endpoint.reshape(shape), student_endpoint.reshape(shape))
    return {"endpoint_cvar20": float(cvar(errors)),
            "trajectory_paired": float(torch.stack(trajectory).mean()),
            "coverage": float(coverage), "precision": float(precision)}


def preferred(a, b):
    keys = ["endpoint_cvar20", "trajectory_paired", "coverage", "precision"]
    votes_a = sum(a[key] < b[key] for key in keys)
    votes_b = len(keys) - votes_a
    if votes_a == votes_b:
        return 0 if a["endpoint_cvar20"] < b["endpoint_cvar20"] else 1
    return 0 if votes_a > votes_b else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--teacher-layers", type=int, default=4)
    p.add_argument("--target-k", type=int, default=3)
    p.add_argument("--inner-epochs", type=int, default=5)
    p.add_argument("--inner-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--outer-groups", type=int, default=32)
    p.add_argument("--samples-per-state", type=int, default=4)
    p.add_argument("--weight-lr", type=float, default=3e-5)
    p.add_argument("--architecture-lr", type=float, default=0.15)
    p.add_argument("--holdout-residue", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(a.buffer, map_location="cpu")
    meta = raw.get("metadata", {})
    assert not meta.get("uses_original_demonstrations", False)
    assert not meta.get("uses_expert_actions", False)
    episode_ids = raw["episode_ids"].long()
    train_indices = torch.where((episode_ids % 10) != a.holdout_residue)[0]
    heldout_indices = torch.where((episode_ids % 10) == a.holdout_residue)[0]
    data = (raw["states"].float(), raw["noises"].float(), raw["teacher_endpoints"].float())
    teacher, _, _ = load_deployed_teacher(a.bundle_dir)
    teacher.eval()

    if not 0 < a.target_k < a.teacher_layers:
        p.error("target-k must be between 1 and teacher-layers-1")
    logits = torch.zeros(a.teacher_layers, device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=a.architecture_lr)
    history, start_round = [], 1
    state_path = a.output_dir / "search_state.pt"
    if a.resume and state_path.exists():
        saved = torch.load(state_path, map_location="cuda")
        logits.data.copy_(saved["logits"])
        optimizer.load_state_dict(saved["optimizer"])
        history, start_round = saved["history"], saved["round"] + 1

    for round_index in range(start_round, a.rounds + 1):
        progress = (round_index - 1) / max(1, a.rounds - 1)
        temperature = 1.5 * (0.30 / 1.5) ** progress
        epsilon = 0.25 + progress * (0.05 - 0.25)
        with torch.no_grad():
            # Exact-K hard subset. Epsilon mixes scores toward uniform exploration.
            effective_logits = (1 - epsilon) * logits
            u = torch.rand_like(logits).clamp_(1e-6, 1-1e-6)
            gumbel = -torch.log(-torch.log(u))
            base = sorted((effective_logits / temperature + gumbel).topk(a.target_k).indices.cpu().tolist())
            outside = [index for index in range(a.teacher_layers) if index not in base]
            removed = base[int(torch.randint(len(base), (1,)))]
            added = outside[int(torch.randint(len(outside), (1,)))]
            swapped = sorted([index for index in base if index != removed] + [added])

        # Round-specific but candidate-paired inner batches and time points.
        generator = torch.Generator().manual_seed(a.seed * 10000 + round_index)
        schedules = []
        for _ in range(a.inner_epochs * a.inner_batches):
            locations = torch.randint(len(train_indices), (a.batch_size,), generator=generator)
            indices = train_indices[locations]
            t = float(torch.rand((), generator=generator).clamp(0.02, 0.98))
            schedules.append((indices, t))

        group_locations = torch.randint(len(heldout_indices), (a.outer_groups,), generator=generator)
        group_states = data[0][heldout_indices[group_locations]].cuda()
        k = a.samples_per_state
        outer_state = group_states[:, None].expand(-1, k, *group_states.shape[1:]).reshape(
            -1, *group_states.shape[1:])
        torch.manual_seed(a.seed * 10000 + round_index + 777)
        outer_noise = torch.randn(len(outer_state), 5, 2, device="cuda")

        candidates, metrics = [base, swapped], []
        for candidate in candidates:
            # Reset stochastic training stream so the pair sees matched dropout/noise.
            torch.manual_seed(a.seed * 10000 + round_index + 123)
            model = build_candidate(teacher, candidate)
            model = repair_candidate(model, teacher, data, schedules, a.weight_lr)
            metrics.append(evaluate_candidate(model, teacher, outer_state, outer_noise,
                                              a.outer_groups, k))
            del model
            torch.cuda.empty_cache()

        winner_position = preferred(metrics[0], metrics[1])
        # The subsets differ by one swap, so preference gives direct layer credit.
        winner_layer, loser_layer = ((removed, added) if winner_position == 0 else (added, removed))
        preference_loss = F.softplus(-(logits[winner_layer] - logits[loser_layer]))
        ordered = logits.sort(descending=True).values
        boundary_margin = ordered[a.target_k - 1] - ordered[a.target_k]
        margin_weight = 0.01 * max(0.0, (progress - 0.60) / 0.40)
        margin_loss = F.softplus(1.0 - boundary_margin)
        loss = preference_loss + margin_weight * margin_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            probability = inclusion_probabilities(logits, a.target_k, temperature)
        row = {"round": round_index, "pair_masks": candidates, "metrics": metrics,
               "swapped_out_layer": removed, "swapped_in_layer": added,
               "winner_position": winner_position, "winner_mask": candidates[winner_position],
               "winner_layer": winner_layer, "loser_layer": loser_layer,
               "preference_loss": float(preference_loss.detach()),
               "temperature": temperature, "epsilon": epsilon,
               "margin_weight": margin_weight, "boundary_margin": float(boundary_margin.detach()),
               "logits": logits.detach().cpu().tolist(),
               "inclusion_probabilities": probability.cpu().tolist()}
        history.append(row)
        torch.save({"round": round_index, "logits": logits.detach(),
                    "optimizer": optimizer.state_dict(), "history": history}, state_path)
        (a.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)

    final_probability = inclusion_probabilities(logits, a.target_k, 0.30, 16384).cpu()
    selected_layers = sorted(logits.detach().topk(a.target_k).indices.cpu().tolist())
    summary = {"experiment": "categorical discrete bilevel depth search",
               "seed": a.seed, "rounds": a.rounds, "inner_epochs": a.inner_epochs,
               "teacher_layers": a.teacher_layers, "target_k": a.target_k,
               "initial_inclusion_probabilities": [a.target_k / a.teacher_layers] * a.teacher_layers,
               "final_logits": logits.detach().cpu().tolist(),
               "final_inclusion_probabilities": final_probability.tolist(),
               "selected_layers": selected_layers,
               "uses_shared_supernet": False, "uses_straight_through": False,
               "uses_mode_labels": False, "uses_environment_rewards": False,
               "uses_success_filter": False, "uses_original_demonstrations": False,
               "history": history}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
