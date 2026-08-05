"""Recoverability-guided hard Top-K depth search for Flow policies.

The gate is trained without demonstrations, rewards, or D3IL mode labels. Model
weights use a train episode split; gate logits use a disjoint held-out episode
split.  Forward passes always execute an exact three-block subnet.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from teacher_flow_deployment import build_flow, load_deployed_teacher
from train_flow_progressive_compression import initialize_student
from train_teacher_generated_flow_v2 import differentiable_integrate


class HardTopKGatedBlocks(nn.Module):
    def __init__(self, blocks, logits, topk=3, temperature=1.0):
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))
        self.logits = nn.Parameter(logits)
        self.topk = topk
        self.temperature = temperature
        self.last_hard_mask = None

    def probabilities(self):
        return self.logits.softmax(0)

    def _mask(self):
        if self.training:
            u = torch.rand_like(self.logits).clamp_(1e-6, 1 - 1e-6)
            scores = (self.logits + torch.log(u) - torch.log1p(-u)) / self.temperature
        else:
            scores = self.logits
        soft = torch.sigmoid(scores)
        soft = soft * (self.topk / soft.sum().clamp_min(1e-6))
        hard = torch.zeros_like(soft).scatter_(0, scores.topk(self.topk).indices, 1.0)
        self.last_hard_mask = hard.detach()
        return hard + soft - soft.detach()

    def forward(self, x):
        gates = self._mask()
        for gate, block in zip(gates, self.blocks):
            updated = block(x)
            x = x + gate * (updated - x)
        return x


def cvar(values, fraction=0.2):
    values = values.flatten()
    n = max(1, int(np.ceil(len(values) * fraction)))
    return values.topk(n).values.mean()


def normalized(value):
    """Equalize proxy gradient scales without changing its direction."""
    return value / value.detach().clamp_min(1e-6)


def set_losses(teacher_x, student_x):
    # [groups, K, horizon, action] -> normalize by teacher set scale per state.
    teacher_x = teacher_x.flatten(2)
    student_x = student_x.flatten(2)
    center = teacher_x.mean(1, keepdim=True)
    scale = (teacher_x - center).square().mean((1, 2), keepdim=True).sqrt().clamp_min(0.05)
    distances = torch.cdist((teacher_x - center) / scale, (student_x - center) / scale)
    distances = distances.square() / teacher_x.shape[-1]
    return distances.min(2).values.mean(), distances.min(1).values.mean()


def sample_batch(states, noises, endpoints, indices, batch_size, generator, device="cuda"):
    pick = indices[torch.randint(len(indices), (batch_size,), generator=generator)]
    return states[pick].to(device), noises[pick].to(device), endpoints[pick].to(device)


def inner_weight_step(model, teacher, batch, optimizer):
    state, noise, endpoint = batch
    t = float(torch.rand(()).clamp(0.02, 0.98))
    tv = torch.full((len(state),), t, device=state.device)
    with torch.no_grad():
        x_t = teacher.integrate(noise, state, start_time=0.0, end_time=t,
                                steps=max(1, round(16 * t)))
        target_velocity = teacher.velocity(x_t, tv, state)
    velocity = F.mse_loss(model.velocity(x_t, tv, state), target_velocity)
    endpoint_loss = F.mse_loss(differentiable_integrate(model, noise, state, steps=16), endpoint)
    loss = velocity + 0.03 * endpoint_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if "logits" not in n], 1.0)
    optimizer.step()
    return velocity.detach(), endpoint_loss.detach()


def outer_gate_step(model, teacher, states, objective, groups, k, optimizer):
    # Same state, multiple noises: native coupled samples for valid conditional comparisons.
    state = states[:, None].expand(-1, k, *states.shape[1:]).reshape(-1, *states.shape[1:])
    noise = torch.randn(len(state), 5, 2, device=state.device)
    with torch.no_grad():
        teacher_endpoint = teacher.integrate(noise, state, steps=16)
    student_endpoint = differentiable_integrate(model, noise, state, steps=16)
    errors = (student_endpoint - teacher_endpoint).flatten(1).square().mean(1)
    mean_endpoint = errors.mean()
    endpoint_cvar = cvar(errors)

    trajectory_terms = []
    if objective in {"cvar_traj", "cvar_traj_set"}:
        for t in (0.25, 0.5, 0.75):
            steps = max(1, round(16 * t))
            with torch.no_grad():
                teacher_x = teacher.integrate(noise, state, start_time=0.0, end_time=t, steps=steps)
            student_x = model.integrate(noise, state, start_time=0.0, end_time=t, steps=steps)
            trajectory_terms.append(F.mse_loss(student_x, teacher_x))
    trajectory = torch.stack(trajectory_terms).mean() if trajectory_terms else mean_endpoint.new_zeros(())

    coverage = precision = mean_endpoint.new_zeros(())
    if objective == "cvar_traj_set":
        shape = (groups, k) + tuple(teacher_endpoint.shape[1:])
        coverage, precision = set_losses(teacher_endpoint.reshape(shape), student_endpoint.reshape(shape))

    if objective == "mean":
        loss = normalized(mean_endpoint)
    elif objective == "cvar":
        loss = normalized(endpoint_cvar)
    elif objective == "cvar_traj":
        loss = normalized(endpoint_cvar) + normalized(trajectory)
    else:
        loss = (normalized(endpoint_cvar) + normalized(trajectory)
                + normalized(coverage) + normalized(precision))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([model.model.blocks.logits], 5.0)
    optimizer.step()
    return {"gate_loss": float(loss.detach()), "endpoint_mean": float(mean_endpoint.detach()),
            "endpoint_cvar20": float(endpoint_cvar.detach()), "trajectory": float(trajectory.detach()),
            "coverage": float(coverage.detach()), "precision": float(precision.detach())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--objective", choices=["mean", "cvar", "cvar_traj", "cvar_traj_set"], required=True)
    p.add_argument("--gate-epochs", type=int, default=250)
    p.add_argument("--inner-batches", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--outer-groups", type=int, default=8)
    p.add_argument("--samples-per-state", type=int, default=4)
    p.add_argument("--weight-lr", type=float, default=3e-5)
    p.add_argument("--gate-lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cpu_gen = torch.Generator().manual_seed(a.seed)

    data = torch.load(a.buffer, map_location="cpu")
    meta = data.get("metadata", {})
    assert not meta.get("uses_original_demonstrations", False)
    assert not meta.get("uses_expert_actions", False)
    states, noises, endpoints = data["states"].float(), data["noises"].float(), data["teacher_endpoints"].float()
    episode_ids = data["episode_ids"].long()
    # Deterministic episode-level split prevents state-window leakage.
    # Episode IDs are contiguous; offset-by-seed modulo split is deterministic and
    # guarantees held-out episodes (the previous LCG modulo 10 was degenerate).
    validation_episode = (episode_ids % 10) == (a.seed % 10)
    train_indices = torch.where(~validation_episode)[0]
    validation_indices = torch.where(validation_episode)[0]

    teacher, _, teacher_meta = load_deployed_teacher(a.bundle_dir)
    teacher.eval()
    model = build_flow(4, 72, 4, "cuda", 16).train()
    model.load_state_dict(teacher.state_dict())
    logits = torch.zeros(4, device="cuda")
    model.model.blocks = HardTopKGatedBlocks(model.model.blocks, logits, topk=3, temperature=2.0).cuda()
    gate = model.model.blocks.logits
    weights = [parameter for name, parameter in model.named_parameters() if "model.blocks.logits" not in name]
    weight_optimizer = torch.optim.AdamW(weights, lr=a.weight_lr, weight_decay=1e-5)
    gate_optimizer = torch.optim.Adam([gate], lr=a.gate_lr)
    history = []

    for epoch in range(1, a.gate_epochs + 1):
        fraction = (epoch - 1) / max(1, a.gate_epochs - 1)
        model.model.blocks.temperature = 2.0 * (0.25 / 2.0) ** fraction
        gate.requires_grad_(False)
        inner_values = []
        for _ in range(a.inner_batches):
            batch = sample_batch(states, noises, endpoints, train_indices, a.batch_size, cpu_gen)
            inner_values.append(inner_weight_step(model, teacher, batch, weight_optimizer))
        gate.requires_grad_(True)
        for parameter in weights: parameter.requires_grad_(False)
        picked = validation_indices[torch.randint(len(validation_indices), (a.outer_groups,), generator=cpu_gen)]
        outer = outer_gate_step(model, teacher, states[picked].cuda(), a.objective,
                                a.outer_groups, a.samples_per_state, gate_optimizer)
        for parameter in weights: parameter.requires_grad_(True)
        row = {"epoch": epoch, "objective": a.objective,
               "velocity": float(torch.stack([x[0] for x in inner_values]).mean()),
               "inner_endpoint": float(torch.stack([x[1] for x in inner_values]).mean()),
               "temperature": model.model.blocks.temperature,
               "logits": gate.detach().cpu().tolist(),
               "probabilities": model.model.blocks.probabilities().detach().cpu().tolist(), **outer}
        history.append(row)
        if epoch % 10 == 0 or epoch == 1:
            print(json.dumps(row), flush=True)

    selected = sorted(gate.detach().topk(3).indices.cpu().tolist())
    ungated = build_flow(4, 72, 4, "cuda", 16)
    wrapped_state = model.state_dict()
    ungated_state = {key.replace("model.blocks.blocks.", "model.blocks."): value
                     for key, value in wrapped_state.items() if key != "model.blocks.logits"}
    ungated.load_state_dict(ungated_state, strict=True)
    compact = build_flow(3, 72, 4, "cuda", 16)
    initialize_student(ungated, compact, torch.eye(72), selected)
    torch.save(compact.state_dict(), a.output_dir / "selected_initial_flow.pth")
    summary = {"experiment": "recoverability-guided hard Top-K depth search",
               "objective": a.objective, "selected_layers": selected,
               "final_logits": gate.detach().cpu().tolist(),
               "train_windows": len(train_indices), "heldout_windows": len(validation_indices),
               "episode_disjoint_split": True, "uses_mode_labels": False,
               "uses_environment_rewards": False, "uses_original_demonstrations": False,
               "teacher": "FM-4x72-16", "target": "FM-3x72-16", "history": history}
    (a.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
