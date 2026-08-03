"""LightDP-style learnable depth pruning adapted to demo-free Flow policies.

This is a literature-aligned comparison, not an exact reproduction of LightDP:
the original method trains on demonstrations and distils a diffusion policy,
whereas this implementation keeps our teacher-rollout-only Flow protocol fixed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.models.diffusion.ema import ExponentialMovingAverage
from teacher_flow_deployment import build_flow, load_deployed_teacher
from train_flow_progressive_compression import initialize_student
from train_teacher_generated_flow_v2 import differentiable_integrate, save_ema


class GatedBlocks(nn.Module):
    """Straight-through Gumbel-Sigmoid residual block gates."""

    def __init__(self, blocks, logits, temperature=1.0):
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))
        self.logits = nn.Parameter(logits)
        self.temperature = temperature

    def probabilities(self):
        return self.logits.sigmoid()

    def forward(self, x):
        for index, block in enumerate(self.blocks):
            if self.training:
                u = torch.rand((), device=x.device).clamp_(1e-6, 1 - 1e-6)
                soft = torch.sigmoid((self.logits[index] + torch.log(u) - torch.log1p(-u)) / self.temperature)
                gate = (soft >= 0.5).to(soft.dtype).detach() - soft.detach() + soft
            else:
                gate = self.probabilities()[index]
            updated = block(x)
            x = x + gate * (updated - x)
        return x


@torch.no_grad()
def svd_importance(blocks, rank_ratio):
    """Aggregate truncated-SVD reconstruction error over block linear maps."""
    scores = []
    for block in blocks:
        score = torch.zeros((), device=next(block.parameters()).device)
        for parameter in block.parameters():
            if parameter.ndim != 2:
                continue
            singular = torch.linalg.svdvals(parameter.float())
            rank = max(1, min(len(singular) - 1, round(len(singular) * rank_ratio)))
            score += singular[rank:].square().sum().sqrt()
        scores.append(score)
    return torch.stack(scores)


def probabilities_with_budget(scores, target_depth):
    normalized = (scores - scores.mean()) / scores.std().clamp_min(1e-6)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        shift = (lo + hi) / 2
        if torch.sigmoid(normalized + shift).sum() < target_depth:
            lo = shift
        else:
            hi = shift
    probabilities = torch.sigmoid(normalized + (lo + hi) / 2).clamp(1e-4, 1 - 1e-4)
    return probabilities


def train_epoch(model, teacher, loader, optimizer, ema, max_batches, endpoint_weight,
                gate_budget_weight=0.0, target_depth=3):
    values = []
    for batch_index, (state, noise, endpoint) in enumerate(loader):
        if batch_index >= max_batches:
            break
        state, noise, endpoint = state.cuda(), noise.cuda(), endpoint.cuda()
        time = float(torch.rand(()).clamp(0.02, 0.98))
        time_vector = torch.full((len(state),), time, device="cuda")
        with torch.no_grad():
            x_t = teacher.integrate(noise, state, start_time=0.0, end_time=time,
                                    steps=max(1, round(16 * time)))
            target_velocity = teacher.velocity(x_t, time_vector, state)
        velocity = F.mse_loss(model.velocity(x_t, time_vector, state), target_velocity)
        endpoint_loss = F.mse_loss(differentiable_integrate(model, noise, state, steps=16), endpoint)
        budget = torch.zeros((), device="cuda")
        if isinstance(model.model.blocks, GatedBlocks):
            budget = (model.model.blocks.probabilities().sum() - target_depth).square()
        loss = velocity + endpoint_weight * endpoint_loss + gate_budget_weight * budget
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_params(), 1.0)
        optimizer.step()
        ema.update(model.get_params())
        values.append((float(velocity.detach()), float(endpoint_loss.detach()), float(budget.detach())))
    return np.asarray(values).mean(0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-epochs", type=int, default=250)
    parser.add_argument("--repair-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gate-learning-rate", type=float, default=3e-4)
    parser.add_argument("--endpoint-weight", type=float, default=0.03)
    parser.add_argument("--budget-weight", type=float, default=0.1)
    parser.add_argument("--temperature-start", type=float, default=2.0)
    parser.add_argument("--temperature-end", type=float, default=0.3)
    parser.add_argument("--svd-rank-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data = torch.load(args.buffer, map_location="cpu")
    assert not data["metadata"]["uses_original_demonstrations"]
    assert not data["metadata"]["uses_expert_actions"]
    keep = data["successes"].bool()[data["episode_ids"].long()]
    dataset = TensorDataset(data["states"].float()[keep], data["noises"].float()[keep],
                            data["teacher_endpoints"].float()[keep])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))
    teacher, _, metadata = load_deployed_teacher(args.bundle_dir)
    assert (metadata["teacher_layers"], metadata["teacher_embed_dim"], metadata["teacher_heads"],
            metadata["teacher_steps"]) == (4, 72, 4, 16)

    gated = build_flow(4, 72, 4, "cuda", 16).train()
    gated.load_state_dict(teacher.state_dict())
    scores = svd_importance(gated.model.blocks, args.svd_rank_ratio)
    initial_p = probabilities_with_budget(scores, 3)
    logits = torch.logit(initial_p)
    gated.model.blocks = GatedBlocks(gated.model.blocks, logits, args.temperature_start).cuda()
    gate_parameters = [gated.model.blocks.logits]
    weight_parameters = [p for name, p in gated.named_parameters() if name != "model.blocks.logits"]
    optimizer = torch.optim.AdamW([
        {"params": weight_parameters, "lr": args.learning_rate},
        {"params": gate_parameters, "lr": args.gate_learning_rate, "weight_decay": 0.0},
    ], weight_decay=1e-5)
    ema = ExponentialMovingAverage(gated.get_params(), 0.995, "cuda")
    history = []
    for epoch in range(1, args.gate_epochs + 1):
        fraction = (epoch - 1) / max(1, args.gate_epochs - 1)
        gated.model.blocks.temperature = args.temperature_start * (args.temperature_end / args.temperature_start) ** fraction
        velocity, endpoint, budget = train_epoch(
            gated, teacher, loader, optimizer, ema, args.max_batches, args.endpoint_weight,
            args.budget_weight, 3)
        row = {"stage": "gate", "epoch": epoch, "velocity_loss": velocity,
               "endpoint_loss": endpoint, "budget_loss": budget,
               "probabilities": gated.model.blocks.probabilities().detach().cpu().tolist()}
        history.append(row)
        if epoch % 25 == 0: print(json.dumps(row), flush=True)

    final_p = gated.model.blocks.probabilities().detach()
    selected = sorted(torch.topk(final_p, 3).indices.cpu().tolist())
    compact = build_flow(3, 72, 4, "cuda", 16).train()
    # Copy the jointly adapted gated model into the selected compact architecture.
    ungated = build_flow(4, 72, 4, "cuda", 16)
    state = gated.state_dict()
    ungated_state = {k.replace("model.blocks.blocks.", "model.blocks."): v
                     for k, v in state.items() if k != "model.blocks.logits"}
    ungated.load_state_dict(ungated_state, strict=True)
    initialize_student(ungated, compact, torch.eye(72), selected)
    torch.save(compact.state_dict(), args.output_dir / "selected_initial_flow.pth")
    optimizer = torch.optim.AdamW(compact.get_params(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.repair_epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(compact.get_params(), 0.995, "cuda")
    for epoch in range(1, args.repair_epochs + 1):
        velocity, endpoint, _ = train_epoch(
            compact, teacher, loader, optimizer, ema, args.max_batches, args.endpoint_weight)
        scheduler.step()
        row = {"stage": "repair", "epoch": epoch, "velocity_loss": velocity,
               "endpoint_loss": endpoint}
        history.append(row)
        if epoch in {50, 100, 250, 500, args.repair_epochs}:
            save_ema(compact, ema, args.output_dir / f"pretrain_epoch_{epoch:04d}.pth")
        if epoch % 25 == 0: print(json.dumps(row), flush=True)

    summary = {
        "experiment": "LightDP-style learnable depth pruning, demo-free Flow adaptation",
        "exact_official_reproduction": False,
        "paper_difference": "teacher rollout replaces original demonstrations; Flow replaces DDPM",
        "teacher": "FM-4x72-16", "student": "FM-3x72-16", "selected_layers": selected,
        "svd_scores": scores.cpu().tolist(), "initial_probabilities": initial_p.cpu().tolist(),
        "final_probabilities": final_p.cpu().tolist(), "teacher_buffer_samples": len(dataset),
        "uses_original_demonstrations": False, "uses_expert_actions": False,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
