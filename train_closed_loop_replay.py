"""Demonstration-free Teacher replay with closed-loop Student preservation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agents.models.diffusion.ema import ExponentialMovingAverage
from teacher_flow_deployment import build_flow


def pseudo_huber_samples(prediction, target):
    value = torch.sqrt((prediction - target).square() + 0.01 ** 2) - 0.01
    return value.flatten(1).mean(1)


def predict(model, state, noise):
    zero = torch.zeros(len(state), device=state.device)
    return model.boundary_transition(noise, zero, torch.ones_like(zero), state)


def episode_groups(episode_ids, eligible):
    indices = torch.where(eligible)[0]
    order = torch.argsort(episode_ids[indices])
    indices = indices[order]
    episodes, counts = torch.unique_consecutive(
        episode_ids[indices], return_counts=True)
    chunks = torch.split(indices, counts.tolist())
    return {int(episode): chunk for episode, chunk in zip(episodes, chunks)}


def sample_episode_uniform(groups, count, generator):
    episodes = list(groups)
    selected = torch.randint(len(episodes), (count,), generator=generator)
    output = []
    for value in selected:
        candidates = groups[episodes[int(value)]]
        output.append(candidates[torch.randint(len(candidates), (), generator=generator)])
    return torch.stack(output)


def sample_latent_episode_uniform(latent_groups, count, generator):
    latents = list(latent_groups)
    selected = torch.randint(len(latents), (count,), generator=generator)
    output = []
    for value in selected:
        groups = latent_groups[latents[int(value)]]
        episodes = list(groups)
        episode = episodes[int(torch.randint(len(episodes), (), generator=generator))]
        candidates = groups[episode]
        output.append(candidates[torch.randint(len(candidates), (), generator=generator)])
    return torch.stack(output)


def save_ema(model, ema, path):
    ema.store(model.get_params())
    ema.copy_to(model.get_params())
    torch.save(model.state_dict(), path)
    ema.restore(model.get_params())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-buffer", type=Path, required=True)
    parser.add_argument("--discovery", type=Path)
    parser.add_argument("--student-buffer", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-sampling", choices=("natural", "balanced"), required=True)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--correction-weight", type=float, default=0.0)
    parser.add_argument("--pcgrad", action="store_true")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batches-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-epochs", default="50,100,250")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    teacher = torch.load(args.teacher_buffer, map_location="cpu")
    student = torch.load(args.student_buffer, map_location="cpu")
    assert not teacher["metadata"]["uses_original_demonstrations"]
    assert not teacher["metadata"]["uses_expert_actions"]
    assert not student["metadata"]["uses_original_demonstrations"]
    teacher_episode_ids = teacher["episode_ids"].long()
    teacher_success = teacher["successes"][teacher_episode_ids].bool()
    if args.replay_sampling == "balanced":
        if args.discovery is None:
            parser.error("--discovery is required for balanced replay")
        discovery = np.load(args.discovery)
        sample_latents = torch.as_tensor(discovery["sample_latents"]).long()
        assert len(sample_latents) == len(teacher["states"])
    else:
        sample_latents = torch.full((len(teacher["states"]),), -1, dtype=torch.long)
    replay_eligible = teacher_success & ((sample_latents >= 0) if args.replay_sampling == "balanced" else True)
    replay_indices = torch.where(replay_eligible)[0]
    replay_by_latent = {}
    if args.replay_sampling == "balanced":
        for latent in range(24):
            eligible = replay_eligible & (sample_latents == latent)
            groups = episode_groups(teacher_episode_ids, eligible)
            if groups:
                replay_by_latent[latent] = groups
        assert len(replay_by_latent) == 24

    student_episode_ids = student["episode_ids"].long()
    unique_student = torch.unique(student_episode_ids)
    row = {int(episode): index for index, episode in enumerate(unique_student)}
    sample_rows = torch.tensor([row[int(episode)] for episode in student_episode_ids])
    student_success = student["student_successes"][sample_rows].bool()
    paired_teacher_success = student["teacher_successes"][sample_rows].bool()
    anchor_groups = episode_groups(student_episode_ids, student_success)
    correction_groups = episode_groups(
        student_episode_ids, (~student_success) & paired_teacher_success)
    assert anchor_groups and correction_groups

    model = build_flow(3, 48, 3, "cuda", 1)
    model.load_state_dict(torch.load(args.student, map_location="cuda"), strict=True)
    model.train()
    parameters = list(model.get_params())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(parameters, 0.995, "cuda")
    milestones = {int(value) for value in args.save_epochs.split(",")}
    history = []

    for epoch in range(1, args.epochs + 1):
        records = []
        for _ in range(args.batches_per_epoch):
            if args.replay_sampling == "balanced":
                replay = sample_latent_episode_uniform(
                    replay_by_latent, args.batch_size, generator)
            else:
                replay = replay_indices[torch.randint(
                    len(replay_indices), (args.batch_size,), generator=generator)]
            replay_prediction = predict(
                model, teacher["states"][replay].float().cuda(),
                teacher["noises"][replay].float().cuda())
            replay_loss = pseudo_huber_samples(
                replay_prediction, teacher["teacher_endpoints"][replay].float().cuda()).mean()
            teacher_loss = replay_loss
            correction_loss = replay_loss * 0
            if args.correction_weight > 0:
                correction = sample_episode_uniform(
                    correction_groups, args.batch_size, generator)
                correction_loss = pseudo_huber_samples(
                    predict(model, student["states"][correction].float().cuda(),
                            student["noises"][correction].float().cuda()),
                    student["teacher_corrections"][correction].float().cuda()).mean()
                teacher_loss = teacher_loss + args.correction_weight * correction_loss
            anchor_loss = replay_loss * 0
            if args.anchor_weight > 0:
                anchor = sample_episode_uniform(anchor_groups, args.batch_size, generator)
                anchor_loss = pseudo_huber_samples(
                    predict(model, student["states"][anchor].float().cuda(),
                            student["noises"][anchor].float().cuda()),
                    student["student_endpoints"][anchor].float().cuda()).mean()

            optimizer.zero_grad(set_to_none=True)
            conflict = 0.0
            if args.pcgrad and args.anchor_weight > 0:
                teacher_gradients = torch.autograd.grad(
                    teacher_loss, parameters, retain_graph=True, allow_unused=True)
                anchor_gradients = torch.autograd.grad(
                    anchor_loss, parameters, allow_unused=True)
                teacher_gradients = [gradient if gradient is not None else torch.zeros_like(parameter)
                                     for gradient, parameter in zip(teacher_gradients, parameters)]
                anchor_gradients = [gradient if gradient is not None else torch.zeros_like(parameter)
                                    for gradient, parameter in zip(anchor_gradients, parameters)]
                dot = sum((left * right).sum() for left, right in
                          zip(teacher_gradients, anchor_gradients))
                anchor_norm = sum((gradient * gradient).sum()
                                  for gradient in anchor_gradients).clamp_min(1e-12)
                coefficient = torch.minimum(dot / anchor_norm, torch.zeros_like(dot))
                conflict = float((dot < 0).float())
                for parameter, teacher_gradient, anchor_gradient in zip(
                        parameters, teacher_gradients, anchor_gradients):
                    parameter.grad = (teacher_gradient - coefficient * anchor_gradient
                                      + args.anchor_weight * anchor_gradient)
                total_loss = teacher_loss + args.anchor_weight * anchor_loss
            else:
                total_loss = teacher_loss + args.anchor_weight * anchor_loss
                total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            ema.update(parameters)
            records.append((float(replay_loss.detach()), float(correction_loss.detach()),
                            float(anchor_loss.detach()), float(total_loss.detach()), conflict))
        scheduler.step()
        values = np.asarray(records)
        record = {"epoch": epoch, "replay_loss": float(values[:, 0].mean()),
                  "correction_loss": float(values[:, 1].mean()),
                  "anchor_loss": float(values[:, 2].mean()),
                  "total_loss": float(values[:, 3].mean()),
                  "gradient_conflict_rate": float(values[:, 4].mean())}
        history.append(record)
        if epoch in milestones:
            output = args.output_dir / "checkpoints" / f"epoch_{epoch:04d}"
            output.mkdir(parents=True, exist_ok=True)
            save_ema(model, ema, output / "eval_best_flow.pth")
        if epoch % 25 == 0:
            print(json.dumps(record), flush=True)
    save_ema(model, ema, args.output_dir / "eval_best_flow.pth")
    summary = {
        "demonstration_free": True,
        "uses_original_demonstrations": False,
        "uses_expert_actions": False,
        "teacher_buffer": str(args.teacher_buffer),
        "student_buffer": str(args.student_buffer),
        "discovery": str(args.discovery) if args.discovery is not None else None,
        "replay_sampling": args.replay_sampling,
        "anchor_weight": args.anchor_weight,
        "correction_weight": args.correction_weight,
        "pcgrad": args.pcgrad,
        "teacher_replay_latents": len(replay_by_latent) if replay_by_latent else None,
        "teacher_replay_samples": len(replay_indices),
        "student_anchor_episodes": len(anchor_groups),
        "student_correction_episodes": len(correction_groups),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
