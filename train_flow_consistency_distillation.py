"""Cross-time consistency distillation for compressed Avoiding Flow policies."""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import trange

from distill_flow_matching_avoiding import (
    conditional_pairwise_geometry,
    make_agent,
    make_student,
    repeat_conditions,
)
from train_flow_compression_stage1 import calibration_activations
from train_flow_progressive_compression import initialize_student, selection_basis


def consistency_endpoint(model, x_t, time, state):
    shape = (time.shape[0],) + (1,) * (x_t.ndim - 1)
    return x_t + (1.0 - time.reshape(shape)) * model.velocity(x_t, time, state)


@torch.no_grad()
def teacher_pair(teacher, noise, state, start, end, teacher_steps):
    if start == 0.0:
        x_start = noise
    else:
        x_start = teacher.integrate(
            noise, state, start_time=0.0, end_time=start,
            steps=max(1, int(round(teacher_steps * start))),
        )
    x_end = teacher.integrate(
        x_start, state, start_time=start, end_time=end,
        steps=max(1, int(round(teacher_steps * (end - start)))),
    )
    return x_start, x_end


def pseudo_huber(prediction, target, delta=0.01):
    error = prediction - target
    return (torch.sqrt(error.square() + delta * delta) - delta).mean()


@torch.no_grad()
def update_target(target, online, decay):
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters()
    ):
        target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)


def initialize(args, teacher, student, agent):
    metadata = {"kind": args.student_init, "load_max_abs_diff": None}
    if args.student_init == "random":
        return metadata
    if args.student_init in {"pointwise", "full"}:
        checkpoint = (
            args.pointwise_dir if args.student_init == "pointwise" else args.full_dir
        ) / "eval_best_flow.pth"
        state = torch.load(checkpoint, map_location=agent.device)
        student.load_state_dict(state, strict=True)
        metadata.update(checkpoint=str(checkpoint), load_max_abs_diff=0.0)
        return metadata
    activations = calibration_activations(
        teacher, agent.train_dataloader, agent.scaler, args.calibration_batches
    )
    basis, details = selection_basis(activations, 3, 3)
    details["load_max_abs_diff"] = initialize_student(
        teacher, student, basis, details["teacher_layers"]
    )
    metadata.update(details)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--student-init",
        choices=["random", "teacher_derived", "pointwise", "full"],
        required=True,
    )
    parser.add_argument("--pointwise-dir", type=Path)
    parser.add_argument("--full-dir", type=Path)
    parser.add_argument("--student-layers", type=int, default=3)
    parser.add_argument("--student-embed-dim", type=int, default=48)
    parser.add_argument("--student-heads", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches-per-epoch", type=int, default=4)
    parser.add_argument("--teacher-steps", type=int, default=16)
    parser.add_argument("--time-intervals", type=int, default=16)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--conditional-samples", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-decay", type=float, default=0.995)
    parser.add_argument("--flow-weight", type=float, default=0.1)
    parser.add_argument("--distribution-weight", type=float, default=0.0)
    parser.add_argument("--ddil-weight", type=float, default=0.0)
    parser.add_argument("--teacher-anchor-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.student_init == "pointwise" and args.pointwise_dir is None:
        parser.error("pointwise initialization requires --pointwise-dir")
    if args.student_init == "full" and args.full_dir is None:
        parser.error("full initialization requires --full-dir")
    if args.time_intervals < 2:
        parser.error("time intervals must be at least 2")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(mode="disabled")
    agent = make_agent(args.teacher_dir, args.batch_size, args.teacher_steps)
    teacher = agent.model.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = make_student(
        agent, 1, args.student_layers, args.student_embed_dim,
        args.student_heads, "random",
    ).to(agent.device)
    initialization = initialize(args, teacher, student, agent)
    target = copy.deepcopy(student).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    student.train()

    optimizer = torch.optim.Adam(student.get_params(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    validation_state, validation_action, _ = next(iter(agent.test_dataloader))
    validation_state = agent.scaler.scale_input(validation_state).float()[:64]
    validation_action = agent.scaler.scale_output(validation_action).float()[:64]
    validation_noise = torch.randn_like(validation_action)
    validation_pairs = []
    with torch.no_grad():
        for index in range(args.time_intervals):
            start, end = index / args.time_intervals, (index + 1) / args.time_intervals
            x_start, x_end = teacher_pair(
                teacher, validation_noise, validation_state, start, end,
                args.teacher_steps,
            )
            validation_pairs.append((start, end, x_start, x_end))

    best_score, best_epoch, history = math.inf, -1, []
    for epoch in trange(args.epochs, desc=f"Flow CD {args.student_init}"):
        losses, cd_values, flow_values = [], [], []
        distribution_values, ddil_values, anchor_values = [], [], []
        for batch_index, (state, action, _) in enumerate(agent.train_dataloader):
            if args.max_batches_per_epoch and batch_index >= args.max_batches_per_epoch:
                break
            state = agent.scaler.scale_input(state).float()
            action = agent.scaler.scale_output(action).float()
            state, action = repeat_conditions(
                state, action, args.conditional_samples
            )
            noise = torch.randn_like(action)
            index = int(torch.randint(0, args.time_intervals, ()).item())
            start, end = index / args.time_intervals, (index + 1) / args.time_intervals
            x_start, x_end = teacher_pair(
                teacher, noise, state, start, end, args.teacher_steps
            )
            t_start = torch.full(
                (state.shape[0],), start, device=state.device, dtype=state.dtype
            )
            t_end = torch.full_like(t_start, end)
            online_endpoint = consistency_endpoint(
                student, x_start, t_start, state
            )
            with torch.no_grad():
                target_endpoint = (
                    x_end if end == 1.0
                    else consistency_endpoint(target, x_end, t_end, state)
                )
                teacher_endpoint = teacher.integrate(
                    x_start, state, start_time=start, end_time=1.0,
                    steps=max(1, int(round(args.teacher_steps * (1.0 - start)))),
                )
            cd_loss = pseudo_huber(online_endpoint, target_endpoint)
            anchor_loss = pseudo_huber(online_endpoint, teacher_endpoint)
            flow_loss = student.loss(action, state)
            distribution_loss = conditional_pairwise_geometry(
                online_endpoint, teacher_endpoint, args.conditional_samples
            )
            ddil_loss = online_endpoint.sum() * 0.0
            if args.ddil_weight > 0 and start > 0:
                with torch.no_grad():
                    induced = student.integrate(
                        noise, state, start_time=0.0, end_time=start,
                        steps=max(1, int(round(args.time_intervals * start))),
                    )
                    induced_next = teacher.integrate(
                        induced, state, start_time=start, end_time=end, steps=1
                    )
                    induced_target = (
                        induced_next if end == 1.0
                        else consistency_endpoint(
                            target, induced_next, t_end, state
                        )
                    )
                ddil_loss = pseudo_huber(
                    consistency_endpoint(student, induced, t_start, state),
                    induced_target,
                )
            loss = (
                cd_loss
                + args.flow_weight * flow_loss
                + args.distribution_weight * distribution_loss
                + args.ddil_weight * ddil_loss
                + args.teacher_anchor_weight * anchor_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            update_target(target, student, args.target_decay)
            losses.append(float(loss.detach()))
            cd_values.append(float(cd_loss.detach()))
            flow_values.append(float(flow_loss.detach()))
            distribution_values.append(float(distribution_loss.detach()))
            ddil_values.append(float(ddil_loss.detach()))
            anchor_values.append(float(anchor_loss.detach()))
        scheduler.step()

        validation_values = []
        with torch.no_grad():
            for start, end, x_start, x_end in validation_pairs:
                t_start = torch.full(
                    (validation_state.shape[0],), start,
                    device=validation_state.device, dtype=validation_state.dtype,
                )
                t_end = torch.full_like(t_start, end)
                prediction = consistency_endpoint(
                    target, x_start, t_start, validation_state
                )
                reference = (
                    x_end if end == 1.0
                    else consistency_endpoint(
                        target, x_end, t_end, validation_state
                    )
                )
                validation_values.append(pseudo_huber(prediction, reference))
        validation_score = float(torch.stack(validation_values).mean())
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "consistency_loss": float(np.mean(cd_values)),
            "flow_loss": float(np.mean(flow_values)),
            "distribution_loss": float(np.mean(distribution_values)),
            "ddil_loss": float(np.mean(ddil_values)),
            "teacher_anchor_loss": float(np.mean(anchor_values)),
            "validation_consistency": validation_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        if validation_score < best_score:
            best_score, best_epoch = validation_score, epoch
            torch.save(target.state_dict(), args.output_dir / "eval_best_flow.pth")
        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(record), flush=True)

    torch.save(target.state_dict(), args.output_dir / "last_flow.pth")
    summary = {
        "method": "cross_time_consistency_distillation",
        "teacher_checkpoint": str(args.teacher_dir / "eval_best_flow.pth"),
        "student_architecture": {
            "layers": args.student_layers,
            "embed_dim": args.student_embed_dim,
            "heads": args.student_heads,
        },
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "student_steps": 1,
        "initialization": initialization,
        "epochs": args.epochs,
        "time_intervals": args.time_intervals,
        "best_epoch": best_epoch,
        "best_validation_consistency": best_score,
        "flow_weight": args.flow_weight,
        "distribution_weight": args.distribution_weight,
        "ddil_weight": args.ddil_weight,
        "teacher_anchor_weight": args.teacher_anchor_weight,
        "conditional_samples": args.conditional_samples,
        "history": history,
    }
    (args.output_dir / "consistency_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
