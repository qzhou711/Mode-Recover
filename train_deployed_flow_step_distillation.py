"""Demonstration-free one-step distillation of an already deployed Flow policy."""
import argparse, copy, json, math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agents.models.flow_matching.ctm import (
    centered_gram_loss, ctm_paths, freeze, pseudo_huber, update_ema,
)
from teacher_flow_deployment import build_flow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-dir", type=Path, required=True)
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", choices=("endpoint", "boundary_ctm"), required=True)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--embed-dim", type=int, default=72)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--teacher-steps", type=int, default=16)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--dsm-weight", type=float, default=0.1)
    p.add_argument("--relation-weight", type=float, default=0.0,
                   help="Same-state multi-noise endpoint geometry weight.")
    p.add_argument("--trajectory-weight", type=float, default=0.0,
                   help="Same-state multi-noise short-horizon geometry weight.")
    p.add_argument("--multi-noise", type=int, default=4)
    p.add_argument("--aux-state-groups", type=int, default=32)
    p.add_argument("--trajectory-times", type=str, default="0.25,0.5,0.75")
    p.add_argument("--save-epochs", type=str, default="100,250,500")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    trajectory_times = tuple(float(value) for value in a.trajectory_times.split(",") if value)
    if a.method != "endpoint" and (a.relation_weight > 0 or a.trajectory_weight > 0):
        p.error("mode-preserving auxiliary losses are defined for --method endpoint")
    if a.multi_noise < 2 and (a.relation_weight > 0 or a.trajectory_weight > 0):
        p.error("auxiliary losses require --multi-noise >= 2")
    if a.trajectory_weight > 0 and not trajectory_times:
        p.error("--trajectory-times cannot be empty when trajectory loss is enabled")
    if any(not 0.0 < value < 1.0 for value in trajectory_times):
        p.error("trajectory times must lie strictly between zero and one")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(a.buffer, map_location="cpu")
    metadata = payload["metadata"]
    assert not metadata["uses_original_demonstrations"]
    assert not metadata["uses_expert_actions"]
    states = payload["states"].float()
    noises = payload["noises"].float()
    loader = DataLoader(TensorDataset(states, noises), batch_size=a.batch_size,
                        shuffle=True, drop_last=True)

    teacher = build_flow(a.layers, a.embed_dim, a.heads, "cuda", a.teacher_steps)
    teacher.load_state_dict(torch.load(a.teacher_dir / "eval_best_flow.pth", map_location="cuda"), strict=True)
    teacher = freeze(teacher)
    student = build_flow(a.layers, a.embed_dim, a.heads, "cuda", 1)
    student.load_state_dict(teacher.state_dict(), strict=True)
    torch.save(student.state_dict(), a.output_dir / "initial_flow.pth")
    target = freeze(copy.deepcopy(student))
    student.train()
    optimizer = torch.optim.Adam(student.get_params(), lr=a.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs, eta_min=1e-6)
    milestones = {int(x) for x in a.save_epochs.split(",") if x}
    best, best_epoch, history = math.inf, -1, []

    for epoch in range(a.epochs):
        losses, primary_values, dsm_values = [], [], []
        relation_values, trajectory_values = [], []
        for batch_index, (state, stored_noise) in enumerate(loader):
            if a.max_batches and batch_index >= a.max_batches:
                break
            state = state.cuda(); stored_noise = stored_noise.cuda()
            # Teacher endpoint is queried online from the actual keep013 teacher.
            with torch.no_grad():
                teacher_endpoint = teacher.integrate(stored_noise, state, steps=a.teacher_steps)
            zero = torch.zeros(len(state), device="cuda")
            one = torch.ones_like(zero)
            if a.method == "endpoint":
                prediction = student.boundary_transition(stored_noise, zero, one, state)
                primary = pseudo_huber(prediction, teacher_endpoint, .01)
                dsm = primary * 0.0
                relation = primary * 0.0
                trajectory = primary * 0.0
                if a.relation_weight > 0 or a.trajectory_weight > 0:
                    groups = min(a.aux_state_groups, len(state))
                    shared_state = state[:groups].repeat_interleave(a.multi_noise, dim=0)
                    auxiliary_noise = torch.randn(
                        groups * a.multi_noise, *stored_noise.shape[1:],
                        device=stored_noise.device, dtype=stored_noise.dtype,
                    )
                    auxiliary_zero = torch.zeros(len(auxiliary_noise), device="cuda")
                    auxiliary_one = torch.ones_like(auxiliary_zero)
                    if a.relation_weight > 0:
                        student_aux_endpoint = student.boundary_transition(
                            auxiliary_noise, auxiliary_zero, auxiliary_one, shared_state)
                        with torch.no_grad():
                            teacher_aux_endpoint = teacher.integrate(
                                auxiliary_noise, shared_state, steps=a.teacher_steps)
                        relation = centered_gram_loss(
                            student_aux_endpoint, teacher_aux_endpoint, a.multi_noise)
                    if a.trajectory_weight > 0:
                        horizon = trajectory_times[int(torch.randint(len(trajectory_times), ()).item())]
                        stop = torch.full_like(auxiliary_zero, horizon)
                        student_waypoint = student.boundary_transition(
                            auxiliary_noise, auxiliary_zero, stop, shared_state)
                        with torch.no_grad():
                            teacher_waypoint = teacher.integrate(
                                auxiliary_noise, shared_state, start_time=0.0,
                                end_time=horizon,
                                steps=max(1, round(a.teacher_steps * horizon)),
                            )
                        trajectory = centered_gram_loss(
                            student_waypoint, teacher_waypoint, a.multi_noise)
                loss = primary + a.relation_weight * relation + a.trajectory_weight * trajectory
            else:
                fresh_noise = torch.randn_like(teacher_endpoint)
                ti = int(torch.randint(0, 15, ()).item())
                si = int(torch.randint(ti + 2, 17, ()).item())
                prediction, reference, x_t, t = ctm_paths(
                    student, target, teacher, teacher_endpoint, state, fresh_noise,
                    (ti, si), 16,
                )
                primary = pseudo_huber(prediction, reference, .01)
                denoised = student.boundary_transition(x_t, t, torch.ones_like(t), state)
                dsm = pseudo_huber(denoised, teacher_endpoint, .01)
                loss = primary + a.dsm_weight * dsm
                relation = primary * 0.0
                trajectory = primary * 0.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step(); update_ema(target, student, .995)
            losses.append(float(loss.detach())); primary_values.append(float(primary.detach()))
            dsm_values.append(float(dsm.detach()))
            relation_values.append(float(relation.detach()))
            trajectory_values.append(float(trajectory.detach()))
        scheduler.step()
        score = float(np.mean(losses))
        record = {"epoch": epoch + 1, "selection_loss": score,
                  "primary_loss": float(np.mean(primary_values)),
                  "dsm_loss": float(np.mean(dsm_values)),
                  "relation_loss": float(np.mean(relation_values)),
                  "trajectory_loss": float(np.mean(trajectory_values))}
        history.append(record)
        if score < best:
            best, best_epoch = score, epoch + 1
            torch.save(target.state_dict(), a.output_dir / "eval_best_flow.pth")
        if epoch + 1 in milestones:
            out = a.output_dir / "checkpoints" / f"epoch_{epoch+1:04d}"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(target.state_dict(), out / "eval_best_flow.pth")
        if epoch % 25 == 0 or epoch + 1 == a.epochs:
            print(json.dumps(record), flush=True)
    torch.save(target.state_dict(), a.output_dir / "last_flow.pth")
    summary = {
        "method": a.method, "teacher": str(a.teacher_dir),
        "student_architecture": {"layers": a.layers, "embed_dim": a.embed_dim, "heads": a.heads},
        "teacher_steps": a.teacher_steps, "student_steps": 1,
        "initialization": "teacher checkpoint", "epochs": a.epochs,
        "best_epoch": best_epoch, "best_selection_loss": best,
        "buffer": str(a.buffer), "teacher_endpoint_source": "online keep013 query",
        "relation_weight": a.relation_weight, "trajectory_weight": a.trajectory_weight,
        "multi_noise": a.multi_noise, "aux_state_groups": a.aux_state_groups,
        "trajectory_times": trajectory_times,
        "auxiliary_protocol": "same state, fresh noises, online self-consistent teacher queries",
        "uses_original_demonstrations": False, "uses_expert_actions": False,
        "history": history,
    }
    (a.output_dir / "distillation_metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
