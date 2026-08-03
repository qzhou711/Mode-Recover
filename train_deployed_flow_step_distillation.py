"""Demonstration-free one-step distillation of an already deployed Flow policy."""
import argparse, copy, json, math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agents.models.flow_matching.ctm import ctm_paths, freeze, pseudo_huber, update_ema
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
    p.add_argument("--save-epochs", type=str, default="100,250,500")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
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
                loss = primary
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step(); update_ema(target, student, .995)
            losses.append(float(loss.detach())); primary_values.append(float(primary.detach()))
            dsm_values.append(float(dsm.detach()))
        scheduler.step()
        score = float(np.mean(losses))
        record = {"epoch": epoch + 1, "selection_loss": score,
                  "primary_loss": float(np.mean(primary_values)),
                  "dsm_loss": float(np.mean(dsm_values))}
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
        "uses_original_demonstrations": False, "uses_expert_actions": False,
        "history": history,
    }
    (a.output_dir / "distillation_metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
