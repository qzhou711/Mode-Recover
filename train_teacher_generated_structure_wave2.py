"""Second-wave demonstration-free structure transfer experiments."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.models.diffusion.ema import ExponentialMovingAverage
from teacher_flow_deployment import build_flow, load_deployed_teacher
from train_teacher_generated_flow_v2 import (
    activation_matrix,
    cross_noise_relation_loss,
    init_student,
    relation_forward,
    relation_loss,
    save_ema,
)


def copy_same_width(source, target, layer_map):
    source_state = source.state_dict()
    mapped = {}
    for key in target.state_dict():
        source_key = key
        if ".blocks." in key:
            prefix, remainder = key.split(".blocks.", 1)
            block, suffix = remainder.split(".", 1)
            source_key = f"{prefix}.blocks.{layer_map[int(block)]}.{suffix}"
        mapped[key] = source_state[source_key].detach().clone()
    target.load_state_dict(mapped, strict=True)


def set_progressive_trainability(model, active_blocks):
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def mask_inactive_block_gradients(model, active_blocks):
    for index, block in enumerate(model.model.blocks):
        if index not in active_blocks:
            for parameter in block.parameters():
                parameter.grad = None


def train_stage(
    source, student, loader, epochs, max_batches, output, progressive=False,
    source_layers=None, multi_noise=1, cross_noise_weight=0.0,
):
    optimizer = torch.optim.Adam(student.get_params(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(student.get_params(), 0.995, "cuda")
    best = math.inf
    history = []
    for epoch in range(epochs):
        if progressive:
            third = max(1, epochs // 3)
            active = [2] if epoch < third else [1, 2] if epoch < 2 * third else [0, 1, 2]
            set_progressive_trainability(student, active)
        else:
            active = [0, 1, 2, 3]
        losses = []
        for batch_index, (state, noise) in enumerate(loader):
            if batch_index >= max_batches:
                break
            state = state.cuda()
            noise = noise.cuda()
            if multi_noise > 1:
                base = max(1, len(state) // multi_noise)
                state = state[:base].repeat_interleave(multi_noise, 0)
                noise = torch.randn(len(state), *noise.shape[1:], device="cuda")
            time = float(torch.rand(()).clamp(0.02, 0.98))
            time_vector = torch.full((len(state),), time, device="cuda")
            with torch.no_grad():
                x_t = source.integrate(
                    noise, state, start_time=0.0, end_time=time,
                    steps=max(1, round(16 * time)),
                )
                mapped_layers = source_layers or list(range(len(student.model.blocks)))
                target, source_relations = relation_forward(
                    source, x_t, time_vector, state, mapped_layers,
                    full_minilm=True,
                )
            prediction, student_relations = relation_forward(
                student, x_t, time_vector, state, list(range(len(student.model.blocks))),
                full_minilm=True,
            )
            velocity = F.mse_loss(prediction, target)
            relations = relation_loss(student_relations, source_relations, full_minilm=True)
            cross_noise = (
                cross_noise_relation_loss(student_relations, source_relations, multi_noise)
                if multi_noise > 1 else velocity * 0
            )
            loss = relations + cross_noise_weight * cross_noise + 0.1 * velocity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if progressive:
                mask_inactive_block_gradients(student, active)
            torch.nn.utils.clip_grad_norm_(student.get_params(), 1.0)
            optimizer.step()
            ema.update(student.get_params())
            losses.append(float(loss.detach()))
        scheduler.step()
        score = float(np.mean(losses))
        history.append({
            "epoch": epoch, "loss": score, "active_blocks": active,
            "multi_noise": multi_noise, "cross_noise_weight": cross_noise_weight,
        })
        if score < best:
            best = score
            save_ema(student, ema, output)
        if epoch % 25 == 0:
            print(json.dumps(history[-1]), flush=True)
    student.load_state_dict(torch.load(output, map_location="cuda"), strict=True)
    return history, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["progressive_replacement", "teacher_assistant"], required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--assistant-epochs", type=int, default=None)
    parser.add_argument("--student-epochs", type=int, default=None)
    parser.add_argument("--assistant-multi-noise", type=int, default=1)
    parser.add_argument("--assistant-cross-noise-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.buffer, map_location="cpu")
    assert not data["metadata"]["uses_original_demonstrations"]
    assert not data["metadata"]["uses_expert_actions"]
    states = data["states"].float()
    noises = data["noises"].float()
    loader = DataLoader(
        TensorDataset(states, noises), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    teacher, _, _ = load_deployed_teacher(args.bundle_dir)
    activations = activation_matrix(teacher, states, noises, 8, args.batch_size)

    if args.method == "progressive_replacement":
        student = build_flow(3, 48, 3, "cuda", 16).train()
        initialization = init_student(teacher, student, "early", activations, 3, 3)
        history, best = train_stage(
            teacher, student, loader, args.epochs, args.max_batches,
            args.output_dir / "eval_best_flow.pth", progressive=True,
        )
        stages = [{"source": "FM-4x72-16", "target": "FM-3x48-16", "epochs": args.epochs}]
    else:
        assistant = build_flow(4, 48, 4, "cuda", 16).train()
        assistant_initialization = init_student(
            teacher, assistant, "width", activations, 4, 4
        )
        assistant_epochs = args.assistant_epochs if args.assistant_epochs is not None else args.epochs // 2
        student_epochs = args.student_epochs if args.student_epochs is not None else args.epochs - assistant_epochs
        assistant_history, assistant_best = train_stage(
            teacher, assistant, loader, assistant_epochs, args.max_batches,
            args.output_dir / "assistant_best_flow.pth",
            multi_noise=args.assistant_multi_noise,
            cross_noise_weight=args.assistant_cross_noise_weight,
        )
        student = build_flow(3, 48, 3, "cuda", 16).train()
        copy_same_width(assistant, student, [0, 2, 3])
        history, best = train_stage(
            assistant, student, loader, student_epochs,
            args.max_batches, args.output_dir / "eval_best_flow.pth",
            source_layers=[0, 2, 3],
        )
        initialization = {
            "method": "teacher_assistant",
            "assistant_initialization": assistant_initialization,
            "student_layer_map": [0, 2, 3],
            "assistant_multi_noise": args.assistant_multi_noise,
            "assistant_cross_noise_weight": args.assistant_cross_noise_weight,
        }
        stages = [
            {"source": "FM-4x72-16", "target": "FM-4x48-16", "epochs": assistant_epochs, "best_loss": assistant_best},
            {"source": "FM-4x48-16", "target": "FM-3x48-16", "epochs": student_epochs},
        ]
        history = {"assistant": assistant_history, "student": history}

    metrics = {
        "method": args.method,
        "student_architecture": {"layers": 3, "embed_dim": 48, "heads": 3},
        "initialization": initialization,
        "stages": stages,
        "best_selection_loss": best,
        "history": history,
        "uses_original_demonstrations": False,
        "uses_expert_actions": False,
        "buffer": str(args.buffer),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
