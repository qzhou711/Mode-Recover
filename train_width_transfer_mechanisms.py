"""Demo-free width-transfer mechanisms inspired by InDistill and PPCL."""
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
from teacher_flow_deployment import build_flow
from train_flow_compression_stage1 import block_diagonal_basis
from train_recoverable_width_compression import canonical_signs
from train_teacher_generated_flow_v2 import activation_matrix, differentiable_integrate, save_ema


METHODS = ("progressive_pca", "ppcl_adapter", "ffn_activation", "ffn_weight_saliency")


def pca_basis(activations, output_dim):
    centered = activations - activations.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)[:output_dim]
    basis = canonical_signs(vectors[:, order])
    fraction = float(values[order].clamp_min(0).sum() / values.clamp_min(0).sum().clamp_min(1e-12))
    return basis, fraction


def initialize_width(teacher, student, basis):
    basis = basis.to(next(teacher.parameters()).device)
    teacher_dim, student_dim = basis.shape
    bases = {
        teacher_dim: basis,
        2 * teacher_dim: block_diagonal_basis(basis, 2),
        4 * teacher_dim: block_diagonal_basis(basis, 4),
    }
    mapped = {}
    for key, target in student.state_dict().items():
        source = teacher.state_dict()[key]
        if source.shape == target.shape:
            value = source.detach().clone()
        elif key.endswith("pos_emb"):
            value = source @ basis
        elif source.ndim == 1:
            if ".ln" in key or key.endswith("ln_f.weight"):
                value = torch.ones_like(target)
            else:
                value = bases[source.shape[0]].T @ source
        elif source.ndim == 2:
            value = source
            if source.shape[0] in bases:
                value = bases[source.shape[0]].T @ value
            if source.shape[1] in bases:
                value = value @ bases[source.shape[1]]
        else:
            raise ValueError(f"unsupported width mapping for {key}: {source.shape}")
        if value.shape != target.shape:
            raise ValueError(f"width mapping mismatch for {key}: {value.shape} != {target.shape}")
        mapped[key] = value.detach().clone()
    student.load_state_dict(mapped, strict=True)
    return max(float((student.state_dict()[key] - value).abs().max()) for key, value in mapped.items())


@torch.no_grad()
def ffn_activation_energy(teacher, states, noises, batches, batch_size):
    energies = [torch.zeros(288) for _ in range(3)]
    counts = [0, 0, 0]
    captures = [[] for _ in range(3)]
    handles = []
    for index, block in enumerate(teacher.model.blocks):
        handles.append(block.mlp[1].register_forward_hook(
            lambda _m, _i, output, index=index: captures[index].append(output.detach().cpu())
        ))
    for offset in range(0, min(len(states), batches * batch_size), batch_size):
        state = states[offset:offset + batch_size].cuda()
        noise = noises[offset:offset + batch_size].cuda()
        time = torch.full((len(state),), 0.5, device="cuda")
        x_t = teacher.integrate(noise, state, start_time=0.0, end_time=0.5, steps=8)
        teacher.velocity(x_t, time, state)
    for handle in handles:
        handle.remove()
    for index, values in enumerate(captures):
        flat = torch.cat([value.reshape(-1, value.shape[-1]) for value in values])
        energies[index] = flat.square().mean(0)
        counts[index] = len(flat)
    return energies, counts


def initialize_ffn(teacher, student, method, states, noises, batch_size):
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    if method == "ffn_activation":
        energies, counts = ffn_activation_energy(teacher, states, noises, 8, batch_size)
    else:
        energies, counts = [], []
        for layer in range(3):
            w1 = teacher_state[f"model.blocks.{layer}.mlp.0.weight"].detach().cpu()
            w2 = teacher_state[f"model.blocks.{layer}.mlp.2.weight"].detach().cpu()
            energies.append(w1.square().sum(1).sqrt() * w2.square().sum(0).sqrt())
            counts.append(0)
    selected_by_layer = []
    mapped = {}
    for key, target in student_state.items():
        if ".mlp.0.weight" in key or ".mlp.0.bias" in key or ".mlp.2.weight" in key:
            layer = int(key.split(".blocks.")[1].split(".")[0])
            if len(selected_by_layer) <= layer:
                selected = torch.topk(energies[layer], 36).indices.sort().values
                selected_by_layer.append(selected)
            selected = selected_by_layer[layer].to(teacher_state[key].device)
            if ".mlp.0.weight" in key or ".mlp.0.bias" in key:
                value = teacher_state[key][selected]
            else:
                value = teacher_state[key][:, selected]
        else:
            value = teacher_state[key]
        if value.shape != target.shape:
            raise ValueError(f"FFN mapping mismatch for {key}: {value.shape} != {target.shape}")
        mapped[key] = value.detach().clone()
    student.load_state_dict(mapped, strict=True)
    return {"selected_channels": [x.tolist() for x in selected_by_layer], "activation_counts": counts}


def capture_block_outputs(flow, fn):
    outputs = []
    handles = [block.register_forward_hook(
        lambda _m, _i, output, index=index: outputs.append((index, output))
    ) for index, block in enumerate(flow.model.blocks)]
    result = fn()
    for handle in handles:
        handle.remove()
    return result, [value for _, value in sorted(outputs, key=lambda pair: pair[0])]


def train_stage(teacher, student, loader, epochs, endpoint_weight, learning_rate,
                output_dir, epoch_offset=0, adapters=None, feature_weight=0.0):
    parameters = list(student.get_params()) + ([] if adapters is None else list(adapters.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    ema = ExponentialMovingAverage(student.get_params(), 0.995, "cuda")
    history, best = [], math.inf
    for local_epoch in range(1, epochs + 1):
        velocity_values, endpoint_values, feature_values = [], [], []
        for batch_index, (state, noise, endpoint) in enumerate(loader):
            if batch_index >= 4:
                break
            state, noise, endpoint = state.cuda(), noise.cuda(), endpoint.cuda()
            time = float(torch.rand(()).clamp(0.02, 0.98))
            tv = torch.full((len(state),), time, device="cuda")
            with torch.no_grad():
                x_t = teacher.integrate(noise, state, start_time=0.0, end_time=time,
                                        steps=max(1, round(16 * time)))
                if adapters is None:
                    target = teacher.velocity(x_t, tv, state)
                    teacher_features = []
                else:
                    (target, teacher_features) = capture_block_outputs(
                        teacher, lambda: teacher.velocity(x_t, tv, state)
                    )
                    teacher_features = [value.detach() for value in teacher_features]
            if adapters is None:
                prediction = student.velocity(x_t, tv, state)
                student_features = []
            else:
                prediction, student_features = capture_block_outputs(
                    student, lambda: student.velocity(x_t, tv, state)
                )
            velocity_loss = F.mse_loss(prediction, target)
            endpoint_loss = F.mse_loss(differentiable_integrate(student, noise, state, 16), endpoint)
            feature_loss = velocity_loss * 0.0
            if adapters is not None:
                terms = []
                for adapter, student_feature, teacher_feature in zip(adapters, student_features, teacher_features):
                    aligned = adapter(student_feature)
                    normalized = F.mse_loss(F.normalize(aligned, dim=-1), F.normalize(teacher_feature, dim=-1))
                    scale = teacher_feature.detach().square().mean().sqrt().clamp_min(1e-6)
                    linear = F.mse_loss(aligned / scale, teacher_feature / scale)
                    terms.append(normalized + 0.1 * linear)
                feature_loss = torch.stack(terms).mean()
            loss = velocity_loss + endpoint_weight * endpoint_loss + feature_weight * feature_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            ema.update(student.get_params())
            velocity_values.append(float(velocity_loss.detach()))
            endpoint_values.append(float(endpoint_loss.detach()))
            feature_values.append(float(feature_loss.detach()))
        scheduler.step()
        total_epoch = epoch_offset + local_epoch
        score = float(np.mean(velocity_values) + endpoint_weight * np.mean(endpoint_values))
        row = {"epoch": total_epoch, "selection_loss": score,
               "velocity_loss": float(np.mean(velocity_values)),
               "endpoint_loss": float(np.mean(endpoint_values)),
               "feature_loss": float(np.mean(feature_values))}
        history.append(row)
        if score < best:
            best = score
            save_ema(student, ema, output_dir / "structure_best_flow.pth")
        if total_epoch in {50, 100, 250, 300, 350, 500}:
            save_ema(student, ema, output_dir / f"pretrain_epoch_{total_epoch:04d}.pth")
        if local_epoch % 25 == 0:
            print(json.dumps(row), flush=True)
    save_ema(student, ema, output_dir / f"pretrain_epoch_{epoch_offset + epochs:04d}.pth")
    return history


def load_buffer(path):
    data = torch.load(path, map_location="cpu")
    assert not data["metadata"]["uses_original_demonstrations"]
    assert not data["metadata"]["uses_expert_actions"]
    episode_ids = data["episode_ids"].long()
    unique_ids = torch.unique(episode_ids, sorted=True)
    dense_ids = torch.searchsorted(unique_ids, episode_ids)
    if len(unique_ids) != len(data["successes"]) or not torch.equal(unique_ids[dense_ids], episode_ids):
        raise ValueError("buffer episode alignment failed")
    keep = data["successes"].bool()[dense_ids]
    return data["states"].float()[keep], data["noises"].float()[keep], data["teacher_endpoints"].float()[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--endpoint-weight", type=float, default=0.03)
    parser.add_argument("--feature-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    metadata = torch.load(args.bundle_dir / "deployment_metadata.pt", map_location="cpu")
    teacher = build_flow(3, 72, 4, "cuda", 16)
    checkpoint = args.teacher if args.teacher.is_file() else args.teacher / "eval_best_flow.pth"
    teacher.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    teacher.min_action = metadata["y_bounds_tensor"][0].cuda()
    teacher.max_action = metadata["y_bounds_tensor"][1].cuda()
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    states, noises, endpoints = load_buffer(args.buffer)
    loader = DataLoader(TensorDataset(states, noises, endpoints), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed))
    details, history = {}, []
    if args.method == "progressive_pca":
        if args.epochs < 2 or args.epochs % 2:
            raise ValueError("progressive_pca requires an even --epochs >= 2")
        stage_epochs = args.epochs // 2
        activations = activation_matrix(teacher, states, noises, 8, args.batch_size)
        basis60, fraction60 = pca_basis(activations, 60)
        assistant = build_flow(3, 60, 4, "cuda", 16).train()
        initialize_width(teacher, assistant, basis60)
        stage1_dir = args.output_dir / "assistant60"
        stage1_dir.mkdir(exist_ok=True)
        history += train_stage(teacher, assistant, loader, stage_epochs, args.endpoint_weight,
                               args.learning_rate, stage1_dir)
        assistant.load_state_dict(torch.load(
            stage1_dir / f"pretrain_epoch_{stage_epochs:04d}.pth", map_location="cuda"
        ), strict=True)
        assistant.eval()
        for parameter in assistant.parameters(): parameter.requires_grad_(False)
        assistant.min_action, assistant.max_action = teacher.min_action, teacher.max_action
        activations60 = activation_matrix(assistant, states, noises, 8, args.batch_size)
        basis48, fraction48 = pca_basis(activations60, 48)
        student = build_flow(3, 48, 4, "cuda", 16).train()
        initialize_width(assistant, student, basis48)
        torch.save(student.state_dict(), args.output_dir / "initial_flow.pth")
        history += train_stage(assistant, student, loader, stage_epochs, args.endpoint_weight,
                               args.learning_rate, args.output_dir, epoch_offset=stage_epochs)
        details = {"path": "72->60->48", "explained_72_60": fraction60,
                   "explained_60_48": fraction48, "final_ffn_dim": 192}
    else:
        if args.method == "ppcl_adapter":
            activations = activation_matrix(teacher, states, noises, 8, args.batch_size)
            basis, fraction = pca_basis(activations, 48)
            student = build_flow(3, 48, 4, "cuda", 16).train()
            initialize_width(teacher, student, basis)
            adapters = nn.ModuleList([nn.Linear(48, 72, bias=False) for _ in range(3)]).cuda()
            for adapter in adapters:
                adapter.weight.data.copy_(basis)
            details = {"path": "72->48 with training-only layer adapters", "explained": fraction}
            feature_weight = args.feature_weight
        else:
            student = build_flow(3, 72, 4, "cuda", 16, ffn_dim=36).train()
            details = initialize_ffn(teacher, student, args.method, states, noises, args.batch_size)
            details.update({"path": "residual72; FFN 288->36", "final_ffn_dim": 36})
            adapters, feature_weight = None, 0.0
        torch.save(student.state_dict(), args.output_dir / "initial_flow.pth")
        history = train_stage(teacher, student, loader, args.epochs, args.endpoint_weight,
                              args.learning_rate, args.output_dir, adapters=adapters,
                              feature_weight=feature_weight)
    model_params = sum(parameter.numel() for parameter in student.parameters())
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "experiment": "audited width-transfer mechanisms", "method": args.method,
        "details": details, "model_parameters": model_params,
        "uses_original_demonstrations": False, "uses_expert_actions": False,
        "teacher_buffer_samples": len(states), "history": history,
    }, indent=2))


if __name__ == "__main__":
    main()
