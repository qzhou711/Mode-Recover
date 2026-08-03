"""Demo-free low-rank operator compression while preserving 72-D residuals."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agents.models.diffusion.operator_compression import (
    LowRankLinear, get_submodule, operator_config,
)
from teacher_flow_deployment import build_flow
from train_width_transfer_mechanisms import load_buffer, train_stage


VARIANTS = ("uniform_svd", "uniform_activation", "routing_aware", "hybrid_balanced")


def copy_matching_state(teacher, student):
    teacher_state, student_state = teacher.state_dict(), student.state_dict()
    copied = {}
    for key, target in student_state.items():
        source = teacher_state.get(key)
        if source is not None and source.shape == target.shape:
            copied[key] = source.detach().clone()
        else:
            copied[key] = target
    student.load_state_dict(copied, strict=True)


@torch.no_grad()
def collect_inputs(teacher, module_names, states, noises, batches=8, batch_size=256):
    captured = {name: [] for name in module_names}
    for batch in range(batches):
        start = batch * batch_size
        state = states[start:start + batch_size].cuda()
        noise = noises[start:start + batch_size].cuda()
        if not len(state):
            break
        time = torch.full((len(state),), 0.5, device="cuda")
        x_t = teacher.integrate(noise, state, start_time=0.0, end_time=0.5, steps=8)
        handles = []
        for name in module_names:
            module = get_submodule(teacher.model, name)
            handles.append(module.register_forward_pre_hook(
                lambda _m, inputs, name=name: captured[name].append(
                    inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
                )
            ))
        teacher.velocity(x_t, time, state)
        for handle in handles:
            handle.remove()
    return {name: torch.cat(values) for name, values in captured.items()}


def factors(weight, rank, inputs=None):
    weight = weight.detach().float().cpu()
    if inputs is None:
        left, singular, right = torch.linalg.svd(weight, full_matrices=False)
        left, singular, right = left[:, :rank], singular[:rank], right[:rank]
        root = singular.sqrt()
        return root[:, None] * right, left * root[None, :]
    inputs = inputs.float()
    covariance = inputs.T @ inputs / max(1, len(inputs))
    epsilon = 1e-4 * covariance.diag().mean().clamp_min(1e-8)
    cholesky = torch.linalg.cholesky(covariance + epsilon * torch.eye(len(covariance)))
    weighted = weight @ cholesky
    left, singular, right = torch.linalg.svd(weighted, full_matrices=False)
    left, singular, right = left[:, :rank], singular[:rank], right[:rank]
    root = singular.sqrt()
    weighted_down = root[:, None] * right
    down = torch.linalg.solve_triangular(
        cholesky.T, weighted_down.T, upper=True
    ).T
    up = left * root[None, :]
    return down, up


def initialize_lowrank(teacher, student, config, captured=None):
    audits = {}
    for name, rank in config["ranks"].items():
        source = get_submodule(teacher.model, name)
        target = get_submodule(student.model, name)
        if not isinstance(target, LowRankLinear):
            raise TypeError(name)
        down, up = factors(source.weight, rank, None if captured is None else captured[name])
        target.down.weight.data.copy_(down.to(target.down.weight.device))
        target.up.weight.data.copy_(up.to(target.up.weight.device))
        if source.bias is not None:
            target.up.bias.data.copy_(source.bias.data)
        with torch.no_grad():
            reconstructed = target.up.weight.detach().cpu() @ target.down.weight.detach().cpu()
            relative = float((reconstructed - source.weight.detach().cpu()).norm() /
                             source.weight.detach().cpu().norm().clamp_min(1e-12))
        audits[name] = {"rank": rank, "relative_weight_error": relative}
    return audits


def initialize_hybrid_ffn(teacher, student, ffn_inputs):
    selected = {}
    for layer in range(3):
        name = f"blocks.{layer}.mlp.2"
        energy = ffn_inputs[name].square().mean(0)
        channels = torch.topk(energy, 96).indices.sort().values
        selected[str(layer)] = channels.tolist()
        source0 = teacher.model.blocks[layer].mlp[0]
        source2 = teacher.model.blocks[layer].mlp[2]
        target0 = student.model.blocks[layer].mlp[0]
        target2 = student.model.blocks[layer].mlp[2]
        index = channels.to(source0.weight.device)
        target0.weight.data.copy_(source0.weight.data[index])
        target0.bias.data.copy_(source0.bias.data[index])
        target2.weight.data.copy_(source2.weight.data[:, index])
        target2.bias.data.copy_(source2.bias.data)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--endpoint-weight", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = operator_config(args.variant)
    (args.output_dir / "operator_config.json").write_text(json.dumps(config, indent=2))

    metadata = torch.load(args.bundle_dir / "deployment_metadata.pt", map_location="cpu")
    teacher = build_flow(3, 72, 4, "cuda", 16)
    checkpoint = args.teacher if args.teacher.is_file() else args.teacher / "eval_best_flow.pth"
    teacher.load_state_dict(torch.load(checkpoint, map_location="cuda"), strict=True)
    teacher.min_action = metadata["y_bounds_tensor"][0].cuda()
    teacher.max_action = metadata["y_bounds_tensor"][1].cuda()
    teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    states, noises, endpoints = load_buffer(args.buffer)

    student = build_flow(3, 72, 4, "cuda", 16, operator_config=config).train()
    copy_matching_state(teacher, student)
    module_names = list(config["ranks"])
    capture_names = module_names if args.variant == "uniform_activation" else []
    if args.variant == "hybrid_balanced":
        capture_names += [f"blocks.{layer}.mlp.2" for layer in range(3)]
    captured = collect_inputs(teacher, capture_names, states, noises, batch_size=args.batch_size) if capture_names else {}
    factor_audit = initialize_lowrank(
        teacher, student, config,
        captured if args.variant == "uniform_activation" else None,
    )
    selected = initialize_hybrid_ffn(teacher, student, captured) if args.variant == "hybrid_balanced" else None
    torch.save(student.state_dict(), args.output_dir / "initial_flow.pth")
    student.min_action, student.max_action = teacher.min_action, teacher.max_action
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(states, noises, endpoints),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    history = train_stage(
        teacher, student, loader, args.epochs, args.endpoint_weight,
        args.learning_rate, args.output_dir,
    )
    summary = {
        "experiment": "residual-preserving operator-rank compression",
        "variant": args.variant,
        "operator_config": config,
        "model_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "teacher_parameters": sum(parameter.numel() for parameter in teacher.parameters()),
        "factorization_audit": factor_audit,
        "hybrid_selected_ffn_channels": selected,
        "teacher_buffer_samples": len(states),
        "uses_original_demonstrations": False,
        "uses_expert_actions": False,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
