import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from visualize_avoiding import make_agent


def select_states(agent, count):
    states = []
    for batch in agent.test_dataloader:
        states.append(batch[0])
        if sum(item.shape[0] for item in states) >= max(1024, count):
            break
    states = torch.cat(states, dim=0)
    final_y = states[:, -1, 3]
    order = torch.argsort(final_y)
    positions = torch.linspace(0, len(order) - 1, count).round().long()
    indices = order[positions]
    return states[indices].float(), indices, final_y[indices]


def sample_in_chunks(model, states, noises, steps, chunk_size):
    state_count, sample_count = noises.shape[:2]
    flat_states = states[:, None].expand(-1, sample_count, -1, -1).reshape(
        state_count * sample_count, *states.shape[1:]
    )
    flat_noises = noises.reshape(state_count * sample_count, *noises.shape[2:])
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat_states), chunk_size):
            outputs.append(
                model.sample(
                    flat_states[start : start + chunk_size].to(model.device),
                    initial_noise=flat_noises[start : start + chunk_size].to(model.device),
                    steps=steps,
                ).cpu()
            )
    return torch.cat(outputs).reshape(state_count, sample_count, *noises.shape[2:])


def pairwise_distance_correlation(teacher, student, limit=256):
    teacher = teacher[:limit].reshape(min(len(teacher), limit), -1)
    student = student[:limit].reshape(min(len(student), limit), -1)
    rows, cols = np.triu_indices(len(teacher), k=1)
    teacher_dist = np.linalg.norm(teacher[rows] - teacher[cols], axis=1)
    student_dist = np.linalg.norm(student[rows] - student[cols], axis=1)
    if teacher_dist.std() == 0 or student_dist.std() == 0:
        return 0.0
    return float(np.corrcoef(teacher_dist, student_dist)[0, 1])


def compute_metrics(teacher, student):
    rows = []
    for state_index in range(len(teacher)):
        teacher_flat = teacher[state_index].reshape(len(teacher[state_index]), -1)
        student_flat = student[state_index].reshape(len(student[state_index]), -1)
        teacher_centered = teacher_flat - teacher_flat.mean(axis=0)
        student_centered = student_flat - student_flat.mean(axis=0)
        teacher_variance = float(np.mean(np.sum(teacher_centered**2, axis=1)))
        student_variance = float(np.mean(np.sum(student_centered**2, axis=1)))
        paired_mse = float(np.mean((teacher_flat - student_flat) ** 2))
        dot = np.sum(teacher_centered * student_centered, axis=1)
        denominator = np.linalg.norm(teacher_centered, axis=1) * np.linalg.norm(
            student_centered, axis=1
        )
        valid = denominator > 1e-12
        cosine = float(np.mean(dot[valid] / denominator[valid])) if valid.any() else 0.0
        rows.append(
            {
                "state_index": state_index,
                "teacher_variance": teacher_variance,
                "student_variance": student_variance,
                "variance_retention": student_variance / max(teacher_variance, 1e-12),
                "paired_mse": paired_mse,
                "centered_cosine": cosine,
                "pairwise_distance_correlation": pairwise_distance_correlation(
                    teacher_flat, student_flat
                ),
            }
        )
    return rows


def plot_summary(teacher, student, state_y, metrics, output_dir):
    count = len(teacher)
    selected = np.unique(np.linspace(0, count - 1, min(4, count)).round().astype(int))
    fig, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 7))
    axes = np.asarray(axes).reshape(2, len(selected))
    for column, state_index in enumerate(selected):
        axes[0, column].scatter(
            teacher[state_index, :, -1, 0],
            teacher[state_index, :, -1, 1],
            s=8,
            alpha=0.35,
            label="teacher",
        )
        axes[0, column].scatter(
            student[state_index, :, -1, 0],
            student[state_index, :, -1, 1],
            s=8,
            alpha=0.35,
            label="student",
        )
        axes[0, column].set_title(f"state {state_index}, y={state_y[state_index]:.3f}")
        axes[0, column].set_xlabel("last action x")
        axes[0, column].set_ylabel("last action y")
        axes[0, column].grid(alpha=0.2)

        joint = np.concatenate(
            [
                teacher[state_index].reshape(len(teacher[state_index]), -1),
                student[state_index].reshape(len(student[state_index]), -1),
            ]
        )
        centered = joint - joint.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
        split = len(teacher[state_index])
        axes[1, column].scatter(
            projected[:split, 0], projected[:split, 1], s=8, alpha=0.35
        )
        axes[1, column].scatter(
            projected[split:, 0], projected[split:, 1], s=8, alpha=0.35
        )
        axes[1, column].set_xlabel("joint PCA 1")
        axes[1, column].set_ylabel("joint PCA 2")
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "conditional_action_scatter.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(count)
    axes[0].plot(x, [row["variance_retention"] for row in metrics], marker="o")
    axes[0].axhline(1.0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("student / teacher variance")
    axes[1].plot(x, [row["centered_cosine"] for row in metrics], marker="o")
    axes[1].set_ylabel("paired centered cosine")
    axes[2].plot(
        x, [row["pairwise_distance_correlation"] for row in metrics], marker="o"
    )
    axes[2].set_ylabel("pairwise distance correlation")
    for axis in axes:
        axis.set_xlabel("state ordered by actual y")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mapping_metrics_by_state.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--teacher-layers", type=int, required=True)
    parser.add_argument("--teacher-embed-dim", type=int, required=True)
    parser.add_argument("--teacher-heads", type=int, required=True)
    parser.add_argument("--teacher-steps", type=int, default=16)
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument("--student-layers", type=int, required=True)
    parser.add_argument("--student-embed-dim", type=int, required=True)
    parser.add_argument("--student-heads", type=int, required=True)
    parser.add_argument("--student-steps", type=int, default=1)
    parser.add_argument("--state-count", type=int, default=12)
    parser.add_argument("--samples-per-state", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(mode="disabled")

    teacher_agent = make_agent(
        "flow_matching_transformer_agent",
        5,
        args.teacher_dir,
        "eval_best_flow.pth",
        [
            f"n_layer={args.teacher_layers}",
            f"n_embd={args.teacher_embed_dim}",
            f"n_head={args.teacher_heads}",
        ],
    )
    student_agent = make_agent(
        "flow_matching_transformer_agent",
        5,
        args.student_dir,
        "eval_best_flow.pth",
        [
            f"n_layer={args.student_layers}",
            f"n_embd={args.student_embed_dim}",
            f"n_head={args.student_heads}",
        ],
    )
    raw_states, dataset_indices, state_y = select_states(
        teacher_agent, args.state_count
    )
    scaled_states = teacher_agent.scaler.scale_input(raw_states).cpu()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noises = torch.randn(
        args.state_count,
        args.samples_per_state,
        raw_states.shape[1],
        2,
        generator=generator,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    teacher = sample_in_chunks(
        teacher_agent.model,
        scaled_states,
        noises,
        args.teacher_steps,
        args.chunk_size,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    student = sample_in_chunks(
        student_agent.model,
        scaled_states,
        noises,
        args.student_steps,
        args.chunk_size,
    )
    reproducibility_samples = min(args.chunk_size, args.samples_per_state)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    repeated = sample_in_chunks(
        teacher_agent.model,
        scaled_states[:1],
        noises[:1, :reproducibility_samples],
        args.teacher_steps,
        args.chunk_size,
    )
    reproducibility_max_abs_error = float(
        (teacher[:1, : repeated.shape[1]] - repeated).abs().max().item()
    )
    reproducible = bool(
        torch.allclose(teacher[:1, : repeated.shape[1]], repeated, atol=1e-6, rtol=1e-6)
    )
    if not reproducible:
        raise RuntimeError(
            f"teacher sampling exceeds reproducibility tolerance: {reproducibility_max_abs_error}"
        )

    teacher_unscaled = teacher_agent.scaler.inverse_scale_output(
        teacher.to(teacher_agent.device)
    ).cpu().numpy()
    student_unscaled = teacher_agent.scaler.inverse_scale_output(
        student.to(teacher_agent.device)
    ).cpu().numpy()
    metric_rows = compute_metrics(teacher_unscaled, student_unscaled)
    summary = {
        "name": args.name,
        "state_count": args.state_count,
        "samples_per_state": args.samples_per_state,
        "teacher_steps": args.teacher_steps,
        "student_steps": args.student_steps,
        "reproducible_within_1e-6": reproducible,
        "reproducibility_max_abs_error": reproducibility_max_abs_error,
        "mean_variance_retention": float(
            np.mean([row["variance_retention"] for row in metric_rows])
        ),
        "mean_paired_mse": float(np.mean([row["paired_mse"] for row in metric_rows])),
        "mean_centered_cosine": float(
            np.mean([row["centered_cosine"] for row in metric_rows])
        ),
        "mean_pairwise_distance_correlation": float(
            np.mean([row["pairwise_distance_correlation"] for row in metric_rows])
        ),
        "states": metric_rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output_dir / "mapping_data.npz",
        raw_states=raw_states.numpy(),
        dataset_indices=dataset_indices.numpy(),
        state_y=state_y.numpy(),
        noises=noises.numpy(),
        teacher_actions=teacher_unscaled,
        student_actions=student_unscaled,
    )
    plot_summary(
        teacher_unscaled,
        student_unscaled,
        state_y.numpy(),
        metric_rows,
        args.output_dir,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "states"}, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
