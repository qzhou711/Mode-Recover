import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualize_avoiding import draw, metrics


def valid_mode_codes():
    codes = []
    for first in range(2):
        for second in range(3):
            for third in range(4):
                vector = np.zeros(9, dtype=np.int8)
                vector[first] = 1
                vector[2 + second] = 1
                vector[5 + third] = 1
                codes.append(int(vector.dot(1 << np.arange(9))))
    return codes


def mode_classes(successes, modes):
    code_to_class = {code: index for index, code in enumerate(valid_mode_codes())}
    encoded = modes.dot(1 << np.arange(modes.shape[1]))
    return np.asarray(
        [code_to_class.get(int(code), 24) if success else 24
         for success, code in zip(successes, encoded)],
        dtype=np.int64,
    )


def transition_matrix(source, target, size):
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (source, target), 1)
    return matrix


def hierarchical_classes(successes, modes, level):
    if level not in {1, 2, 3}:
        raise ValueError("level must be 1, 2, or 3")
    first = modes[:, :2].argmax(axis=1)
    if level == 1:
        classes, failure_class = first, 2
    else:
        second = modes[:, 2:5].argmax(axis=1)
        if level == 2:
            classes, failure_class = first * 3 + second, 6
        else:
            return mode_classes(successes, modes)
    return np.where(successes, classes, failure_class).astype(np.int64)


def entropy_from_counts(counts):
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0


def information_metrics(matrix):
    total = matrix.sum()
    joint = matrix / total
    source = joint.sum(axis=1)
    target = joint.sum(axis=0)
    mutual_information = 0.0
    for row, col in zip(*np.nonzero(joint)):
        mutual_information += joint[row, col] * np.log(
            joint[row, col] / (source[row] * target[col])
        )
    source_entropy = entropy_from_counts(matrix.sum(axis=1))
    target_entropy = entropy_from_counts(matrix.sum(axis=0))
    conditional_entropy = entropy_from_counts(matrix.ravel()) - source_entropy
    denominator = np.sqrt(source_entropy * target_entropy)
    return {
        "mutual_information": float(mutual_information),
        "normalized_mutual_information": float(mutual_information / denominator)
        if denominator > 0
        else 0.0,
        "student_given_teacher_entropy": float(conditional_entropy),
    }


def js_divergence(teacher_counts, student_counts):
    teacher = teacher_counts / max(teacher_counts.sum(), 1)
    student = student_counts / max(student_counts.sum(), 1)
    midpoint = 0.5 * (teacher + student)
    terms = []
    for distribution in (teacher, student):
        valid = distribution > 0
        terms.append(
            0.5 * np.sum(distribution[valid] * np.log(distribution[valid] / midpoint[valid]))
        )
    return float(sum(terms))


def plot_matrix(matrix, labels, title, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xlabel("student")
    ax.set_ylabel("teacher")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keys = [
        "episode_ids", "teacher_paths", "student_paths", "teacher_successes",
        "student_successes", "teacher_modes", "student_modes",
        "teacher_steps", "student_steps",
    ]
    merged = {key: [] for key in keys}
    for shard in args.shards:
        data = np.load(shard / "paired_rollouts.npz", allow_pickle=True)
        for key in keys:
            merged[key].extend(list(data[key]))
    order = np.argsort(np.asarray(merged["episode_ids"], dtype=int))
    for key in keys:
        merged[key] = np.asarray(merged[key], dtype=object)[order]
    episode_ids = merged["episode_ids"].astype(int)
    if len(episode_ids) != args.expected_episodes or len(np.unique(episode_ids)) != len(episode_ids):
        raise RuntimeError("paired shards are incomplete or contain duplicate episode ids")

    teacher_successes = merged["teacher_successes"].astype(bool)
    student_successes = merged["student_successes"].astype(bool)
    teacher_modes = np.stack(merged["teacher_modes"]).astype(np.int8)
    student_modes = np.stack(merged["student_modes"]).astype(np.int8)
    teacher_classes = mode_classes(teacher_successes, teacher_modes)
    student_classes = mode_classes(student_successes, student_modes)
    matrix = transition_matrix(teacher_classes, student_classes, 25)
    both_success = teacher_successes & student_successes
    same_mode = both_success & (teacher_classes == student_classes)

    teacher_metric = metrics(teacher_successes, teacher_modes)
    student_metric = metrics(student_successes, student_modes)
    result = {
        "episodes": len(episode_ids),
        "teacher": teacher_metric,
        "student": student_metric,
        "teacher_success_to_student_failure": int(
            np.sum(teacher_successes & ~student_successes)
        ),
        "teacher_success_to_student_failure_rate": float(
            np.sum(teacher_successes & ~student_successes)
            / max(np.sum(teacher_successes), 1)
        ),
        "exact_mode_retention_over_teacher_success": float(
            np.sum(same_mode) / max(np.sum(teacher_successes), 1)
        ),
        "exact_mode_retention_when_both_succeed": float(
            np.sum(same_mode) / max(np.sum(both_success), 1)
        ),
        "successful_mode_js_divergence": js_divergence(
            matrix[:24].sum(axis=1), matrix[:, :24].sum(axis=0)
        ),
        **information_metrics(matrix),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        args.output_dir / "paired_merged.npz",
        **merged,
        teacher_classes=teacher_classes,
        student_classes=student_classes,
        transition_matrix=matrix,
    )

    labels = [
        f"{first + 1}-{second + 1}-{third + 1}"
        for first in range(2) for second in range(3) for third in range(4)
    ] + ["Failure"]
    plot_matrix(
        matrix,
        labels,
        "Teacher to student mode transitions",
        args.output_dir / "mode_transition_matrix.png",
    )

    for level, size, level_labels in (
        (1, 3, ["1", "2", "Failure"]),
        (
            2,
            7,
            [f"{first + 1}-{second + 1}" for first in range(2) for second in range(3)]
            + ["Failure"],
        ),
    ):
        teacher_level = hierarchical_classes(teacher_successes, teacher_modes, level)
        student_level = hierarchical_classes(student_successes, student_modes, level)
        level_matrix = transition_matrix(teacher_level, student_level, size)
        result[f"level_{level}_exact_retention_over_teacher_success"] = float(
            np.sum(
                teacher_successes
                & student_successes
                & (teacher_level == student_level)
            )
            / max(np.sum(teacher_successes), 1)
        )
        plot_matrix(
            level_matrix,
            level_labels,
            f"Teacher to student transitions through level {level}",
            args.output_dir / f"mode_transition_level_{level}.png",
        )
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    draw(axes[0], merged["teacher_paths"], teacher_successes, "Teacher paired rollouts")
    draw(axes[1], merged["student_paths"], student_successes, "Student paired rollouts")
    fig.savefig(args.output_dir / "paired_trajectories.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
