#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("logs/avoiding/flow_capacity_scan")
EXPERIMENTS = [
    (
        "2×36",
        Path("logs/avoiding/flow_auto/small_geo0/distillation_metrics.json"),
        Path("logs/avoiding/flow_followup/small500_geo0_eval480/metrics.json"),
    ),
    (
        "3×48",
        ROOT / "large_to_3x48/model/distillation_metrics.json",
        ROOT / "large_to_3x48/eval480/metrics.json",
    ),
    (
        "4×64",
        ROOT / "large_to_4x64/model/distillation_metrics.json",
        ROOT / "large_to_4x64/eval480/metrics.json",
    ),
]


def main():
    rows = []
    for label, train_path, eval_path in EXPERIMENTS:
        train = json.loads(train_path.read_text())
        result = json.loads(eval_path.read_text())["Flow-Matching"]
        rows.append(
            {
                "architecture": label,
                "parameters": train["student_parameters"],
                "success_rate": result["success_rate"],
                "successful_trajectories": result["successful_trajectories"],
                "unique_successful_modes": result["unique_successful_modes"],
                "normalized_mode_entropy": result["normalized_mode_entropy"],
            }
        )

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "capacity_scan_results.json").write_text(json.dumps(rows, indent=2))
    x = [row["parameters"] for row in rows]
    plots = [
        ("success_rate", "Success rate", "capacity_vs_success_rate.png"),
        ("normalized_mode_entropy", "Normalized mode entropy", "capacity_vs_mode_entropy.png"),
        ("unique_successful_modes", "Covered modes", "capacity_vs_mode_count.png"),
    ]
    for key, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        y = [row[key] for row in rows]
        ax.plot(x, y, marker="o", linewidth=2)
        for row, x_value, y_value in zip(rows, x, y):
            ax.annotate(row["architecture"], (x_value, y_value), xytext=(5, 5),
                        textcoords="offset points")
        ax.set_xlabel("Student parameters")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROOT / filename, dpi=180)
        plt.close(fig)

    lines = [
        "# Flow Matching student capacity scan",
        "",
        "| Student | Parameters | Success | Modes | Entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {row['parameters']:,} | "
            f"{row['successful_trajectories']}/480 "
            f"({100 * row['success_rate']:.1f}%) | "
            f"{row['unique_successful_modes']}/24 | "
            f"{row['normalized_mode_entropy']:.3f} |"
        )
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
