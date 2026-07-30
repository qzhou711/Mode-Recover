#!/usr/bin/env python3
"""Merge ordered evaluate_deployed_flow.py shards without changing metrics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths, successes, modes = [], [], []
    for shard in args.shards:
        data = np.load(shard / "trajectories.npz", allow_pickle=True)
        paths.extend(data["trajectories"].tolist())
        successes.extend(data["successes"].astype(bool).tolist())
        modes.extend(data["modes"].astype(np.int8).tolist())
    successes = np.asarray(successes, dtype=bool)
    modes = np.asarray(modes, dtype=np.int8)
    successful_modes = modes[successes]
    if len(successful_modes):
        encoded = successful_modes.dot(1 << np.arange(successful_modes.shape[1]))
        _, counts = np.unique(encoded, return_counts=True)
        probabilities = counts / counts.sum()
        coverage = len(counts)
        entropy = float(-(probabilities * np.log(probabilities) / np.log(24)).sum())
    else:
        coverage, entropy = 0, 0.0
    metrics = {
        "n_trajectories": len(successes),
        "success_rate": float(successes.mean()),
        "successful_trajectories": int(successes.sum()),
        "unique_successful_modes": int(coverage),
        "normalized_mode_entropy": entropy,
        "uses_original_demonstrations": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        args.output_dir / "trajectories.npz",
        trajectories=np.asarray(paths, dtype=object), successes=successes, modes=modes,
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    for path, success in zip(paths, successes):
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], color="#1976d2" if success else "#d32f2f", alpha=0.45, lw=1.2)
    ax.axhline(0.35, color="#2e7d32", ls="--")
    ax.set(
        xlim=(0.25, 0.75), ylim=(-0.32, 0.42), xlabel="x [m]", ylabel="y [m]",
        title=f"SR={metrics['success_rate']:.1%}, modes={coverage}/24, H={entropy:.3f}",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "trajectory_comparison.png", dpi=220)
    plt.close(fig)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
