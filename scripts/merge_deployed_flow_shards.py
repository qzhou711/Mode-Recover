#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from visualize_avoiding import draw, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trajectories, successes, modes = [], [], []
    for shard in args.shards:
        data = np.load(shard / "trajectories.npz", allow_pickle=True)
        trajectories.extend(list(data["trajectories"]))
        successes.extend(data["successes"].astype(bool).tolist())
        modes.extend(list(data["modes"]))

    successes = np.asarray(successes, dtype=bool)
    modes = np.asarray(modes, dtype=np.int8)
    result = metrics(successes, modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "trajectories.npz",
        trajectories=np.asarray(trajectories, dtype=object),
        successes=successes,
        modes=modes,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"Flow-Matching": result}, indent=2)
    )

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    draw(
        ax,
        trajectories,
        successes,
        f"Flow-Matching: success={result['success_rate']:.1%}, "
        f"modes={result['unique_successful_modes']}",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.savefig(args.output_dir / "trajectory_comparison.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"Flow-Matching": result}, indent=2))


if __name__ == "__main__":
    main()
