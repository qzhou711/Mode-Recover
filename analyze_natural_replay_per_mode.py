"""Three-seed per-mode success and exact retention for Natural Replay."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    data = np.load(path / "trajectories.npz", allow_pickle=True)
    modes = data["modes"]
    return data["successes"].astype(bool), modes.dot(1 << np.arange(modes.shape[1]))


parser = argparse.ArgumentParser()
parser.add_argument("--teacher", type=Path, required=True)
parser.add_argument("--students", type=Path, nargs=3, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
teacher_success, teacher_mode = load(args.teacher)
modes = np.unique(teacher_mode[teacher_success])
success_rates, retention_rates, rows = [], [], []
for seed, path in zip((42, 43, 44), args.students):
    success, mode = load(path)
    if len(success) != len(teacher_success):
        raise ValueError(f"paired length mismatch for {path}")
    seed_success, seed_retention = [], []
    for code in modes:
        condition = teacher_success & (teacher_mode == code)
        sr = float(success[condition].mean())
        retention = float((success[condition] & (mode[condition] == code)).mean())
        seed_success.append(sr)
        seed_retention.append(retention)
        rows.append({"train_seed": seed, "teacher_mode_code": int(code),
                     "teacher_support": int(condition.sum()), "student_success_rate": sr,
                     "exact_mode_retention_rate": retention})
    success_rates.append(seed_success)
    retention_rates.append(seed_retention)
success_rates = np.asarray(success_rates)
retention_rates = np.asarray(retention_rates)
with (args.output_dir / "per_mode_rates.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
summary = {
    "conditioning": "paired Teacher-success mode under identical evaluation episode/noise",
    "train_seeds": [42, 43, 44],
    "teacher_mode_codes": modes.tolist(),
    "success_mean": success_rates.mean(0).tolist(),
    "success_std": success_rates.std(0, ddof=1).tolist(),
    "retention_mean": retention_rates.mean(0).tolist(),
    "retention_std": retention_rates.std(0, ddof=1).tolist(),
    "macro_per_mode_success_mean": float(success_rates.mean(1).mean()),
    "macro_exact_retention_mean": float(retention_rates.mean(1).mean()),
}
(args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
x = np.arange(len(modes))
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].errorbar(x, success_rates.mean(0), yerr=success_rates.std(0, ddof=1),
                 marker="o", ms=3, capsize=2)
axes[1].errorbar(x, retention_rates.mean(0), yerr=retention_rates.std(0, ddof=1),
                 marker="o", ms=3, capsize=2)
axes[0].set(ylabel="Student success rate", title="Natural Replay per-mode success", ylim=(-.05, 1.05))
axes[1].set(ylabel="Exact mode retention", xlabel="Teacher mode code", ylim=(-.05, 1.05))
axes[1].set_xticks(x, modes, rotation=60)
for axis in axes: axis.grid(alpha=.25)
fig.tight_layout(); fig.savefig(args.output_dir / "per_mode_success_and_retention.png", dpi=180)
print(json.dumps(summary, indent=2))
