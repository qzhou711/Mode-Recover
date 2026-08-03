import argparse
import json
import shutil
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("--lane", type=Path, required=True)
p.add_argument("--epochs", type=int, nargs="+", default=[100, 250, 500])
p.add_argument("--min-success", type=float, default=0.75)
p.add_argument("--min-coverage", type=int, default=20)
a = p.parse_args()

rows = []
for epoch in a.epochs:
    metrics_path = a.lane / f"eval120_epoch{epoch}" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    rows.append(
        {
            "epoch": epoch,
            "success_rate": metrics["success_rate"],
            "coverage": metrics["unique_successful_modes"],
            "entropy": metrics["normalized_mode_entropy"],
            "metrics": str(metrics_path),
        }
    )

eligible = [
    row
    for row in rows
    if row["success_rate"] >= a.min_success and row["coverage"] >= a.min_coverage
]
ranked = eligible if eligible else rows
selected = max(
    ranked,
    key=lambda row: (row["success_rate"], row["coverage"], row["entropy"]),
)
selected["passed_gate"] = bool(eligible)
source = a.lane / "model" / f"pretrain_epoch_{selected['epoch']:04d}.pth"
target_dir = a.lane / "selected_model"
target_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, target_dir / "eval_best_flow.pth")
summary = {
    "protocol": "Standard-120: require SR>=0.75 and coverage>=20; then maximize SR, coverage, entropy",
    "rows": rows,
    "selected": selected,
}
(a.lane / "selection.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
