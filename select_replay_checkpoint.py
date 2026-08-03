"""Select one checkpoint per replay method using a pre-registered rule."""
import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--lane", type=Path, required=True)
parser.add_argument("--minimum-sr", type=float, default=0.60)
args = parser.parse_args()
candidates = []
for epoch in (50, 100, 250):
    metrics_path = args.lane / f"epoch_{epoch:04d}" / "eval120" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    candidates.append({
        "epoch": epoch,
        "success_rate": metrics["success_rate"],
        "coverage": metrics["unique_successful_modes"],
        "entropy": metrics["normalized_mode_entropy"],
        "eligible": metrics["success_rate"] >= args.minimum_sr,
    })
eligible = [candidate for candidate in candidates if candidate["eligible"]]
pool = eligible if eligible else candidates
selected = max(pool, key=lambda candidate: (
    candidate["coverage"], candidate["success_rate"], candidate["entropy"]))
result = {
    "selection_protocol": "SR>=0.60, then lexicographic max(coverage, SR, entropy)",
    "selection_declared_before_epoch100_and_epoch250_results": True,
    "candidates": candidates,
    "selected": selected,
}
(args.lane / "standard480_selection.json").write_text(json.dumps(result, indent=2))
print(selected["epoch"])
