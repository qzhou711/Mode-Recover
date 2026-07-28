#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
ROOT=logs/avoiding/flow_compression_stage1
METHODS=(activation pca merge random)

mkdir -p "$ROOT"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -lt 4 ]]; then
  echo "NEED_4_GPUS found=$gpu_count $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

echo "TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1 2 3; do
  method=${METHODS[$gpu]}
  out="$ROOT/$method"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_flow_compression_stage1.py \
    --teacher-dir "$TEACHER" \
    --output-dir "$out/model" \
    --method "$method" \
    --epochs 500 \
    --batch-size 256 \
    --max-batches-per-epoch 4 \
    --calibration-batches 8 \
    --learning-rate 1e-4 \
    --flow-weight 0.1 \
    --seed 42 \
    > "$out/train.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

for method in "${METHODS[@]}"; do
  test -s "$ROOT/$method/model/eval_best_flow.pth"
  test -s "$ROOT/$method/model/compression_metrics.json"
  "$PY" -c '
import json
import math
import sys

with open(sys.argv[1]) as handle:
    metrics = json.load(handle)
assert metrics["method"] == sys.argv[2]
assert metrics["teacher_steps"] == metrics["student_steps"] == 16
assert metrics["student_parameters"] == 37982
assert all(math.isfinite(value) for value in metrics["open_loop"].values())
if metrics["method"] != "random":
    assert metrics["initialization"]["load_max_abs_diff"] == 0.0
' "$ROOT/$method/model/compression_metrics.json" "$method"
done

echo "EVALUATING_120 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1 2 3; do
  method=${METHODS[$gpu]}
  out="$ROOT/$method"
  mkdir -p "$out/eval120"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow \
    --flow-weights-dir "$out/model" \
    --flow-steps 16 \
    --flow-layers 2 \
    --flow-embed-dim 36 \
    --flow-heads 3 \
    --n-trajectories 120 \
    --progress-every 10 \
    --output-dir "$out/eval120" \
    > "$out/eval120.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

"$PY" -c '
import json
from pathlib import Path

root = Path("logs/avoiding/flow_compression_stage1")
summary = {}
for method in ("activation", "pca", "merge", "random"):
    compression = json.loads((root / method / "model/compression_metrics.json").read_text())
    closed = json.loads((root / method / "eval120/metrics.json").read_text())["Flow-Matching"]
    summary[method] = {
        "open_loop": compression["open_loop"],
        "best_epoch": compression["best_epoch"],
        "closed_loop_120": closed,
        "passes_mode_gate": closed["unique_successful_modes"] >= 18,
    }
(root / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
' > "$ROOT/summary.log" 2>&1

echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
