#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/flow_capacity_scan
OUT="$ROOT/large_to_3x48"

echo EVAL_RESTARTED > "$ROOT/PIPELINE_STATUS"
CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py \
  --models flow \
  --flow-weights-dir "$OUT/model" \
  --flow-steps 1 \
  --flow-layers 3 \
  --flow-embed-dim 48 \
  --flow-heads 3 \
  --n-trajectories 480 \
  --progress-every 10 \
  --output-dir "$OUT/eval480" \
  > "$OUT/eval480_resume.log" 2>&1

test -s "$OUT/eval480/metrics.json"
echo LARGE_TO_3X48_COMPLETE > "$ROOT/GPU0_STATUS"
"$PY" scripts/plot_flow_capacity_scan.py > "$ROOT/plot.log" 2>&1
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
