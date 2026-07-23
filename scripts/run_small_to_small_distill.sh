#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_small16_5000_seed42
ROOT=logs/avoiding/small_to_small_distill_500
mkdir -p "$ROOT"
CUDA_VISIBLE_DEVICES=0 "$PY" -u distill_flow_matching_avoiding.py \
  --teacher-dir "$TEACHER" \
  --teacher-steps 16 \
  --teacher-layers 2 \
  --teacher-embed-dim 36 \
  --teacher-heads 3 \
  --student-steps 1 \
  --epochs 500 \
  --batch-size 256 \
  --max-batches-per-epoch 4 \
  --flow-weight 0.1 \
  --geometry-weight 0 \
  --output-dir "$ROOT/model" \
  > "$ROOT/train.log" 2>&1

test -s "$ROOT/model/eval_best_flow.pth"
echo TRAIN_COMPLETE > "$ROOT/PIPELINE_STATUS"
CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py \
  --models flow \
  --flow-weights-dir "$ROOT/model" \
  --flow-steps 1 \
  --flow-layers 2 \
  --flow-embed-dim 36 \
  --flow-heads 3 \
  --n-trajectories 480 \
  --progress-every 10 \
  --output-dir "$ROOT/eval480" \
  > "$ROOT/eval480.log" 2>&1
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
