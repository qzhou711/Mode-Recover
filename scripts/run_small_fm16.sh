#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/small_fm16
WEIGHTS=logs/avoiding/trained/flow_matching_small16_5000_seed42

mkdir -p "$ROOT"

CUDA_VISIBLE_DEVICES=0 "$PY" -u run.py \
  agents=flow_matching_transformer_agent \
  agent_name=flow_matching_small16_5000 \
  window_size=5 \
  n_layer=2 \
  n_embd=36 \
  n_head=3 \
  epoch=5000 \
  eval_every_n_epochs=50 \
  train_batch_size=256 \
  val_batch_size=256 \
  num_workers=4 \
  train_only=True \
  hydra.run.dir="$WEIGHTS" \
  > "$ROOT/train.log" 2>&1

test -s "$WEIGHTS/eval_best_flow.pth"
echo TRAIN_COMPLETE > "$ROOT/PIPELINE_STATUS"

CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py \
  --models flow \
  --flow-weights-dir "$WEIGHTS" \
  --flow-steps 16 \
  --flow-layers 2 \
  --flow-embed-dim 36 \
  --flow-heads 3 \
  --n-trajectories 480 \
  --progress-every 10 \
  --output-dir "$ROOT/eval_step16_480" \
  > "$ROOT/eval_step16_480.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "$PY" -u visualize_avoiding.py \
  --models flow \
  --flow-weights-dir "$WEIGHTS" \
  --flow-steps 1 \
  --flow-layers 2 \
  --flow-embed-dim 36 \
  --flow-heads 3 \
  --n-trajectories 480 \
  --progress-every 10 \
  --output-dir "$ROOT/eval_step1_480" \
  > "$ROOT/eval_step1_480.log" 2>&1 &
pid1=$!

wait "$pid0"
wait "$pid1"
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
