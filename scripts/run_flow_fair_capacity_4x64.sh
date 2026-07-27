#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
FULL=logs/avoiding/trained/flow_matching_4x64_5000_seed42
ROOT=logs/avoiding/flow_fair_capacity_4x64
DISTILLED="$ROOT/fm_4x64_1_distill_4x72_full_init"

mkdir -p "$ROOT"
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 2 ]]; then
  echo NEED_TWO_GPUS > "$ROOT/STATUS"
  exit 1
fi

mkdir -p "$FULL" "$DISTILLED/model"
echo TRAINING_FULL > "$ROOT/STATUS"

CUDA_VISIBLE_DEVICES=0 "$PY" -u run.py \
  agents=flow_matching_transformer_agent \
  agent_name=flow_matching_4x64_5000 \
  window_size=5 \
  n_layer=4 \
  n_embd=64 \
  n_head=4 \
  epoch=5000 \
  eval_every_n_epochs=50 \
  train_batch_size=256 \
  val_batch_size=256 \
  num_workers=4 \
  train_only=True \
  hydra.run.dir="$FULL" \
  > "$ROOT/full_train.log" 2>&1
test -s "$FULL/eval_best_flow.pth"

eval_two_gpu() {
  local model=$1
  local steps=$2
  local output=$3
  local shards="$output/shards"
  mkdir -p "$shards"
  for gpu in 0 1; do
    start=$((gpu * 240))
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
      --models flow \
      --flow-weights-dir "$model" \
      --flow-steps "$steps" \
      --flow-layers 4 \
      --flow-embed-dim 64 \
      --flow-heads 4 \
      --episode-start "$start" \
      --n-trajectories 240 \
      --progress-every 10 \
      --output-dir "$shards/gpu${gpu}" \
      > "$output/gpu${gpu}.log" 2>&1 &
    pids[$gpu]=$!
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi
  "$PY" scripts/merge_avoiding_eval_shards.py \
    --shards "$shards/gpu0" "$shards/gpu1" \
    --output-dir "$output" \
    > "$output/merge.log" 2>&1
}

echo EVALUATING_FULL16 > "$ROOT/STATUS"
eval_two_gpu "$FULL" 16 "$ROOT/fm_4x64_16_full_eval480"
echo EVALUATING_SOLVER1 > "$ROOT/STATUS"
eval_two_gpu "$FULL" 1 "$ROOT/fm_4x64_1_solver_eval480"

echo DISTILLING_FULL_INIT > "$ROOT/STATUS"
CUDA_VISIBLE_DEVICES=0 "$PY" -u distill_flow_matching_avoiding.py \
  --teacher-dir "$TEACHER" \
  --teacher-steps 16 \
  --teacher-layers 4 \
  --teacher-embed-dim 72 \
  --teacher-heads 4 \
  --student-steps 1 \
  --student-layers 4 \
  --student-embed-dim 64 \
  --student-heads 4 \
  --student-init checkpoint \
  --student-init-dir "$FULL" \
  --epochs 500 \
  --batch-size 256 \
  --max-batches-per-epoch 4 \
  --flow-weight 0.1 \
  --geometry-weight 0 \
  --seed 42 \
  --output-dir "$DISTILLED/model" \
  > "$DISTILLED/train.log" 2>&1
test -s "$DISTILLED/model/eval_best_flow.pth"

"$PY" - "$DISTILLED/model/distillation_metrics.json" <<'PY'
import json
import sys
from pathlib import Path
metrics = json.loads(Path(sys.argv[1]).read_text())
assert metrics["student_initialization"] == "checkpoint"
assert metrics["initialization_max_abs_diff"] == 0.0
PY

echo EVALUATING_DISTILLED1 > "$ROOT/STATUS"
eval_two_gpu "$DISTILLED/model" 1 "$DISTILLED/eval480"
echo COMPLETE > "$ROOT/STATUS"
