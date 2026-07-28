#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
ROOT=logs/avoiding/flow_structured_distribution_scan
D1="$ROOT/fm_2x36_1_distill_4x72_structured_gram_w1"
D10="$ROOT/fm_2x36_1_distill_4x72_structured_gram_w10"

mkdir -p "$ROOT"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -lt 4 ]]; then
  echo "Need four visible GPUs, found $gpu_count" > "$ROOT/STATUS"
  exit 1
fi

train_one() {
  local gpu=$1
  local weight=$2
  local output=$3
  mkdir -p "$output/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u distill_flow_matching_avoiding.py \
    --teacher-dir "$TEACHER" \
    --teacher-steps 16 \
    --teacher-layers 4 \
    --teacher-embed-dim 72 \
    --teacher-heads 4 \
    --student-steps 1 \
    --student-layers 2 \
    --student-embed-dim 36 \
    --student-heads 3 \
    --student-init structured_teacher \
    --epochs 500 \
    --batch-size 256 \
    --max-batches-per-epoch 4 \
    --flow-weight 0.1 \
    --geometry-weight 0 \
    --distribution-weight "$weight" \
    --conditional-samples 4 \
    --seed 42 \
    --output-dir "$output/model" \
    > "$output/train.log" 2>&1
}

validate_model() {
  local metrics=$1
  local expected_weight=$2
  "$PY" -c '
import json
import math
import sys

with open(sys.argv[1]) as handle:
    metrics = json.load(handle)
assert metrics["student_initialization"] == "structured_teacher"
assert metrics["student_initialization_checkpoint"] is None
assert metrics["student_parameters"] == 37982
assert metrics["conditional_samples"] == 4
assert math.isclose(metrics["distribution_weight"], float(sys.argv[2]))
assert all(math.isfinite(row["distribution_loss"]) for row in metrics["history"])
' "$metrics" "$expected_weight"
}

eval_two_gpu() {
  local model=$1
  local output=$2
  local shards="$output/shards"
  mkdir -p "$shards"
  local pids=()
  for gpu in 2 3; do
    local shard=$((gpu - 2))
    local start=$((shard * 240))
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
      --models flow \
      --flow-weights-dir "$model" \
      --flow-steps 1 \
      --flow-layers 2 \
      --flow-embed-dim 36 \
      --flow-heads 3 \
      --episode-start "$start" \
      --n-trajectories 240 \
      --progress-every 10 \
      --output-dir "$shards/gpu${gpu}" \
      > "$output/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" -eq 0 ]]
  "$PY" scripts/merge_avoiding_eval_shards.py \
    --shards "$shards/gpu2" "$shards/gpu3" \
    --output-dir "$output" \
    > "$output/merge.log" 2>&1
}

echo "TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
train_one 2 1.0 "$D1" &
pid2=$!
train_one 3 10.0 "$D10" &
pid3=$!
failed=0
wait "$pid2" || failed=1
wait "$pid3" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo "TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

validate_model "$D1/model/distillation_metrics.json" 1.0
validate_model "$D10/model/distillation_metrics.json" 10.0

echo "EVALUATING_W1 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
eval_two_gpu "$D1/model" "$D1/eval480"

echo "EVALUATING_W10 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
eval_two_gpu "$D10/model" "$D10/eval480"

echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
