#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
FM2_INIT=logs/avoiding/trained/flow_matching_small16_5000_seed42
FM3_INIT=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
ROOT=logs/avoiding/flow_full_student_warmstart
FM2="$ROOT/fm_2x36_1_distill_4x72_full_init"
FM3="$ROOT/fm_3x48_1_distill_4x72_full_init"

mkdir -p "$FM2/model" "$FM3/model"
echo TRAINING > "$ROOT/STATUS"

train_warmstart() {
  local gpu=$1
  local init_dir=$2
  local layers=$3
  local embed=$4
  local heads=$5
  local output=$6
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u distill_flow_matching_avoiding.py \
    --teacher-dir "$TEACHER" \
    --teacher-steps 16 \
    --teacher-layers 4 \
    --teacher-embed-dim 72 \
    --teacher-heads 4 \
    --student-steps 1 \
    --student-layers "$layers" \
    --student-embed-dim "$embed" \
    --student-heads "$heads" \
    --student-init checkpoint \
    --student-init-dir "$init_dir" \
    --epochs 500 \
    --batch-size 256 \
    --max-batches-per-epoch 4 \
    --flow-weight 0.1 \
    --geometry-weight 0 \
    --seed 42 \
    --output-dir "$output/model" \
    > "$output/train.log" 2>&1
}

train_warmstart 0 "$FM2_INIT" 2 36 3 "$FM2" &
pid0=$!
train_warmstart 1 "$FM3_INIT" 3 48 3 "$FM3" &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo TRAIN_FAILED > "$ROOT/STATUS"
  exit 1
fi

for model in "$FM2" "$FM3"; do
  test -s "$model/model/eval_best_flow.pth"
  "$PY" - "$model/model/distillation_metrics.json" <<'PY'
import json
import sys
from pathlib import Path
metrics = json.loads(Path(sys.argv[1]).read_text())
assert metrics["student_initialization"] == "checkpoint"
assert metrics["initialization_max_abs_diff"] == 0.0
assert metrics["student_initialization_checkpoint"]
PY
done

eval_two_gpu() {
  local model=$1
  local layers=$2
  local embed=$3
  local heads=$4
  local output=$5
  local shards="$output/shards"
  mkdir -p "$shards"
  for gpu in 0 1; do
    start=$((gpu * 240))
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
      --models flow \
      --flow-weights-dir "$model" \
      --flow-steps 1 \
      --flow-layers "$layers" \
      --flow-embed-dim "$embed" \
      --flow-heads "$heads" \
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

echo EVALUATING_FM2X36 > "$ROOT/STATUS"
eval_two_gpu "$FM2/model" 2 36 3 "$FM2/eval480"
echo EVALUATING_FM3X48 > "$ROOT/STATUS"
eval_two_gpu "$FM3/model" 3 48 3 "$FM3/eval480"

echo COMPLETE > "$ROOT/STATUS"
