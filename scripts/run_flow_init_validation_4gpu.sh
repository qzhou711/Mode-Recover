#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
ROOT=logs/avoiding/flow_init_validation_3x48
METHODS=(structured activation pca early)
mkdir -p "$ROOT"
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -lt 4 ]]; then
  echo "NEED_4_GPUS $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi
run_parallel() {
  local failed=0
  local pids=()
  for gpu in 0 1 2 3; do
    "$@" "$gpu" "${METHODS[$gpu]}" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}
train_one() {
  local gpu=$1 method=$2
  local out="$ROOT/$method"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_flow_teacher_alignment_repair.py \
    --teacher-dir "$TEACHER" --output-dir "$out/model" \
    --student-layers 3 --student-embed-dim 48 --student-heads 3 \
    --init-method "$method" --epochs 300 --batch-size 256 \
    --max-batches-per-epoch 4 --calibration-batches 8 \
    --conditional-samples 4 --repair-steps 16 \
    --feature-weight 0.1 --endpoint-weight 1.0 --geometry-weight 0.1 \
    --flow-weight 0.1 --learning-rate 1e-4 --seed 42 \
    > "$out/train.log" 2>&1
}
eval_initial_one() {
  local gpu=$1 method=$2
  local out="$ROOT/$method"
  mkdir -p "$out/initial_model" "$out/eval120_initial"
  cp "$out/model/initial_flow.pth" "$out/initial_model/eval_best_flow.pth"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$out/initial_model" --flow-steps 16 \
    --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
    --n-trajectories 120 --progress-every 10 --output-dir "$out/eval120_initial" \
    > "$out/eval120_initial.log" 2>&1
}
eval_repair_one() {
  local gpu=$1 method=$2
  local out="$ROOT/$method"
  mkdir -p "$out/eval120_repair"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$out/model" --flow-steps 16 \
    --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
    --n-trajectories 120 --progress-every 10 --output-dir "$out/eval120_repair" \
    > "$out/eval120_repair.log" 2>&1
}
echo "TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
run_parallel train_one || { echo "TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; }
echo "EVAL_INITIAL $(date --iso-8601=seconds)" > "$ROOT/STATUS"
run_parallel eval_initial_one || { echo "INITIAL_EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; }
echo "EVAL_REPAIR $(date --iso-8601=seconds)" > "$ROOT/STATUS"
run_parallel eval_repair_one || { echo "REPAIR_EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; }
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
