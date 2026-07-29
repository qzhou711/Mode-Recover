#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
ROOT=logs/avoiding/flow_init_validation_3x48
METHODS=(structured activation pca early)
run_ctm() {
  local gpu=$1 method=$2
  local init="$ROOT/$method/model" out="$ROOT/$method/ctm"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_flow_ctm_avoiding.py \
    --teacher-dir "$TEACHER" --output-dir "$out/model" \
    --student-layers 3 --student-embed-dim 48 --student-heads 3 \
    --student-init checkpoint --init-dir "$init" \
    --epochs 500 --batch-size 256 --max-batches-per-epoch 4 \
    --dsm-weight 0.1 --seed 42 --save-epochs 10 25 50 100 250 500 \
    > "$out/train.log" 2>&1
}
eval_ctm() {
  local gpu=$1 method=$2
  local out="$ROOT/$method/ctm"
  mkdir -p "$out/eval120_step1"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$out/model" --flow-steps 1 \
    --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
    --n-trajectories 120 --progress-every 10 --output-dir "$out/eval120_step1" \
    > "$out/eval120_step1.log" 2>&1
}
parallel_stage() {
  local fn=$1 failed=0 pids=()
  for gpu in 0 1 2 3; do "$fn" "$gpu" "${METHODS[$gpu]}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}
echo "CTM_TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
parallel_stage run_ctm || { echo "CTM_TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; }
echo "CTM_EVAL120 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
parallel_stage eval_ctm || { echo "CTM_EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; }
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
