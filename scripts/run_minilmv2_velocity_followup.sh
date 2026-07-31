#!/usr/bin/env bash
set -euo pipefail

MODE=$1
GPU=$2
TAG=$3
VALUE=$4

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/teacher_generated_minilmv2_velocity_followup
SOURCE_ROOT=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model
LANE="$ROOT/$TAG"
mkdir -p "$LANE/model"

if [[ "$MODE" == "eval_epoch" ]]; then
  EPOCH=$(printf '%04d' "$VALUE")
  SOURCE="$SOURCE_ROOT/pretrain_epoch_${EPOCH}.pth"
  test -s "$SOURCE"
  cp "$SOURCE" "$LANE/model/eval_best_flow.pth"
  echo "EVALUATING epoch=$VALUE $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$LANE/model" "$LANE/eval120" 120 0 42 4 16 3 48 3 \
    > "$LANE/eval120.log" 2>&1
elif [[ "$MODE" == "train_weight" ]]; then
  echo "TRAINING lambda_v=$VALUE $(date --iso-8601=seconds)" > "$LANE/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle \
    --buffer logs/avoiding/teacher_generated_transfer/transfer_buffer.pt \
    --output-dir "$LANE/model" --method minilmv2_relation \
    --initial-structure \
      logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/epoch_6000/model/eval_best_flow.pth \
    --pretrain-epochs 500 --learning-rate 3e-5 \
    --relation-velocity-weight "$VALUE" \
    --relation-endpoint-weight 0 --student-induced-weight 0 \
    --save-pretrain-epochs 100,250,500 \
    --batch-size 256 --max-batches 4 --pretrain-only --seed 42 \
    > "$LANE/train.log" 2>&1
  cp "$LANE/model/structure_best_flow.pth" "$LANE/model/eval_best_flow.pth"
  echo "EVALUATING lambda_v=$VALUE $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$LANE/model" "$LANE/eval120" 120 0 42 4 16 3 48 3 \
    > "$LANE/eval120.log" 2>&1
elif [[ "$MODE" == "train_variant" ]]; then
  RAMP_EPOCHS=${VALUE%%:*}
  BALANCED=${VALUE##*:}
  EXTRA_ARGS=()
  if [[ "$BALANCED" == "1" ]]; then
    EXTRA_ARGS+=(--mode-balanced-sampling)
  fi
  echo "TRAINING ramp=$RAMP_EPOCHS balanced=$BALANCED $(date --iso-8601=seconds)" > "$LANE/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle \
    --buffer logs/avoiding/teacher_generated_transfer/transfer_buffer.pt \
    --output-dir "$LANE/model" --method minilmv2_relation \
    --initial-structure \
      logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/epoch_6000/model/eval_best_flow.pth \
    --pretrain-epochs 250 --learning-rate 3e-5 \
    --relation-velocity-weight 1.0 --velocity-ramp-epochs "$RAMP_EPOCHS" \
    --relation-endpoint-weight 0 --student-induced-weight 0 \
    --save-pretrain-epochs 250 --batch-size 256 --max-batches 4 \
    --pretrain-only --seed 42 "${EXTRA_ARGS[@]}" \
    > "$LANE/train.log" 2>&1
  cp "$LANE/model/pretrain_epoch_0250.pth" "$LANE/model/eval_best_flow.pth"
  echo "EVALUATING ramp=$RAMP_EPOCHS balanced=$BALANCED $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$LANE/model" "$LANE/eval120" 120 0 42 4 16 3 48 3 \
    > "$LANE/eval120.log" 2>&1
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi

echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
