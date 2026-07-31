#!/usr/bin/env bash
set -euo pipefail

GPU=$1
TAG=$2
VELOCITY_WEIGHT=$3
INDUCED_WEIGHT=$4

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/teacher_generated_minilmv2_velocity_dense_scan
LANE="$ROOT/$TAG"
MODEL="$LANE/model"
EPOCHS=(150 200 250 300 350 500)
mkdir -p "$MODEL"

if [[ ! -s "$MODEL/pretrain_epoch_0500.pth" ]]; then
  echo "TRAINING lambda_v=$VELOCITY_WEIGHT lambda_i=$INDUCED_WEIGHT $(date --iso-8601=seconds)" > "$LANE/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle \
    --buffer logs/avoiding/teacher_generated_transfer/transfer_buffer.pt \
    --output-dir "$MODEL" --method minilmv2_relation \
    --initial-structure \
      logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/epoch_6000/model/eval_best_flow.pth \
    --pretrain-epochs 500 --learning-rate 3e-5 \
    --relation-velocity-weight "$VELOCITY_WEIGHT" \
    --student-induced-weight "$INDUCED_WEIGHT" \
    --relation-endpoint-weight 0 --endpoint-steps 16 \
    --save-pretrain-epochs 150,200,250,300,350,500 \
    --batch-size 256 --max-batches 4 --pretrain-only --seed 42 \
    > "$LANE/train.log" 2>&1
fi

for EPOCH in "${EPOCHS[@]}"; do
  PADDED=$(printf '%04d' "$EPOCH")
  SOURCE="$MODEL/pretrain_epoch_${PADDED}.pth"
  OUT="$LANE/epoch_${PADDED}"
  test -s "$SOURCE"
  if [[ -s "$OUT/eval120/metrics.json" ]]; then
    continue
  fi
  mkdir -p "$OUT/model"
  cp "$SOURCE" "$OUT/model/eval_best_flow.pth"
  echo "EVALUATING epoch=$EPOCH lambda_v=$VELOCITY_WEIGHT lambda_i=$INDUCED_WEIGHT $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$OUT/model" "$OUT/eval120" 120 0 42 4 16 3 48 3 \
    > "$OUT/eval120.log" 2>&1
done

echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
