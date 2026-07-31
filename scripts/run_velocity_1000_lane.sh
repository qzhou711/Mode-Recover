#!/usr/bin/env bash
set -euo pipefail

GPU=$1
SEED=$2

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/teacher_generated_minilmv2_velocity_1000
LANE="$ROOT/seed_${SEED}"
MODEL="$LANE/model"
mkdir -p "$MODEL"

if [[ ! -s "$MODEL/pretrain_epoch_1000.pth" ]]; then
  echo "TRAINING seed=$SEED $(date --iso-8601=seconds)" > "$LANE/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle \
    --buffer logs/avoiding/teacher_generated_transfer/transfer_buffer.pt \
    --output-dir "$MODEL" --method minilmv2_relation \
    --initial-structure \
      logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/epoch_6000/model/eval_best_flow.pth \
    --pretrain-epochs 1000 --learning-rate 3e-5 \
    --relation-velocity-weight 1.0 --student-induced-weight 0 \
    --relation-endpoint-weight 0 --save-pretrain-epochs 750,1000 \
    --batch-size 256 --max-batches 4 --pretrain-only --seed "$SEED" \
    > "$LANE/train.log" 2>&1
fi

for EPOCH in 750 1000; do
  PADDED=$(printf '%04d' "$EPOCH")
  SOURCE="$MODEL/pretrain_epoch_${PADDED}.pth"
  OUT="$LANE/epoch_${PADDED}"
  test -s "$SOURCE"
  if [[ -s "$OUT/eval120/metrics.json" ]]; then
    continue
  fi
  mkdir -p "$OUT/model"
  cp "$SOURCE" "$OUT/model/eval_best_flow.pth"
  echo "EVALUATING epoch=$EPOCH seed=$SEED $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$OUT/model" "$OUT/eval120" 120 0 42 4 16 3 48 3 \
    > "$OUT/eval120.log" 2>&1
done

echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
