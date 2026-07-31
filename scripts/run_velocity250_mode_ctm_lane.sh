#!/usr/bin/env bash
set -euo pipefail

GPU=$1
TAG=$2
ENDPOINT_WEIGHT=$3
MODE_WEIGHT=$4

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/velocity250_mode_ctm
LANE="$ROOT/$TAG"
MODEL="$LANE/model"
SOURCE=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth
mkdir -p "$MODEL"
test -s "$SOURCE"

if [[ ! -s "$MODEL/checkpoints/epoch_0500/eval_best_flow.pth" ]]; then
  echo "TRAINING endpoint=$ENDPOINT_WEIGHT mode=$MODE_WEIGHT $(date --iso-8601=seconds)" > "$LANE/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle \
    --buffer logs/avoiding/teacher_generated_transfer/transfer_buffer.pt \
    --output-dir "$MODEL" --method minilmv2_relation \
    --pretrained-structure "$SOURCE" \
    --ctm-epochs 500 --ctm-dsm-weight 0.1 \
    --ctm-endpoint-anchor-weight "$ENDPOINT_WEIGHT" \
    --ctm-mode-weight "$MODE_WEIGHT" --ctm-conditional-samples 4 \
    --save-ctm-epochs 100,250,500 \
    --batch-size 256 --max-batches 4 --seed 42 \
    > "$LANE/train.log" 2>&1
fi

for EPOCH in 100 250 500; do
  PADDED=$(printf '%04d' "$EPOCH")
  CHECKPOINT="$MODEL/checkpoints/epoch_${PADDED}"
  OUT="$LANE/epoch_${PADDED}/eval120"
  test -s "$CHECKPOINT/eval_best_flow.pth"
  if [[ -s "$OUT/metrics.json" ]]; then
    continue
  fi
  mkdir -p "$LANE/epoch_${PADDED}"
  echo "EVALUATING epoch=$EPOCH endpoint=$ENDPOINT_WEIGHT mode=$MODE_WEIGHT $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$CHECKPOINT" "$OUT" 120 0 42 4 1 3 48 3 \
    > "$LANE/epoch_${PADDED}/eval120.log" 2>&1
done

echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
