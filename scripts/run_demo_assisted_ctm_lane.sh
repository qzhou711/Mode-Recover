#!/usr/bin/env bash
set -euo pipefail

GPU=$1
TAG=$2
DATA_SOURCE=$3

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/demonstration_assisted_ctm_oracle
LANE="$ROOT/$TAG"
MODEL="$LANE/model"
SOURCE=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth
mkdir -p "$MODEL"
test -s "$SOURCE"

echo "NON_DEMONSTRATION_FREE data_source=$DATA_SOURCE $(date --iso-8601=seconds)" > "$LANE/STATUS"
"$PY" -u train_teacher_generated_flow_v2.py \
  --bundle-dir logs/avoiding/teacher_deployment_bundle \
  --buffer logs/avoiding/teacher_generated_transfer/transfer_buffer.pt \
  --demonstration-dir environments/dataset/data/avoiding/data \
  --ctm-data-source "$DATA_SOURCE" \
  --output-dir "$MODEL" --method minilmv2_relation \
  --pretrained-structure "$SOURCE" \
  --ctm-epochs 500 --ctm-dsm-weight 0.1 \
  --save-ctm-epochs 100,250,500 \
  --batch-size 256 --max-batches 4 --seed 42 \
  > "$LANE/train.log" 2>&1

for EPOCH in 100 250 500; do
  PADDED=$(printf '%04d' "$EPOCH")
  CHECKPOINT="$MODEL/checkpoints/epoch_${PADDED}"
  OUT="$LANE/epoch_${PADDED}/eval120"
  test -s "$CHECKPOINT/eval_best_flow.pth"
  mkdir -p "$LANE/epoch_${PADDED}"
  echo "NON_DEMONSTRATION_FREE EVALUATING epoch=$EPOCH data_source=$DATA_SOURCE $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$CHECKPOINT" "$OUT" 120 0 42 4 1 3 48 3 \
    > "$LANE/epoch_${PADDED}/eval120.log" 2>&1
done

echo "COMPLETE NON_DEMONSTRATION_FREE data_source=$DATA_SOURCE $(date --iso-8601=seconds)" > "$LANE/STATUS"
