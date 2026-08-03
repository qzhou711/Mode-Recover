#!/usr/bin/env bash
set -euo pipefail
GPU=$1
TAG=$2
BALANCE=$3
LABELS=${4:-}
SUCCESS_ONLY=${5:-0}

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250
LANE="$ROOT/ctm/$TAG"
MODEL="$LANE/model"
SOURCE=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth
mkdir -p "$MODEL"
EXTRA=()
if [[ "$BALANCE" == ground_truth ]]; then EXTRA+=(--ctm-ground-truth-mode-balanced); fi
if [[ "$BALANCE" == bmd ]]; then EXTRA+=(--ctm-latent-labels "$LABELS"); fi
if [[ "$SUCCESS_ONLY" == 1 ]]; then EXTRA+=(--ctm-success-only); fi
echo "TRAINING balance=$BALANCE success_only=$SUCCESS_ONLY $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
  --bundle-dir logs/avoiding/teacher_deployment_bundle \
  --buffer "$ROOT/transfer_buffer.pt" --output-dir "$MODEL" \
  --method minilmv2_relation --pretrained-structure "$SOURCE" \
  --ctm-epochs 500 --ctm-dsm-weight 0.1 --save-ctm-epochs 100,250,500 \
  --batch-size 256 --max-batches 4 --seed 42 "${EXTRA[@]}" > "$LANE/train.log" 2>&1
for EPOCH in 100 250 500; do
  PADDED=$(printf '%04d' "$EPOCH")
  OUT="$LANE/epoch_${PADDED}/eval120"
  mkdir -p "$LANE/epoch_${PADDED}"
  echo "EVALUATING epoch=$EPOCH $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh "$GPU" "$MODEL/checkpoints/epoch_${PADDED}" \
    "$OUT" 120 0 42 4 1 3 48 3 > "$LANE/epoch_${PADDED}/eval120.log" 2>&1
done
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
