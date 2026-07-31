#!/usr/bin/env bash
set -euo pipefail

MODE=$1
GPU=$2
EPOCH=$3
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/teacher_generated_transfer/transfer_buffer.pt
ROOT=logs/avoiding/teacher_generated_minilmv2_repair_ctm_480

if [[ "$EPOCH" == 3500 ]]; then
  SOURCE=logs/avoiding/teacher_generated_minilmv2_4000_scan/epoch_3500/model/eval_best_flow.pth
elif [[ "$EPOCH" == 6000 ]]; then
  SOURCE=logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/epoch_6000/model/eval_best_flow.pth
else
  echo "Unsupported epoch: $EPOCH" >&2
  exit 2
fi

LANE="$ROOT/epoch_${EPOCH}_$MODE"
mkdir -p "$LANE"

if [[ "$MODE" == repair16 ]]; then
  mkdir -p "$LANE/model"
  cp "$SOURCE" "$LANE/model/eval_best_flow.pth"
  echo "EVALUATING_REPAIR16_480 $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$LANE/model" "$LANE/eval480" 480 0 42 4 16 3 48 3 \
    > "$LANE/eval480.log" 2>&1
else
  mkdir -p "$LANE/model"
  echo "CTM_TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$LANE/model" \
    --method minilmv2_relation --pretrained-structure "$SOURCE" \
    --pretrain-epochs 1 --ctm-epochs 500 --batch-size 256 --max-batches 4 \
    --seed 42 > "$LANE/train.log" 2>&1
  echo "EVALUATING_CTM1_480 $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$LANE/model" "$LANE/eval480" 480 0 42 4 1 3 48 3 \
    > "$LANE/eval480.log" 2>&1
fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
