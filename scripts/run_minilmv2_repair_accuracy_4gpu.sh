#!/usr/bin/env bash
set -euo pipefail

GPU=$1
TAG=$2
VELOCITY_WEIGHT=$3
ENDPOINT_WEIGHT=$4
INDUCED_WEIGHT=$5
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/teacher_generated_transfer/transfer_buffer.pt
SOURCE=logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/epoch_6000/model/eval_best_flow.pth
ROOT=logs/avoiding/teacher_generated_minilmv2_repair_accuracy
LANE="$ROOT/$TAG"
mkdir -p "$LANE/model"

echo "TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
  --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$LANE/model" \
  --method minilmv2_relation --initial-structure "$SOURCE" \
  --pretrain-epochs 500 --learning-rate 3e-5 \
  --relation-velocity-weight "$VELOCITY_WEIGHT" \
  --relation-endpoint-weight "$ENDPOINT_WEIGHT" \
  --student-induced-weight "$INDUCED_WEIGHT" --endpoint-steps 16 \
  --save-pretrain-epochs 100,250,500 \
  --batch-size 256 --max-batches 4 --pretrain-only --seed 42 \
  > "$LANE/train.log" 2>&1
cp "$LANE/model/structure_best_flow.pth" "$LANE/model/eval_best_flow.pth"

echo "EVALUATING_4WORKER $(date --iso-8601=seconds)" > "$LANE/STATUS"
scripts/run_deployed_flow_parallel_eval.sh \
  "$GPU" "$LANE/model" "$LANE/eval120" 120 0 42 4 16 3 48 3 \
  > "$LANE/eval120.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
