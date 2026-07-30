#!/usr/bin/env bash
set -euo pipefail

MODE=$1
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/teacher_generated_transfer/transfer_buffer.pt
ROOT=logs/avoiding/teacher_generated_minilmv2_4000_scan
MODEL="$ROOT/training/model"
mkdir -p "$MODEL"

if [[ "$MODE" == train ]]; then
  echo "TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$MODEL" \
    --method minilmv2_relation --pretrain-epochs 4000 \
    --save-pretrain-epochs 2500,3000,3500,4000 \
    --batch-size 256 --max-batches 4 --pretrain-only --seed 42 \
    > "$ROOT/train.log" 2>&1
  echo "TRAIN_COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit
fi

GPU=$2
EPOCH=$3
SOURCE="$MODEL/pretrain_epoch_$(printf '%04d' "$EPOCH").pth"
LANE="$ROOT/epoch_$EPOCH"
mkdir -p "$LANE/model"
echo "WAITING_CHECKPOINT $(date --iso-8601=seconds)" > "$LANE/STATUS"
while [[ ! -s "$SOURCE" ]]; do
  sleep 5
done
sleep 2
cp "$SOURCE" "$LANE/model/eval_best_flow.pth"
echo "EVALUATING_4WORKER $(date --iso-8601=seconds)" > "$LANE/STATUS"
scripts/run_deployed_flow_parallel_eval.sh \
  "$GPU" "$LANE/model" "$LANE/eval120" 120 0 42 4 16 3 48 3 \
  > "$LANE/eval120.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
