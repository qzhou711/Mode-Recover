#!/usr/bin/env bash
set -euo pipefail
GPU=$1
TAG=$2
MODEL=$3
DISCOVERY=$4
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
OUT=logs/avoiding/bmd_velocity250_expanded/inference_models/$TAG
mkdir -p "$OUT"
echo "TRAINING $(date --iso-8601=seconds)" > "$OUT/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_bmd_inference_model.py \
  --buffer logs/avoiding/bmd_velocity250_expanded/transfer_buffer.pt \
  --discovery "$DISCOVERY" --output-dir "$OUT" --model "$MODEL" \
  --epochs 200 --batch-size 128 --seed 42 > "$OUT/train.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$OUT/STATUS"
