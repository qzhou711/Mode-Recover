#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/flow_init_validation_3x48/activation
for epoch in 0010 0025 0050 0100; do
  model="$ROOT/ctm/model/checkpoints/epoch_$epoch"
  out="$ROOT/ctm_epoch_localization/epoch_$epoch"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=3 "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$model" --flow-steps 1 \
    --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
    --n-trajectories 30 --progress-every 10 --output-dir "$out" \
    > "$out.log" 2>&1
done
echo "CTM_EPOCHS_COMPLETE" > logs/avoiding/flow_init_validation_3x48/CTM_LOCALIZATION_STATUS
