#!/usr/bin/env bash
set -euo pipefail
GPU=$1
TAG=$2
HORIZON=$3
WEIGHT=$4
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250/hierarchical_trajectory_mmd
LANE="$ROOT/$TAG"
LABELS=logs/avoiding/bmd_velocity250/hierarchical_k24/shape_nowhite_spectral_seed42/discovery.npz
mkdir -p "$LANE/model"
echo "TRAINING horizon=$HORIZON weight=$WEIGHT $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
  --bundle-dir logs/avoiding/teacher_deployment_bundle \
  --buffer logs/avoiding/bmd_velocity250/transfer_buffer.pt \
  --output-dir "$LANE/model" --method minilmv2_relation \
  --pretrained-structure logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth \
  --ctm-latent-labels "$LABELS" --ctm-labels-for-loss-only \
  --ctm-epochs 500 --ctm-dsm-weight 0.1 \
  --ctm-trajectory-mmd-weight "$WEIGHT" --ctm-trajectory-horizon "$HORIZON" \
  --save-ctm-epochs 100,250,500 --batch-size 256 --max-batches 4 --seed 42 \
  > "$LANE/train.log" 2>&1
for epoch in 100 250 500; do
  padded=$(printf '%04d' "$epoch")
  echo "EVALUATING epoch=$epoch $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh "$GPU" "$LANE/model/checkpoints/epoch_$padded" \
    "$LANE/epoch_$padded/eval120" 120 0 42 4 1 3 48 3 \
    > "$LANE/epoch_$padded.log" 2>&1
done
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
