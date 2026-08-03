#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

ROOT=logs/avoiding/bmd_velocity250/confirm
mkdir -p "$ROOT"

eval480() {
  local gpu=$1 name=$2 model=$3
  local out="$ROOT/$name/standard480_seed42"
  echo "EVALUATING $(date --iso-8601=seconds)" > "$ROOT/$name.STATUS"
  scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$model" "$out" 480 0 42 4 1 3 48 3 \
    > "$ROOT/$name.log" 2>&1
  echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/$name.STATUS"
}

# Select checkpoints by the primary Standard-120 diversity criterion, not training loss:
# baseline e100/e250 both cover 8 (e100 has higher H); oracle e100 covers 11;
# K=16 e250 covers 10 and has the highest H among its checkpoints.
eval480 0 baseline_e100 \
  logs/avoiding/velocity250_mode_ctm/baseline/model/checkpoints/epoch_0100 &
p0=$!
eval480 1 oracle_e100 \
  logs/avoiding/bmd_velocity250/ctm/ground_truth_balanced/model/checkpoints/epoch_0100 &
p1=$!
eval480 2 bmd_k16_e250 \
  logs/avoiding/bmd_velocity250/ctm/bmd_best_scan/model/checkpoints/epoch_0250 &
p2=$!

# First positive experiment: per-z endpoint distribution matching on the isolated
# Velocity-250 -> one-step CTM chain. K=16 is the strongest unsupervised discovery.
MMD=logs/avoiding/bmd_velocity250/ctm/bmd_k16_endpoint_mmd_w01
mkdir -p "$MMD/model"
echo "TRAINING $(date --iso-8601=seconds)" > "$MMD/STATUS"
CUDA_VISIBLE_DEVICES=3 /jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python -u \
  train_teacher_generated_flow_v2.py \
  --bundle-dir logs/avoiding/teacher_deployment_bundle \
  --buffer logs/avoiding/bmd_velocity250/transfer_buffer.pt \
  --output-dir "$MMD/model" \
  --method minilmv2_relation \
  --pretrained-structure logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth \
  --ctm-latent-labels logs/avoiding/bmd_velocity250/discovery/k16_seed42/discovery.npz \
  --ctm-epochs 500 --ctm-dsm-weight 0.1 --ctm-endpoint-mmd-weight 0.1 \
  --save-ctm-epochs 100,250,500 --batch-size 256 --max-batches 4 --seed 42 \
  > "$MMD/train.log" 2>&1
for epoch in 100 250 500; do
  padded=$(printf '%04d' "$epoch")
  echo "EVALUATING epoch=$epoch $(date --iso-8601=seconds)" > "$MMD/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh 3 "$MMD/model/checkpoints/epoch_$padded" \
    "$MMD/epoch_$padded/eval120" 120 0 42 4 1 3 48 3 \
    > "$MMD/epoch_$padded.log" 2>&1
done
echo "COMPLETE $(date --iso-8601=seconds)" > "$MMD/STATUS"

wait "$p0" "$p1" "$p2"
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
