#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

ROOT=logs/avoiding/bmd_velocity250/confirm
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
SOURCE=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth
BUFFER=logs/avoiding/bmd_velocity250/transfer_buffer.pt
LABELS=logs/avoiding/bmd_velocity250/discovery/k16_seed42/discovery.npz

while [[ ! -f "$ROOT/STATUS" ]] || ! grep -q '^COMPLETE' "$ROOT/STATUS"; do
  sleep 30
done

train_eval() {
  local gpu=$1 kind=$2 seed=$3 selected_epoch=$4
  local lane="$ROOT/${kind}_trainseed${seed}"
  local extra=()
  if [[ "$kind" == oracle ]]; then extra+=(--ctm-ground-truth-mode-balanced); fi
  if [[ "$kind" == bmd_k16 ]]; then extra+=(--ctm-latent-labels "$LABELS"); fi
  mkdir -p "$lane/model"
  echo "TRAINING $(date --iso-8601=seconds)" > "$lane/STATUS"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle \
    --buffer "$BUFFER" --output-dir "$lane/model" \
    --method minilmv2_relation --pretrained-structure "$SOURCE" \
    --ctm-epochs 500 --ctm-dsm-weight 0.1 --save-ctm-epochs 100,250,500 \
    --batch-size 256 --max-batches 4 --seed "$seed" "${extra[@]}" \
    > "$lane/train.log" 2>&1
  local padded
  padded=$(printf '%04d' "$selected_epoch")
  echo "EVALUATING epoch=$selected_epoch $(date --iso-8601=seconds)" > "$lane/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$lane/model/checkpoints/epoch_$padded" \
    "$lane/epoch_$padded/standard480" 480 0 42 4 1 3 48 3 \
    > "$lane/eval480.log" 2>&1
  echo "COMPLETE $(date --iso-8601=seconds)" > "$lane/STATUS"
}

# Fixed checkpoints chosen before seeing seed 43/44 results.
train_eval 0 baseline 43 100 & p0=$!
train_eval 1 oracle 43 100 & p1=$!
train_eval 2 bmd_k16 43 250 & p2=$!
train_eval 3 baseline 44 100 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

train_eval 0 oracle 44 100 & p0=$!
train_eval 1 bmd_k16 44 250 & p1=$!
wait "$p0" "$p1"
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/MULTISEED_STATUS"
