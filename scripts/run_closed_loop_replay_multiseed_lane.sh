#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
METHOD="$2"
SEED="$3"
SELECTED_EPOCH="$4"
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250_expanded/closed_loop_replay_multiseed
TAG="${METHOD}_seed${SEED}"
LANE="$ROOT/$TAG"
mkdir -p "$LANE/model"
echo "TRAINING method=$METHOD seed=$SEED $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_closed_loop_replay.py \
  --teacher-buffer logs/avoiding/bmd_velocity250_expanded/transfer_buffer.pt \
  --discovery logs/avoiding/bmd_velocity250_expanded/hierarchical_k24/shape_nowhite_spectral_seed42/discovery.npz \
  --student-buffer logs/avoiding/bmd_velocity250_expanded/student_induced/student_induced_buffer.pt \
  --student logs/avoiding/velocity250_mode_ctm/baseline/model/checkpoints/epoch_0100/eval_best_flow.pth \
  --output-dir "$LANE/model" --replay-sampling "$METHOD" --seed "$SEED" \
  > "$LANE/train.log" 2>&1
E=$(printf '%04d' "$SELECTED_EPOCH")
echo "EVALUATING_STANDARD480 fixed_epoch=$SELECTED_EPOCH seed=$SEED $(date --iso-8601=seconds)" \
  > "$LANE/STATUS"
bash scripts/run_deployed_flow_parallel_eval.sh "$GPU" \
  "$LANE/model/checkpoints/epoch_$E" "$LANE/epoch_$E/eval480" \
  480 0 42 4 1 3 48 3 > "$LANE/eval480.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
