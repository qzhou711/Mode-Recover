#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/small_flow_improvements
mkdir -p "$ROOT"

run_eval() {
  gpu=$1; weights=$2; layers=$3; embed=$4; heads=$5; steps=$6; out=$7
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py --models flow \
    --flow-weights-dir "$weights" --flow-steps "$steps" \
    --flow-layers "$layers" --flow-embed-dim "$embed" --flow-heads "$heads" \
    --n-trajectories 480 --progress-every 10 --output-dir "$out" \
    > "${out}.log" 2>&1
}

structure_lane() {
  weights=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
  CUDA_VISIBLE_DEVICES=0 "$PY" -u run.py agents=flow_matching_transformer_agent \
    agent_name=flow_matching_small_struct_3x48_5000 window_size=5 \
    n_layer=3 n_embd=48 n_head=3 epoch=5000 eval_every_n_epochs=50 \
    train_batch_size=256 val_batch_size=256 num_workers=4 train_only=True \
    hydra.run.dir="$weights" > "$ROOT/structure_train.log" 2>&1
  test -s "$weights/eval_best_flow.pth"
  echo STRUCTURE_TRAIN_COMPLETE > "$ROOT/structure_status"
  run_eval 0 "$weights" 3 48 3 16 "$ROOT/structure_step16_eval480"
  run_eval 0 "$weights" 3 48 3 1 "$ROOT/structure_step1_eval480"
  echo COMPLETE > "$ROOT/structure_status"
}

robust_lane() {
  weights=logs/avoiding/trained/flow_matching_small_robust_2x36_5000_seed42
  CUDA_VISIBLE_DEVICES=1 "$PY" -u run.py agents=flow_matching_transformer_agent \
    agent_name=flow_matching_small_robust_2x36_5000 window_size=5 \
    n_layer=2 n_embd=36 n_head=3 \
    agents.state_noise_std=0.15 agents.state_noise_prob=0.5 \
    epoch=5000 eval_every_n_epochs=50 train_batch_size=256 val_batch_size=256 \
    num_workers=4 train_only=True hydra.run.dir="$weights" \
    > "$ROOT/robust_train.log" 2>&1
  test -s "$weights/eval_best_flow.pth"
  echo ROBUST_TRAIN_COMPLETE > "$ROOT/robust_status"
  run_eval 1 "$weights" 2 36 3 16 "$ROOT/robust_step16_eval480"
  run_eval 1 "$weights" 2 36 3 1 "$ROOT/robust_step1_eval480"
  echo COMPLETE > "$ROOT/robust_status"
}

structure_lane & pid0=$!
robust_lane & pid1=$!
wait "$pid0"
wait "$pid1"
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
