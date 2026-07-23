#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
OLD=logs/avoiding/flow_auto
ROOT=logs/avoiding/flow_followup
mkdir -p "$ROOT"
run_eval() {
  gpu=$1; weights=$2; steps=$3; out=$4; shift 4
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$weights" --flow-steps "$steps" --n-trajectories 480 --progress-every 10 --output-dir "$out" "$@" > "${out}.log" 2>&1
}

# Stage 1: confirm the three key 120-rollout findings at n=480.
run_eval 0 "$TEACHER" 16 "$ROOT/teacher16_eval480" & pid0=$!
(
  run_eval 1 "$OLD/direct_16to1_same" 1 "$ROOT/direct16to1_eval480"
  run_eval 1 "$OLD/small_geo0" 1 "$ROOT/small500_geo0_eval480" --flow-layers 2 --flow-embed-dim 36 --flow-heads 3
) & pid1=$!
wait "$pid0"; wait "$pid1"
echo STAGE1_COMPLETE > "$ROOT/PIPELINE_STATUS"

# Stage 2: separate insufficient training from the effect of a conditional diversity constraint.
train_small() {
  gpu=$1; out=$2; geometry=$3; samples=$4
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u distill_flow_matching_avoiding.py --teacher-dir "$TEACHER" --output-dir "$out" --teacher-steps 16 --student-steps 1 --epochs 2000 --batch-size 256 --max-batches-per-epoch 4 --flow-weight 0.1 --geometry-weight "$geometry" --conditional-samples "$samples" --student-layers 2 --student-embed-dim 36 --student-heads 3 > "${out}_train.log" 2>&1
}
train_small 0 "$ROOT/small2000_control" 0.0 1 & pid0=$!
train_small 1 "$ROOT/small2000_conditional" 0.1 4 & pid1=$!
wait "$pid0"; wait "$pid1"
echo STAGE2_COMPLETE > "$ROOT/PIPELINE_STATUS"

# Stage 3: matched 120-rollout screening.
CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$ROOT/small2000_control" --flow-steps 1 --flow-layers 2 --flow-embed-dim 36 --flow-heads 3 --n-trajectories 120 --progress-every 10 --output-dir "$ROOT/small2000_control_eval120" > "$ROOT/small2000_control_eval120.log" 2>&1 & pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$ROOT/small2000_conditional" --flow-steps 1 --flow-layers 2 --flow-embed-dim 36 --flow-heads 3 --n-trajectories 120 --progress-every 10 --output-dir "$ROOT/small2000_conditional_eval120" > "$ROOT/small2000_conditional_eval120.log" 2>&1 & pid1=$!
wait "$pid0"; wait "$pid1"
echo STAGE3_COMPLETE > "$ROOT/PIPELINE_STATUS"

# Stage 4: evaluate both matched variants at n=480, regardless of the screening winner.
CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$ROOT/small2000_control" --flow-steps 1 --flow-layers 2 --flow-embed-dim 36 --flow-heads 3 --n-trajectories 480 --progress-every 10 --output-dir "$ROOT/small2000_control_eval480" > "$ROOT/small2000_control_eval480.log" 2>&1 & pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$ROOT/small2000_conditional" --flow-steps 1 --flow-layers 2 --flow-embed-dim 36 --flow-heads 3 --n-trajectories 480 --progress-every 10 --output-dir "$ROOT/small2000_conditional_eval480" > "$ROOT/small2000_conditional_eval480.log" 2>&1 & pid1=$!
wait "$pid0"; wait "$pid1"
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
