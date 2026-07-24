#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
LARGE=logs/avoiding/trained/flow_matching_transformer_5000_seed42
STRUCTURE=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
ROOT=logs/avoiding/flow_capacity_scan
mkdir -p "$ROOT"

train_distill() {
  gpu=$1
  teacher=$2
  teacher_layers=$3
  teacher_embed=$4
  teacher_heads=$5
  student_layers=$6
  student_embed=$7
  student_heads=$8
  out=$9
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u distill_flow_matching_avoiding.py \
    --teacher-dir "$teacher" \
    --teacher-steps 16 \
    --teacher-layers "$teacher_layers" \
    --teacher-embed-dim "$teacher_embed" \
    --teacher-heads "$teacher_heads" \
    --student-steps 1 \
    --student-layers "$student_layers" \
    --student-embed-dim "$student_embed" \
    --student-heads "$student_heads" \
    --epochs 500 \
    --batch-size 256 \
    --max-batches-per-epoch 4 \
    --flow-weight 0.1 \
    --geometry-weight 0 \
    --seed 42 \
    --output-dir "$out/model" \
    > "$out/train.log" 2>&1
  test -s "$out/model/eval_best_flow.pth"
}

eval_distill() {
  gpu=$1
  layers=$2
  embed=$3
  heads=$4
  out=$5
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow \
    --flow-weights-dir "$out/model" \
    --flow-steps 1 \
    --flow-layers "$layers" \
    --flow-embed-dim "$embed" \
    --flow-heads "$heads" \
    --n-trajectories 480 \
    --progress-every 10 \
    --output-dir "$out/eval480" \
    > "$out/eval480.log" 2>&1
  test -s "$out/eval480/metrics.json"
}

gpu0_lane() {
  out="$ROOT/same_3x48"
  train_distill 0 "$STRUCTURE" 3 48 3 3 48 3 "$out"
  echo SAME_3X48_TRAIN_COMPLETE > "$ROOT/GPU0_STATUS"
  eval_distill 0 3 48 3 "$out"
  echo SAME_3X48_COMPLETE > "$ROOT/GPU0_STATUS"

  out="$ROOT/large_to_3x48"
  train_distill 0 "$LARGE" 4 72 4 3 48 3 "$out"
  echo LARGE_TO_3X48_TRAIN_COMPLETE > "$ROOT/GPU0_STATUS"
  eval_distill 0 3 48 3 "$out"
  echo COMPLETE > "$ROOT/GPU0_STATUS"
}

gpu1_lane() {
  out="$ROOT/large_to_4x64"
  train_distill 1 "$LARGE" 4 72 4 4 64 4 "$out"
  echo LARGE_TO_4X64_TRAIN_COMPLETE > "$ROOT/GPU1_STATUS"
  eval_distill 1 4 64 4 "$out"
  echo COMPLETE > "$ROOT/GPU1_STATUS"
}

echo RUNNING > "$ROOT/PIPELINE_STATUS"
gpu0_lane & gpu0_pid=$!
gpu1_lane & gpu1_pid=$!
wait "$gpu0_pid"
wait "$gpu1_pid"
"$PY" scripts/plot_flow_capacity_scan.py > "$ROOT/plot.log" 2>&1
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
