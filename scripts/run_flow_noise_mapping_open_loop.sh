#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/flow_noise_mapping
LARGE=logs/avoiding/trained/flow_matching_transformer_5000_seed42
STRUCT=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
SMALL=logs/avoiding/trained/flow_matching_small16_5000_seed42
mkdir -p "$ROOT"

run_mapping() {
  gpu=$1
  name=$2
  teacher=$3
  tl=$4
  te=$5
  th=$6
  student=$7
  sl=$8
  se=$9
  sh=${10}
  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
    analyze_flow_noise_mapping.py \
    --name "$name" \
    --teacher-dir "$teacher" \
    --teacher-layers "$tl" --teacher-embed-dim "$te" --teacher-heads "$th" \
    --student-dir "$student" \
    --student-layers "$sl" --student-embed-dim "$se" --student-heads "$sh" \
    --state-count 12 --samples-per-state 2048 --chunk-size 1024 \
    --output-dir "$ROOT/$name" > "$ROOT/${name}.log" 2>&1
}

gpu0_lane() {
  run_mapping 0 large_to_large "$LARGE" 4 72 4 \
    logs/avoiding/flow_auto/direct_16to1_same 4 72 4
  echo LARGE_TO_LARGE_COMPLETE > "$ROOT/GPU0_STATUS"
  run_mapping 0 large_to_small "$LARGE" 4 72 4 \
    logs/avoiding/flow_auto/small_geo0 2 36 3
  echo COMPLETE > "$ROOT/GPU0_STATUS"
}

gpu1_lane() {
  run_mapping 1 struct_to_struct "$STRUCT" 3 48 3 \
    logs/avoiding/flow_capacity_scan/same_3x48/model 3 48 3
  echo STRUCT_TO_STRUCT_COMPLETE > "$ROOT/GPU1_STATUS"
  run_mapping 1 small_to_small "$SMALL" 2 36 3 \
    logs/avoiding/small_to_small_distill_500/model 2 36 3
  echo COMPLETE > "$ROOT/GPU1_STATUS"
}

echo RUNNING > "$ROOT/OPEN_LOOP_STATUS"
gpu0_lane & pid0=$!
gpu1_lane & pid1=$!
wait "$pid0"
wait "$pid1"
echo COMPLETE > "$ROOT/OPEN_LOOP_STATUS"
