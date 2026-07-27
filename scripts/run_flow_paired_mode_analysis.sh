#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/flow_noise_mapping/paired
EPISODES=1000
WORKERS_PER_GPU=4
mkdir -p "$ROOT"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
  echo "Expected two visible GPUs, found $GPU_COUNT; refusing fallback." >&2
  exit 1
fi
printf 'visible_gpus=%s\nworkers_per_gpu=%s\nepisodes_per_comparison=%s\n' \
  "$GPU_COUNT" "$WORKERS_PER_GPU" "$EPISODES" > "$ROOT/RESOURCES"

run_comparison() {
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
  out="$ROOT/$name"
  mkdir -p "$out/shards"
  pids=()
  shard_dirs=()
  for worker in 0 1 2 3; do
    start=$((worker * EPISODES / WORKERS_PER_GPU))
    end=$(((worker + 1) * EPISODES / WORKERS_PER_GPU))
    count=$((end - start))
    shard="$out/shards/episodes_${start}_$((end - 1))"
    shard_dirs+=("$shard")
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u analyze_flow_paired_rollouts.py \
      --teacher-dir "$teacher" \
      --teacher-layers "$tl" --teacher-embed-dim "$te" --teacher-heads "$th" \
      --student-dir "$student" \
      --student-layers "$sl" --student-embed-dim "$se" --student-heads "$sh" \
      --episode-start "$start" --n-episodes "$count" --progress-every 10 \
      --output-dir "$shard" > "${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "$PY" merge_flow_paired_rollouts.py \
    --shards "${shard_dirs[@]}" \
    --expected-episodes "$EPISODES" \
    --output-dir "$out/merged" > "$out/merge.log" 2>&1
}

gpu0_lane() {
  run_comparison 0 large_to_large \
    logs/avoiding/trained/flow_matching_transformer_5000_seed42 4 72 4 \
    logs/avoiding/flow_auto/direct_16to1_same 4 72 4
  echo LARGE_TO_LARGE_COMPLETE > "$ROOT/GPU0_STATUS"
  run_comparison 0 large_to_small \
    logs/avoiding/trained/flow_matching_transformer_5000_seed42 4 72 4 \
    logs/avoiding/flow_auto/small_geo0 2 36 3
  echo COMPLETE > "$ROOT/GPU0_STATUS"
}

gpu1_lane() {
  run_comparison 1 struct_to_struct \
    logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42 3 48 3 \
    logs/avoiding/flow_capacity_scan/same_3x48/model 3 48 3
  echo STRUCT_TO_STRUCT_COMPLETE > "$ROOT/GPU1_STATUS"
  run_comparison 1 small_to_small \
    logs/avoiding/trained/flow_matching_small16_5000_seed42 2 36 3 \
    logs/avoiding/small_to_small_distill_500/model 2 36 3
  echo COMPLETE > "$ROOT/GPU1_STATUS"
}

echo RUNNING > "$ROOT/PIPELINE_STATUS"
gpu0_lane & pid0=$!
gpu1_lane & pid1=$!
wait "$pid0"
wait "$pid1"
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
