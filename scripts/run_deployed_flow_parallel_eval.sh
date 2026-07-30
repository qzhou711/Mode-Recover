#!/usr/bin/env bash
set -euo pipefail

GPU=$1
MODEL_DIR=$2
OUTPUT_DIR=$3
N_TRAJECTORIES=${4:-120}
EPISODE_START=${5:-0}
SEED=${6:-42}
WORKERS=${7:-4}
STEPS=${8:-16}
LAYERS=${9:-3}
EMBED_DIM=${10:-48}
HEADS=${11:-3}

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUNDLE=logs/avoiding/teacher_deployment_bundle
mkdir -p "$OUTPUT_DIR/shards"
pids=()
shards=()
base=$((N_TRAJECTORIES / WORKERS))
extra=$((N_TRAJECTORIES % WORKERS))
offset=0
for ((worker=0; worker<WORKERS; worker++)); do
  count=$base
  ((worker < extra)) && count=$((count + 1))
  start=$((EPISODE_START + offset))
  shard="$OUTPUT_DIR/shards/worker_${worker}"
  shards+=("$shard")
  mkdir -p "$shard"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u evaluate_deployed_flow.py \
    --bundle-dir "$BUNDLE" --model-dir "$MODEL_DIR" \
    --layers "$LAYERS" --embed-dim "$EMBED_DIM" --heads "$HEADS" --steps "$STEPS" \
    --n-trajectories "$count" --episode-start "$start" --seed "$SEED" \
    --output-dir "$shard" > "$OUTPUT_DIR/worker_${worker}.log" 2>&1 &
  pids+=("$!")
  offset=$((offset + count))
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
((failed == 0))
"$PY" scripts/merge_deployed_flow_eval_shards.py \
  --shards "${shards[@]}" --output-dir "$OUTPUT_DIR"
