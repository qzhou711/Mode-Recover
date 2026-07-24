#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/flow_capacity_scan
MODEL="$ROOT/large_to_3x48/model"
OUT="$ROOT/large_to_3x48/eval480"
SHARDS="$ROOT/large_to_3x48/eval_shards"
mkdir -p "$SHARDS"

run_shard() {
  gpu=$1
  start=$2
  shard=$3
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow \
    --flow-weights-dir "$MODEL" \
    --flow-steps 1 \
    --flow-layers 3 \
    --flow-embed-dim 48 \
    --flow-heads 3 \
    --episode-start "$start" \
    --n-trajectories 240 \
    --progress-every 10 \
    --output-dir "$shard" \
    > "${shard}.log" 2>&1
}

echo PARALLEL_EVAL_RUNNING > "$ROOT/PIPELINE_STATUS"
run_shard 0 0 "$SHARDS/episodes_000_239" & pid0=$!
run_shard 1 240 "$SHARDS/episodes_240_479" & pid1=$!
wait "$pid0"
wait "$pid1"

"$PY" scripts/merge_avoiding_eval_shards.py \
  --shards "$SHARDS/episodes_000_239" "$SHARDS/episodes_240_479" \
  --output-dir "$OUT" \
  > "$ROOT/large_to_3x48/merge.log" 2>&1
echo LARGE_TO_3X48_COMPLETE > "$ROOT/GPU0_STATUS"
"$PY" scripts/plot_flow_capacity_scan.py > "$ROOT/plot.log" 2>&1
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
