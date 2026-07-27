#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
MODEL=logs/avoiding/trained/flow_matching_transformer_5000_seed42
ROOT=logs/avoiding/flow_solver_baselines/fm_4x72_step1_eval480
SHARDS="$ROOT/shards"

mkdir -p "$SHARDS"
echo RUNNING > "$ROOT/STATUS"

CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py \
  --models flow \
  --flow-weights-dir "$MODEL" \
  --flow-steps 1 \
  --episode-start 0 \
  --n-trajectories 240 \
  --progress-every 10 \
  --output-dir "$SHARDS/episodes_000_239" \
  > "$ROOT/shard0.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "$PY" -u visualize_avoiding.py \
  --models flow \
  --flow-weights-dir "$MODEL" \
  --flow-steps 1 \
  --episode-start 240 \
  --n-trajectories 240 \
  --progress-every 10 \
  --output-dir "$SHARDS/episodes_240_479" \
  > "$ROOT/shard1.log" 2>&1 &
pid1=$!

failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo EVAL_FAILED > "$ROOT/STATUS"
  exit 1
fi

"$PY" scripts/merge_avoiding_eval_shards.py \
  --shards "$SHARDS/episodes_000_239" "$SHARDS/episodes_240_479" \
  --output-dir "$ROOT" \
  > "$ROOT/merge.log" 2>&1

echo COMPLETE > "$ROOT/STATUS"
