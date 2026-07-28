#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/flow_cd_matrix
BASE="$ROOT/stage1_init/pointwise"
SHARDS="$BASE/eval120_resume_shards"

mkdir -p "$SHARDS/episodes_000_089" "$SHARDS/episodes_090_104" "$SHARDS/episodes_105_119"
cp "$BASE/eval120/rollout_checkpoint.npz" \
  "$SHARDS/episodes_000_089/flow_matching_trajectories.npz"

echo "POINTWISE_RESUME_EVALUATING_90_119 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py \
  --models flow --flow-weights-dir "$BASE/model" --flow-steps 1 \
  --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
  --episode-start 90 --n-trajectories 15 --progress-every 5 \
  --output-dir "$SHARDS/episodes_090_104" \
  > "$SHARDS/episodes_090_104.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "$PY" -u visualize_avoiding.py \
  --models flow --flow-weights-dir "$BASE/model" --flow-steps 1 \
  --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
  --episode-start 105 --n-trajectories 15 --progress-every 5 \
  --output-dir "$SHARDS/episodes_105_119" \
  > "$SHARDS/episodes_105_119.log" 2>&1 &
pid1=$!

failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo "POINTWISE_RESUME_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

"$PY" scripts/merge_avoiding_eval_shards.py \
  --shards \
    "$SHARDS/episodes_000_089" \
    "$SHARDS/episodes_090_104" \
    "$SHARDS/episodes_105_119" \
  --output-dir "$BASE/eval120" \
  > "$BASE/eval120_resume_merge.log" 2>&1

echo "POINTWISE_STAGE1_COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
