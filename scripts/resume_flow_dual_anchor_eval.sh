#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/flow_dual_anchor_scan
W01="$ROOT/fm_2x36_1_distill_4x72_dual_anchor_w01"
W10="$ROOT/fm_2x36_1_distill_4x72_dual_anchor_w10"

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -lt 2 ]]; then
  echo "Need at least two visible GPUs, found $gpu_count" > "$ROOT/STATUS"
  exit 1
fi

eval_two_gpu() {
  local model=$1
  local output=$2
  local shards="$output/shards"
  mkdir -p "$shards"
  local pids=()
  for gpu in 0 1; do
    local start=$((gpu * 240))
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
      --models flow \
      --flow-weights-dir "$model" \
      --flow-steps 1 \
      --flow-layers 2 \
      --flow-embed-dim 36 \
      --flow-heads 3 \
      --episode-start "$start" \
      --n-trajectories 240 \
      --progress-every 10 \
      --output-dir "$shards/gpu${gpu}" \
      > "$output/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" -eq 0 ]]
  "$PY" scripts/merge_avoiding_eval_shards.py \
    --shards "$shards/gpu0" "$shards/gpu1" \
    --output-dir "$output" \
    > "$output/merge.log" 2>&1
}

test -s "$W01/model/eval_best_flow.pth"
test -s "$W10/model/eval_best_flow.pth"

echo "EVALUATING_W01 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
eval_two_gpu "$W01/model" "$W01/eval480"

echo "EVALUATING_W10 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
eval_two_gpu "$W10/model" "$W10/eval480"

echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
