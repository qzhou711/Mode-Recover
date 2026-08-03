#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
ROOT=logs/avoiding/head_width_mechanism_audit

eval_lane() {
  local gpu=$1 lane=$2 embed=$3 heads=$4
  for tag in initial 250; do
    local source
    if [[ "$tag" == initial ]]; then
      source="$ROOT/$lane/model/initial_flow.pth"
    else
      source="$ROOT/$lane/model/pretrain_epoch_0250.pth"
    fi
    local model="$ROOT/$lane/$tag/model"
    mkdir -p "$model"
    cp "$source" "$model/eval_best_flow.pth"
    scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$model" "$ROOT/$lane/$tag/eval120" \
      120 0 42 4 16 3 "$embed" "$heads" > "$ROOT/$lane/eval_${tag}.log" 2>&1
  done
}

gpu0() {
  eval_lane 0 head_only_3 72 3
  eval_lane 0 per_head_coordinate_4 48 4
}

gpu1() {
  eval_lane 1 per_head_pca_4 48 4
  eval_lane 1 global_pca_3 48 3
}

gpu0 & p0=$!
gpu1 & p1=$!
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
if ((failed != 0)); then
  echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/EVAL_INITIAL_250_STATUS"
  exit 1
fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/EVAL_INITIAL_250_STATUS"
