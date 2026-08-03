#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
ROOT=logs/avoiding/head_width_mechanism_audit

run_one() {
  local gpu=$1 lane=$2 tag=$3 embed=$4 heads=$5 source
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
}

run_one 0 per_head_coordinate_4 initial 48 4 & p0=$!
run_one 1 per_head_coordinate_4 250 48 4 & p1=$!
run_one 2 global_pca_3 250 48 3 & p2=$!
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
wait "$p2" || failed=1
if ((failed != 0)); then
  echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/MISSING_EVAL_STATUS"
  exit 1
fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/MISSING_EVAL_STATUS"
