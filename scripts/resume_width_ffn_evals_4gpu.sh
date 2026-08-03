#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
ROOT=logs/avoiding/width_transfer_mechanisms

run_one() {
  local gpu=$1 lane=$2 tag=$3 source model
  if [[ "$tag" == initial ]]; then
    source="$ROOT/$lane/model/initial_flow.pth"
  else
    source="$ROOT/$lane/model/pretrain_epoch_$(printf '%04d' "$tag").pth"
  fi
  model="$ROOT/$lane/$tag/model"
  mkdir -p "$model"
  cp "$source" "$model/eval_best_flow.pth"
  scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$model" "$ROOT/$lane/$tag/eval120" \
    120 0 42 4 16 3 72 4 36 > "$ROOT/$lane/eval_${tag}.log" 2>&1
}

lane_pair() {
  local gpu=$1 lane=$2 first=$3 second=$4
  run_one "$gpu" "$lane" "$first"
  run_one "$gpu" "$lane" "$second"
}

lane_pair 0 ffn_activation initial 50 & p0=$!
lane_pair 1 ffn_activation 250 500 & p1=$!
lane_pair 2 ffn_weight_saliency initial 50 & p2=$!
lane_pair 3 ffn_weight_saliency 250 500 & p3=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
if ((failed != 0)); then
  echo "FFN_EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi
echo "FFN_EVAL_COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
