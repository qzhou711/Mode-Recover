#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/flow_init_validation_3x48
METHODS=(activation pca early)
pids=()
for gpu in 0 1 2; do
  method=${METHODS[$gpu]}
  out="$ROOT/$method/repair_eval120_step16"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$ROOT/$method/model" --flow-steps 16 \
    --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
    --n-trajectories 120 --progress-every 10 --output-dir "$out" \
    > "$out.log" 2>&1 & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ $failed -ne 0 ]]; then echo "REPAIR_EVAL_FAILED" > "$ROOT/LOCALIZATION_STATUS"; exit 1; fi
echo "REPAIR_EVAL_COMPLETE" > "$ROOT/LOCALIZATION_STATUS"
