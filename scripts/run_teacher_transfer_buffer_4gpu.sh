#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/teacher_generated_transfer
mkdir -p "$ROOT/shards"
echo "GENERATING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1 2 3; do
  start=$((gpu*60)); out="$ROOT/shards/episodes_${start}_$((start+59))"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u generate_teacher_transfer_buffer.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle --output-dir "$out" \
    --episode-start "$start" --n-episodes 60 --seed 2027 --progress-every 10 \
    > "$out.log" 2>&1 & pids+=("$!")
done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
if [[ $failed -ne 0 ]]; then echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
echo "SHARDS_COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
