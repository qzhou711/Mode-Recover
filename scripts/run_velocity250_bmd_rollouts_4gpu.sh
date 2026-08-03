#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250
SOURCE=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth
mkdir -p "$ROOT/shards"
test -s "$SOURCE"
echo "RUNNING ROLLOUTS $(date --iso-8601=seconds)" > "$ROOT/STATUS"

run_shard() {
  local gpu=$1 start=$2
  local out="$ROOT/shards/episodes_${start}_$((start+119))"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u generate_teacher_transfer_buffer.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle \
    --model-dir "$SOURCE" --layers 3 --embed-dim 48 --heads 3 --steps 16 \
    --output-dir "$out" --episode-start "$start" --n-episodes 120 \
    --seed 31415 --progress-every 10 > "$out/run.log" 2>&1
}

run_shard 0 0 & P0=$!
run_shard 1 120 & P1=$!
run_shard 2 240 & P2=$!
run_shard 3 360 & P3=$!
FAILED=0
wait "$P0" || FAILED=1
wait "$P1" || FAILED=1
wait "$P2" || FAILED=1
wait "$P3" || FAILED=1
if [[ "$FAILED" -ne 0 ]]; then
  echo "ROLLOUT_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi
echo "MERGING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
"$PY" merge_teacher_transfer_buffer.py --root "$ROOT" > "$ROOT/merge.log" 2>&1

echo "DISCOVERING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
run_discovery() {
  local k=$1 seed=$2
  local out="$ROOT/discovery/k${k}_seed${seed}"
  mkdir -p "$out"
  "$PY" discover_teacher_behavior_modes.py --buffer "$ROOT/transfer_buffer.pt" \
    --output-dir "$out" --clusters "$k" --seed "$seed" > "$out/run.log" 2>&1
}
run_discovery 4 42 & D0=$!
run_discovery 8 42 & D1=$!
run_discovery 16 42 & D2=$!
run_discovery 16 43 & D3=$!
FAILED=0
wait "$D0" || FAILED=1
wait "$D1" || FAILED=1
wait "$D2" || FAILED=1
wait "$D3" || FAILED=1
if [[ "$FAILED" -ne 0 ]]; then
  echo "DISCOVERY_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi
echo "DISCOVERY_COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
bash scripts/run_bmd_ctm_after_discovery_4gpu.sh
