#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250_expanded
SOURCE=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth
BASE=logs/avoiding/bmd_velocity250/transfer_buffer.pt
mkdir -p "$ROOT/shards"
echo "ROLLOUT_RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
run_shard() {
  local gpu=$1 start=$2
  local out="$ROOT/shards/episodes_${start}_$((start+479))"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u generate_teacher_transfer_buffer.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle --model-dir "$SOURCE" \
    --layers 3 --embed-dim 48 --heads 3 --steps 16 \
    --output-dir "$out" --episode-start "$start" --n-episodes 480 \
    --seed 31415 --progress-every 20 > "$out/run.log" 2>&1
}
run_shard 0 480 & p0=$!
run_shard 1 960 & p1=$!
run_shard 2 1440 & p2=$!
run_shard 3 1920 & p3=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
if ((failed)); then echo "ROLLOUT_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
echo "MERGING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
"$PY" merge_teacher_transfer_buffer.py --root "$ROOT" --base-buffer "$BASE" > "$ROOT/merge.log" 2>&1
echo "ROLLOUT_COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
