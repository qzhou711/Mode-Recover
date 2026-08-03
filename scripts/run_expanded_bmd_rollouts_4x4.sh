#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250_expanded
SOURCE=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth
BASE=logs/avoiding/bmd_velocity250/transfer_buffer.pt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$ROOT/shards"
echo "ROLLOUT_RUNNING_4GPU_X_4WORKERS $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1 2 3; do
  for worker in 0 1 2 3; do
    index=$((gpu*4+worker))
    start=$((480+index*120))
    out="$ROOT/shards/episodes_${start}_$((start+119))"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u generate_teacher_transfer_buffer.py \
      --bundle-dir logs/avoiding/teacher_deployment_bundle --model-dir "$SOURCE" \
      --layers 3 --embed-dim 48 --heads 3 --steps 16 \
      --output-dir "$out" --episode-start "$start" --n-episodes 120 \
      --seed 31415 --progress-every 10 > "$out/run.log" 2>&1 &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then echo "ROLLOUT_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
echo "MERGING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
"$PY" merge_teacher_transfer_buffer.py --root "$ROOT" --base-buffer "$BASE" > "$ROOT/merge.log" 2>&1
echo "ROLLOUT_COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
