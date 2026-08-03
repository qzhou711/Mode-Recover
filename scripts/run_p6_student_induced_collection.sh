#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/p6_student_induced_round1
STUDENT=logs/avoiding/p6_strong_teacher_structure_repair/velocity_endpoint/selected_model
mkdir -p "$ROOT/shards"
exec 9>"$ROOT/pipeline.lock"
flock 9
if [[ -f "$ROOT/STATUS" ]] && grep -q '^COMPLETE ' "$ROOT/STATUS"; then exit 0; fi
echo "COLLECTING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
pids=()
for GPU in 0 1 2 3; do
  for WORKER in 0 1 2 3; do
    INDEX=$((GPU * 4 + WORKER))
    (
      for CHUNK in 0 1; do
        START=$((3000 + INDEX * 30 + CHUNK * 15))
        END=$((START + 14))
        OUT="$ROOT/shards/episodes_${START}_${END}"
        mkdir -p "$OUT"
        if [[ -s "$OUT/student_induced_buffer.pt" ]]; then continue; fi
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u generate_structure_student_induced_buffer.py \
          --bundle-dir logs/avoiding/teacher_deployment_bundle --student "$STUDENT" \
          --output-dir "$OUT" --episode-start "$START" --n-episodes 15 \
          --seed 314159 --progress-every 5 > "$OUT/run.log" 2>&1
      done
    ) & pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == "1" ]]; then echo "COLLECTION_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
echo "MERGING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
"$PY" merge_structure_student_induced_buffer.py --root "$ROOT" \
  --expected-episodes 480 --episode-start 3000 > "$ROOT/merge.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
