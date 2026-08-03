#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/p6_intervention_dagger_round2;STUDENT=logs/avoiding/p6_strong_teacher_structure_repair/velocity_endpoint/selected_model
mkdir -p "$ROOT";exec 9>"$ROOT/collection.lock";flock 9
echo "COLLECTING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
pids=()
for GPU in 0 1 2 3; do
  H=4; START0=4000; LOCAL_GPU=$GPU
  if ((GPU>=2));then H=8;START0=5000;LOCAL_GPU=$((GPU-2));fi
  COND="$ROOT/h${H}";mkdir -p "$COND/shards"
  for WORKER in 0 1 2 3;do
    INDEX=$((LOCAL_GPU*4+WORKER));(
      for CHUNK in 0 1 2 3;do START=$((START0+INDEX*60+CHUNK*15));END=$((START+14));OUT="$COND/shards/episodes_${START}_${END}";mkdir -p "$OUT";[[ -s "$OUT/intervention_buffer.pt" ]]&&continue
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u generate_intervention_dagger_buffer.py --bundle-dir logs/avoiding/teacher_deployment_bundle --student "$STUDENT" --output-dir "$OUT" --episode-start "$START" --n-episodes 15 --takeover-horizon "$H" --threshold 0.4180944411691571 --seed 271828 --progress-every 5 > "$OUT/run.log" 2>&1
      done) & pids+=("$!")
  done
done
failed=0;for p in "${pids[@]}";do wait "$p"||failed=1;done;((failed==0))||{ echo COLLECTION_FAILED > "$ROOT/STATUS";exit 1;}
echo "MERGING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
"$PY" merge_intervention_dagger_buffer.py --root "$ROOT/h4" --episodes 480 --start 4000 --horizon 4 > "$ROOT/h4/merge.log" 2>&1
"$PY" merge_intervention_dagger_buffer.py --root "$ROOT/h8" --episodes 480 --start 5000 --horizon 8 > "$ROOT/h8/merge.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
