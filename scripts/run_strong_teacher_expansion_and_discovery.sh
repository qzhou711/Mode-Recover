#!/usr/bin/env bash
set -euo pipefail

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/strong_teacher_expanded_2400
BASE=logs/avoiding/teacher_generated_transfer/transfer_buffer.pt
BUNDLE=logs/avoiding/teacher_deployment_bundle
mkdir -p "$ROOT/shards"

# Serialize an interactive runner and a queued Slurm fallback.  The fallback
# blocks here and takes over the saved shards if the interactive allocation is
# cancelled; after successful completion it exits without repeating work.
exec 9>"$ROOT/pipeline.lock"
flock 9
if [[ -f "$ROOT/STATUS" ]] && grep -q '^COMPLETE_P6_1 ' "$ROOT/STATUS"; then
  exit 0
fi
echo "GENERATING $(date --iso-8601=seconds)" > "$ROOT/STATUS"

PIDS=()
for GPU in 0 1 2 3; do
  for WORKER in 0 1 2 3; do
    INDEX=$((GPU * 4 + WORKER))
    # Keep four persistent workers per GPU, but checkpoint every 15 episodes.
    # A cancelled interactive allocation then loses at most one short chunk per
    # worker instead of all 135 episodes.  Completed chunks are skipped on resume.
    (
      for CHUNK in 0 1 2 3 4 5 6 7 8; do
        START=$((240 + INDEX * 135 + CHUNK * 15))
        END=$((START + 14))
        OUT="$ROOT/shards/episodes_${START}_${END}"
        mkdir -p "$OUT"
        if [[ -s "$OUT/transfer_buffer.pt" ]]; then
          continue
        fi
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u generate_teacher_transfer_buffer.py \
          --bundle-dir "$BUNDLE" --output-dir "$OUT" --episode-start "$START" \
          --n-episodes 15 --seed 2027 --progress-every 5 \
          > "$ROOT/shards/episodes_${START}_${END}.log" 2>&1
      done
    ) &
    PIDS+=("$!")
  done
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
if [[ "$FAILED" == "1" ]]; then
  echo "GENERATION_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

echo "MERGING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
"$PY" merge_teacher_transfer_buffer.py --root "$ROOT" --base-buffer "$BASE" \
  > "$ROOT/merge.log" 2>&1
"$PY" validate_strong_teacher_buffer.py --buffer "$ROOT/transfer_buffer.pt" \
  --bundle-dir "$BUNDLE" --output "$ROOT/validation.json" \
  > "$ROOT/validation.log" 2>&1

echo "DISCOVERING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
DISCOVERY="$ROOT/hierarchical_k24"
mkdir -p "$DISCOVERY"
run_one() {
  local NAME=$1 SEED=$2 FEATURE=$3 WHITEN=$4 CLUSTERER=$5
  local OUT="$DISCOVERY/${NAME}_seed${SEED}"
  "$PY" discover_teacher_behavior_modes.py --buffer "$ROOT/transfer_buffer.pt" \
    --output-dir "$OUT" --clusters 24 --seed "$SEED" --feature-kind "$FEATURE" \
    --pca-whiten "$WHITEN" --clusterer "$CLUSTERER" \
    --hierarchical-success --failure-clusters 4 > "$OUT.log" 2>&1
}
PIDS=()
for SEED in 42 43; do
  run_one basic_white_diag "$SEED" basic 1 gmm_diag & PIDS+=("$!")
  run_one basic_nowhite_full "$SEED" basic 0 gmm_full & PIDS+=("$!")
  run_one kinematic_nowhite_full "$SEED" kinematic 0 gmm_full & PIDS+=("$!")
  run_one shape_nowhite_spectral "$SEED" shape 0 spectral & PIDS+=("$!")
done
FAILED=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
if [[ "$FAILED" == "1" ]]; then
  echo "DISCOVERY_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi
"$PY" validate_k24_discovery_scan.py --root "$DISCOVERY" --min-size 20 \
  --min-success 20 --min-stability-nmi 0.8 > "$DISCOVERY/validation.log" 2>&1
echo "COMPLETE_P6_1 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
