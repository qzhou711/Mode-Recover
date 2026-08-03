#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250/k24_discovery
BUFFER=logs/avoiding/bmd_velocity250/transfer_buffer.pt
mkdir -p "$ROOT"

run_one() {
  local name=$1 seed=$2 feature=$3 whiten=$4 clusterer=$5
  local out="$ROOT/${name}_seed${seed}"
  "$PY" discover_teacher_behavior_modes.py \
    --buffer "$BUFFER" --output-dir "$out" --clusters 24 --seed "$seed" \
    --feature-kind "$feature" --pca-whiten "$whiten" --clusterer "$clusterer" \
    > "$out.log" 2>&1
}

pids=()
for seed in 42 43; do
  run_one basic_white_diag "$seed" basic 1 gmm_diag & pids+=("$!")
  run_one basic_nowhite_full "$seed" basic 0 gmm_full & pids+=("$!")
  run_one kinematic_nowhite_full "$seed" kinematic 0 gmm_full & pids+=("$!")
  run_one shape_nowhite_spectral "$seed" shape 0 spectral & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0))
"$PY" validate_k24_discovery_scan.py --root "$ROOT" > "$ROOT/validation.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
