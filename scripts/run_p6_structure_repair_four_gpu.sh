#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
ROOT=logs/avoiding/p6_strong_teacher_structure_repair
mkdir -p "$ROOT"
exec 9>"$ROOT/pipeline.lock"
flock 9
echo "RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS"

pids=()
bash scripts/run_p6_structure_repair_lane.sh 0 natural_velocity activation_dynamic 1.0 0.0 & pids+=("$!")
bash scripts/run_p6_structure_repair_lane.sh 1 velocity_endpoint pca_multinoise_endpoint 1.0 0.03 & pids+=("$!")
bash scripts/run_p6_structure_repair_lane.sh 2 velocity_relation minilmv2_relation 1.0 0.0 & pids+=("$!")
bash scripts/run_p6_structure_repair_lane.sh 3 velocity_relation_endpoint minilmv2_relation 1.0 0.03 & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == "1" ]]; then
  echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
