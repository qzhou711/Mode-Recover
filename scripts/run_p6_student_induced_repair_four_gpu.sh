#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
ROOT=logs/avoiding/p6_student_induced_repair_round1
mkdir -p "$ROOT"
exec 9>"$ROOT/pipeline.lock"; flock 9
if [[ -f "$ROOT/STATUS" ]] && grep -q '^COMPLETE ' "$ROOT/STATUS"; then exit 0; fi
echo "RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
bash scripts/run_p6_student_induced_repair_lane.sh 0 induced025 0.25 0 & pids+=("$!")
bash scripts/run_p6_student_induced_repair_lane.sh 1 induced050 0.50 0 & pids+=("$!")
bash scripts/run_p6_student_induced_repair_lane.sh 2 induced075 0.75 0 & pids+=("$!")
bash scripts/run_p6_student_induced_repair_lane.sh 3 induced050_bmd 0.50 1 & pids+=("$!")
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == "1" ]]; then echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
