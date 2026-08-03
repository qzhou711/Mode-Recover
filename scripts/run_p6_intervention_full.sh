#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
bash scripts/run_p6_intervention_collection.sh
ROOT=logs/avoiding/p6_intervention_dagger_repair
mkdir -p "$ROOT"
exec 9>"$ROOT/pipeline.lock"
flock 9
echo "RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
p=()
bash scripts/run_p6_intervention_repair_lane.sh 0 h4_all 4 0 & p+=("$!")
bash scripts/run_p6_intervention_repair_lane.sh 1 h4_recovery 4 1 & p+=("$!")
bash scripts/run_p6_intervention_repair_lane.sh 2 h8_all 8 0 & p+=("$!")
bash scripts/run_p6_intervention_repair_lane.sh 3 h8_recovery 8 1 & p+=("$!")
f=0
for x in "${p[@]}"; do wait "$x" || f=1; done
if [[ "$f" == 1 ]]; then echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
