#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
ROOT=logs/avoiding/demonstration_assisted_ctm_oracle
mkdir -p "$ROOT"
echo "RUNNING NON_DEMONSTRATION_FREE $(date --iso-8601=seconds)" > "$ROOT/STATUS"

bash scripts/run_demo_assisted_ctm_lane.sh 0 demonstration_only demonstration &
P0=$!
bash scripts/run_demo_assisted_ctm_lane.sh 1 rollout_demo_50_50 mixed &
P1=$!

FAILED=0
wait "$P0" || FAILED=1
wait "$P1" || FAILED=1
if [[ "$FAILED" -ne 0 ]]; then
  echo "FAILED NON_DEMONSTRATION_FREE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi
echo "COMPLETE NON_DEMONSTRATION_FREE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
