#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250
"$PY" select_bmd_discovery.py --root "$ROOT/discovery" > "$ROOT/discovery/selection.log" 2>&1
BEST=$("$PY" -c "import json;print(json.load(open('$ROOT/discovery/selection.json'))['best']['directory'])")
ALT=$("$PY" -c "import json;print(json.load(open('$ROOT/discovery/selection.json'))['alternative']['directory'])")
echo "CTM_RUNNING best=$BEST alternative=$ALT $(date --iso-8601=seconds)" > "$ROOT/STATUS"
bash scripts/run_bmd_ctm_lane.sh 0 ground_truth_balanced ground_truth "" 0 & P0=$!
bash scripts/run_bmd_ctm_lane.sh 1 bmd_balanced bmd "$BEST/discovery.npz" 0 & P1=$!
bash scripts/run_bmd_ctm_lane.sh 2 bmd_balanced_success bmd "$BEST/discovery.npz" 1 & P2=$!
bash scripts/run_bmd_ctm_lane.sh 3 bmd_alternative bmd "$ALT/discovery.npz" 0 & P3=$!
FAILED=0
wait "$P0" || FAILED=1; wait "$P1" || FAILED=1; wait "$P2" || FAILED=1; wait "$P3" || FAILED=1
if [[ "$FAILED" -ne 0 ]]; then echo "CTM_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
