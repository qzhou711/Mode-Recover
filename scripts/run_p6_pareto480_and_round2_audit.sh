#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/p6_intervention_pareto_confirm
mkdir -p "$ROOT"
exec 9>"$ROOT/pipeline.lock"
flock 9
if [[ -f "$ROOT/STATUS" ]] && grep -q '^COMPLETE ' "$ROOT/STATUS"; then exit 0; fi
echo "RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS"

# GPU 0/1: confirm the diversity and success Pareto candidates with Standard-480.
scripts/run_deployed_flow_parallel_eval.sh 0 \
  logs/avoiding/p6_intervention_dagger_repair/h4_recovery/selected_model \
  "$ROOT/h4_recovery_eval480" 480 0 42 4 16 3 48 3 \
  > "$ROOT/h4_recovery_eval480.log" 2>&1 & P0=$!
scripts/run_deployed_flow_parallel_eval.sh 1 \
  logs/avoiding/p6_intervention_dagger_repair/h8_all/selected_model \
  "$ROOT/h8_all_eval480" 480 0 42 4 16 3 48 3 \
  > "$ROOT/h8_all_eval480.log" 2>&1 & P1=$!

# GPU 2/3: collect each updated Student's own state distribution for Round 2.
collect_branch() {
  local GPU=$1 NAME=$2 STUDENT=$3 START0=$4
  local OUTROOT="$ROOT/${NAME}_round2_audit"
  mkdir -p "$OUTROOT/shards"
  local pids=()
  for WORKER in 0 1 2 3; do
    (
      for CHUNK in 0 1 2 3 4 5 6 7; do
        START=$((START0 + WORKER * 120 + CHUNK * 15)); END=$((START + 14))
        OUT="$OUTROOT/shards/episodes_${START}_${END}"; mkdir -p "$OUT"
        [[ -s "$OUT/student_induced_buffer.pt" ]] && continue
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u generate_structure_student_induced_buffer.py \
          --bundle-dir logs/avoiding/teacher_deployment_bundle --student "$STUDENT" \
          --output-dir "$OUT" --episode-start "$START" --n-episodes 15 \
          --seed 161803 --progress-every 5 > "$OUT/run.log" 2>&1
      done
    ) & pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]]
  "$PY" merge_structure_student_induced_buffer.py --root "$OUTROOT" \
    --expected-episodes 480 --episode-start "$START0" > "$OUTROOT/merge.log" 2>&1
  "$PY" - "$OUTROOT" <<'PY'
import json, sys, torch
from pathlib import Path
root=Path(sys.argv[1]);d=torch.load(root/'student_induced_buffer.pt',map_location='cpu')
dis=(d['student_endpoints']-d['teacher_corrections']).square().mean((1,2)).sqrt()
audit={'q80':float(torch.quantile(dis,.8)),'q90':float(torch.quantile(dis,.9)),
       'mean':float(dis.mean()),'samples':len(dis)}
(root/'disagreement_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
PY
}
collect_branch 2 h4_recovery \
  logs/avoiding/p6_intervention_dagger_repair/h4_recovery/selected_model 6000 & P2=$!
collect_branch 3 h8_all \
  logs/avoiding/p6_intervention_dagger_repair/h8_all/selected_model 7000 & P3=$!

failed=0
for pid in "$P0" "$P1" "$P2" "$P3"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
