#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/endpoint_mode_preservation_2x2
TEACHER=logs/avoiding/tinysr_depth_recoverability/selected_model
BUFFER=logs/avoiding/keep013_width_compression/teacher_buffer_2400/transfer_buffer.pt
mkdir -p "$ROOT"

cat > "$ROOT/audit.json" <<'EOF'
{
  "long_term_objective": "demo-free transferable architecture and step distillation for robot/WAM policies while preserving multimodality",
  "short_term_bottleneck": "Endpoint-250 preserves 23/24 modes but shifts normalized entropy from 0.929 to 0.807",
  "stage_targeted": "step distillation only; FM-3x72 architecture fixed",
  "mechanism_hypothesis": "paired endpoint preserves individual targets, while same-state multi-noise and short-horizon geometry protect basin separation and path diversity",
  "causal_design": "2x2 endpoint / endpoint+relation / endpoint+trajectory / endpoint+both; same teacher, initialization, buffer, seed, epochs and evaluator",
  "data_access": "teacher rollout states and online teacher queries only; no demonstrations, expert actions or 24-mode training labels",
  "identity_audit": "main endpoint uses native state-noise pairs; auxiliary fresh noises are queried online from Teacher under the same repeated state",
  "screen": "Standard-120 with seed42 and episodes0-119; SR, successful-mode coverage, entropy and trajectories",
  "go": "SR>=90%, coverage=24, entropy moves from 0.807 toward 0.929; confirm winner with multi-seed Standard-480",
  "resource": "four V100, one factorial cell per GPU, tmux, checkpoints 100/250/500",
  "interpretation": "relation gain supports endpoint-basin geometry mechanism; trajectory-only gain supports path mismatch; no gain falsifies these auxiliary proxies"
}
EOF

train_lane() {
  local gpu=$1 tag=$2 relation=$3 trajectory=$4
  local out="$ROOT/$tag/model"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_deployed_flow_step_distillation.py \
    --teacher-dir "$TEACHER" --buffer "$BUFFER" --output-dir "$out" \
    --method endpoint --epochs 500 --save-epochs 100,250,500 --seed 42 \
    --relation-weight "$relation" --trajectory-weight "$trajectory" \
    --multi-noise 4 --aux-state-groups 32 --trajectory-times 0.25,0.5,0.75 \
    > "$ROOT/$tag/train.log" 2>&1
}

train_lane 0 A_endpoint 0.0 0.0 & p0=$!
train_lane 1 B_relation 0.1 0.0 & p1=$!
train_lane 2 C_trajectory 0.0 0.1 & p2=$!
train_lane 3 D_combined 0.1 0.1 & p3=$!
failed=0; for p in "$p0" "$p1" "$p2" "$p3"; do wait "$p" || failed=1; done
if ((failed)); then echo "TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi

eval_lane() {
  local gpu=$1 tag=$2
  for epoch in 100 250 500; do
    local e4; e4=$(printf '%04d' "$epoch")
    scripts/run_deployed_flow_parallel_eval.sh "$gpu" \
      "$ROOT/$tag/model/checkpoints/epoch_${e4}" "$ROOT/$tag/epoch_${epoch}/eval120" \
      120 0 42 4 1 3 72 4 "" --inference-mode boundary \
      > "$ROOT/$tag/eval_${epoch}.log" 2>&1
  done
}

eval_lane 0 A_endpoint & e0=$!
eval_lane 1 B_relation & e1=$!
eval_lane 2 C_trajectory & e2=$!
eval_lane 3 D_combined & e3=$!
failed=0; for p in "$e0" "$e1" "$e2" "$e3"; do wait "$p" || failed=1; done
if ((failed)); then echo "EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi

"$PY" - <<'PY' > "$ROOT/summary.json"
import json
from pathlib import Path
root=Path('logs/avoiding/endpoint_mode_preservation_2x2')
rows=[]
for tag in ('A_endpoint','B_relation','C_trajectory','D_combined'):
    for epoch in (100,250,500):
        path=root/tag/f'epoch_{epoch}/eval120/metrics.json'
        data=json.load(open(path))
        rows.append({'method':tag,'epoch':epoch,**data})
print(json.dumps({'protocol':'Standard-120 screening only','rows':rows},indent=2))
PY
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
