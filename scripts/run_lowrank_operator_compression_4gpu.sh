#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/lowrank_operator_compression
BUFFER=logs/avoiding/keep013_width_compression/teacher_buffer_2400/transfer_buffer.pt
TEACHER=logs/avoiding/tinysr_depth_recoverability/selected_model
BUNDLE=logs/avoiding/teacher_deployment_bundle
mkdir -p "$ROOT"
echo "RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS"

"$PY" - <<'PY' > "$ROOT/audit.json"
import json
print(json.dumps({
  'long_term_objective':'demo-free transferable structure and step distillation for WAM with mode preservation',
  'short_term_bottleneck':'global residual-width compression collapses modes; concentrated FFN pruning loses execution reliability',
  'stage_targeted':'structure distillation only; depth3/residual72/heads4/steps16 fixed',
  'mechanism_hypothesis':'preserving all residual coordinates while distributing compression across operator ranks preserves noise-to-mode routing',
  'lanes':{
    'uniform_svd':'97,634 params; uniform ranks; weight-SVD initialization',
    'uniform_activation':'97,634 params; identical ranks; rollout-activation-weighted initialization',
    'routing_aware':'93,314 params; higher Q/K rank, moderate V/proj, lower FFN rank',
    'hybrid_balanced':'107,426 params; FFN288->96 plus Q/K rank32 and V/proj rank16'},
  'controls':{
    'teacher':'FM-3x72-16 keep013: Standard480 SR91.0%, coverage24',
    'global_width':'3x48 global PCA: Standard120 SR78.3%, coverage4',
    'ffn_activation':'residual72 FFN36 epoch500: Standard120 SR46.7%, coverage15'},
  'data':'same 140867 successful teacher-rollout states/noises/endpoints',
  'uses_original_demonstrations':False,
  'uses_expert_actions':False,
  'repair':'identical velocity + 0.03 endpoint; 500 epochs; seed42',
  'screening':'initial/50/250/500 Standard120; four workers per GPU',
  'go_no_go':'SR>=80% and coverage>=22; then independent Standard480 and multiple seeds',
  'expected_interpretation':'activation>SVD means task-aware geometry matters; routing-aware gain implicates Q/K; hybrid gain means compression must be distributed'},indent=2))
PY

train_lane() {
  local gpu=$1
  local variant=$2
  local out="$ROOT/$variant"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_lowrank_operator_compression.py \
    --bundle-dir "$BUNDLE" --teacher "$TEACHER" --buffer "$BUFFER" \
    --output-dir "$out/model" --variant "$variant" --epochs 500 --batch-size 256 \
    --learning-rate 3e-5 --endpoint-weight .03 --seed 42 > "$out/train.log" 2>&1
}

train_lane 0 uniform_svd & p0=$!
train_lane 1 uniform_activation & p1=$!
train_lane 2 routing_aware & p2=$!
train_lane 3 hybrid_balanced & p3=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
if ((failed != 0)); then
  echo "TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

eval_lane() {
  local gpu=$1
  local variant=$2
  local out="$ROOT/$variant"
  for tag in initial 50 250 500; do
    local source
    if [[ "$tag" == initial ]]; then
      source="$out/model/initial_flow.pth"
    else
      source="$out/model/pretrain_epoch_$(printf '%04d' "$tag").pth"
    fi
    local model="$out/$tag/model"
    mkdir -p "$model"
    cp "$source" "$model/eval_best_flow.pth"
    cp "$out/model/operator_config.json" "$model/operator_config.json"
    scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$model" "$out/$tag/eval120" \
      120 0 42 4 16 3 72 4 > "$out/eval_${tag}.log" 2>&1
  done
}

eval_lane 0 uniform_svd & e0=$!
eval_lane 1 uniform_activation & e1=$!
eval_lane 2 routing_aware & e2=$!
eval_lane 3 hybrid_balanced & e3=$!
failed=0
for pid in "$e0" "$e1" "$e2" "$e3"; do wait "$pid" || failed=1; done
if ((failed != 0)); then
  echo "EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

"$PY" - <<'PY' > "$ROOT/results.json"
import json
from pathlib import Path
root=Path('logs/avoiding/lowrank_operator_compression')
rows=[]
for lane in ('uniform_svd','uniform_activation','routing_aware','hybrid_balanced'):
  for metric in sorted((root/lane).glob('*/eval120/metrics.json')):
    m=json.loads(metric.read_text())
    rows.append({'lane':lane,'checkpoint':metric.parts[-3],
                 'success_rate':m['success_rate'],'coverage':m['unique_successful_modes'],
                 'entropy':m['normalized_mode_entropy']})
passed=[row for row in rows if row['success_rate']>=.8 and row['coverage']>=22]
print(json.dumps({'protocol':'Standard-120 screening; single seed','rows':rows,'passed':passed},indent=2))
PY
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
