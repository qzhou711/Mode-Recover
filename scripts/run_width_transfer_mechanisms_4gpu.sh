#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/width_transfer_mechanisms
BUFFER=logs/avoiding/keep013_width_compression/teacher_buffer_2400/transfer_buffer.pt
TEACHER=logs/avoiding/tinysr_depth_recoverability/selected_model
BUNDLE=logs/avoiding/teacher_deployment_bundle
mkdir -p "$ROOT"
echo "RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS"

"$PY" - <<'PY' > "$ROOT/audit.json"
import json
print(json.dumps({
  'long_term_objective':'demo-free transferable structure and step distillation for robot/WAM with mode preservation',
  'short_term_bottleneck':'global residual-width compression 72->48 destroys mode routing before repair',
  'stage_targeted':'structure distillation only; depth=3, heads=4 where possible, solver=16 fixed',
  'mechanism_hypothesis':'the shared residual coordinate system, not raw parameter capacity or head count, carries noise-to-mode routing',
  'lanes':{
    'progressive_pca':'InDistill-inspired 72->60->48, 250+250 updates',
    'ppcl_adapter':'72->48 with training-only layerwise linear alignment adapters',
    'ffn_activation':'keep residual72, prune FFN 288->36 by teacher activation energy',
    'ffn_weight_saliency':'keep residual72, prune FFN 288->36 by input-output weight saliency'},
  'matched_control':'existing direct global PCA 72->48: epoch250 SR78.3%, coverage4/24',
  'teacher':'FM-3x72-16 keep013; Standard480 SR91.0%, coverage24/24',
  'data':'same successful native teacher rollout buffer; 140867 states',
  'uses_original_demonstrations':False,
  'uses_expert_actions':False,
  'base_repair':'velocity + 0.03 endpoint; PPCL lane additionally uses documented training-only feature alignment',
  'screening':'initial/50/250/500 as applicable; Standard120 with four workers per GPU',
  'go_no_go':'advance only candidates with SR>=80% and coverage>=22; confirm with Standard480 and seeds',
  'interpretation':'good residual72 FFN lanes implicate shared-coordinate destruction; progressive gain supports capacity-gap curriculum; PPCL gain supports learned alignment'},indent=2))
PY

train_lane() {
  local gpu=$1
  local method=$2
  local out="$ROOT/$method"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_width_transfer_mechanisms.py \
    --bundle-dir "$BUNDLE" --teacher "$TEACHER" --buffer "$BUFFER" \
    --output-dir "$out/model" --method "$method" --epochs 500 --batch-size 256 \
    --learning-rate 3e-5 --endpoint-weight .03 --feature-weight .1 --seed 42 \
    > "$out/train.log" 2>&1
}

if [[ "${SKIP_TRAIN:-0}" != 1 ]]; then
  train_lane 0 progressive_pca & p0=$!
  train_lane 1 ppcl_adapter & p1=$!
  train_lane 2 ffn_activation & p2=$!
  train_lane 3 ffn_weight_saliency & p3=$!
  failed=0
  for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
  if ((failed != 0)); then
    echo "TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
    exit 1
  fi
fi

eval_checkpoint() {
  local gpu=$1 lane=$2 tag=$3 embed=$4 heads=$5 ffn=${6:-} source model
  if [[ "$tag" == initial ]]; then
    source="$ROOT/$lane/model/initial_flow.pth"
  else
    source="$ROOT/$lane/model/pretrain_epoch_$(printf '%04d' "$tag").pth"
  fi
  model="$ROOT/$lane/$tag/model"
  mkdir -p "$model"
  cp "$source" "$model/eval_best_flow.pth"
  scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$model" "$ROOT/$lane/$tag/eval120" \
    120 0 42 4 16 3 "$embed" "$heads" "$ffn" > "$ROOT/$lane/eval_${tag}.log" 2>&1
}

eval_progressive() {
  eval_checkpoint 0 progressive_pca initial 48 4
  eval_checkpoint 0 progressive_pca 500 48 4
}
eval_ppcl() {
  for tag in initial 50 250 500; do eval_checkpoint 1 ppcl_adapter "$tag" 48 4; done
}
eval_ffn_activation() {
  for tag in initial 50 250 500; do eval_checkpoint 2 ffn_activation "$tag" 72 4 36; done
}
eval_ffn_weight() {
  for tag in initial 50 250 500; do eval_checkpoint 3 ffn_weight_saliency "$tag" 72 4 36; done
}
eval_progressive & e0=$!
eval_ppcl & e1=$!
eval_ffn_activation & e2=$!
eval_ffn_weight & e3=$!
failed=0
for pid in "$e0" "$e1" "$e2" "$e3"; do wait "$pid" || failed=1; done
if ((failed != 0)); then
  echo "EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

"$PY" - <<'PY' > "$ROOT/results.json"
import json
from pathlib import Path
root=Path('logs/avoiding/width_transfer_mechanisms')
rows=[]
for lane in ('progressive_pca','ppcl_adapter','ffn_activation','ffn_weight_saliency'):
  for metric in sorted((root/lane).glob('*/eval120/metrics.json')):
    m=json.loads(metric.read_text())
    rows.append({'lane':lane,'checkpoint':metric.parts[-3],
                 'success_rate':m['success_rate'],'coverage':m['unique_successful_modes'],
                 'entropy':m['normalized_mode_entropy']})
passed=[r for r in rows if r['success_rate']>=.8 and r['coverage']>=22]
print(json.dumps({'protocol':'Standard-120 screening; single seed','rows':rows,'passed':passed},indent=2))
PY
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
