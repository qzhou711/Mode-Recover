#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/head_width_mechanism_audit
BUFFER=logs/avoiding/keep013_width_compression/teacher_buffer_2400/transfer_buffer.pt
TEACHER=logs/avoiding/tinysr_depth_recoverability/selected_model
BUNDLE=logs/avoiding/teacher_deployment_bundle
mkdir -p "$ROOT"

"$PY" - <<'PY' > "$ROOT/audit.json"
import json
print(json.dumps({
    'long_term_objective': 'demo-free transferable cross-architecture and step distillation with mode preservation',
    'short_term_bottleneck': 'FM-3x72-16 -> FM-3x48-16 loses noise-to-mode routing',
    'stage': 'structure distillation only; depth=3 and solver_steps=16 fixed',
    'hypothesis': 'coupled 4->3 attention-head repartition and width projection destroys functional head identity',
    'teacher': 'FM-3x72-16 keep013 Standard-480 91.0%, 24/24',
    'buffer': 'same successful native teacher rollout buffer for all lanes',
    'uses_original_demonstrations': False,
    'uses_expert_actions': False,
    'lanes': ['3x72-h3 head-only', '3x48-h4 per-head coordinate', '3x48-h4 per-head PCA', '3x48-h3 global PCA control'],
    'repair': 'identical velocity + 0.03 endpoint, seed 42, epochs 500',
    'screening': 'initial and epochs 50/100/250/500, Standard-120, four workers/GPU',
    'go_no_go': 'SR>=80% and coverage>=22; only passing candidates merit Standard-480',
}, indent=2))
PY

lane() {
  local gpu=$1 method=$2 embed=$3 heads=$4 out="$ROOT/$2"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_head_width_mechanism_audit.py \
    --bundle-dir "$BUNDLE" --teacher "$TEACHER" --buffer "$BUFFER" \
    --output-dir "$out/model" --method "$method" --epochs 500 --batch-size 256 \
    --max-batches 4 --learning-rate 3e-5 --endpoint-weight .03 \
    --save-epochs 50,100,250,500 --seed 42 > "$out/train.log" 2>&1

  for tag in initial 50 100 250 500; do
    tmp="$out/$tag/model"
    mkdir -p "$tmp"
    if [[ "$tag" == initial ]]; then
      cp "$out/model/initial_flow.pth" "$tmp/eval_best_flow.pth"
    else
      e4=$(printf '%04d' "$tag")
      cp "$out/model/pretrain_epoch_${e4}.pth" "$tmp/eval_best_flow.pth"
    fi
    scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$tmp" "$out/$tag/eval120" \
      120 0 42 4 16 3 "$embed" "$heads" > "$out/eval_${tag}.log" 2>&1
  done
  echo "COMPLETE $(date --iso-8601=seconds)" > "$out/STATUS"
}

lane 0 head_only_3 72 3 & p0=$!
lane 1 per_head_coordinate_4 48 4 & p1=$!
lane 2 per_head_pca_4 48 4 & p2=$!
lane 3 global_pca_3 48 3 & p3=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
if ((failed != 0)); then
  echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi
"$PY" - <<'PY' > "$ROOT/results.json"
import json
from pathlib import Path
root = Path('logs/avoiding/head_width_mechanism_audit')
rows=[]
for lane in ('head_only_3','per_head_coordinate_4','per_head_pca_4','global_pca_3'):
    for tag in ('initial','50','100','250','500'):
        m=json.loads((root/lane/tag/'eval120/metrics.json').read_text())
        rows.append({'lane':lane,'checkpoint':tag,'success_rate':m['success_rate'],
                     'coverage':m['unique_successful_modes'],
                     'entropy':m['normalized_mode_entropy']})
passed=[r for r in rows if r['success_rate']>=.8 and r['coverage']>=22]
print(json.dumps({'protocol':'Standard-120 screening only','rows':rows,'passed':passed},indent=2))
PY
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
