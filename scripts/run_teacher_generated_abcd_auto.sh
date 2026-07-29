#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUFFER_ROOT=logs/avoiding/teacher_generated_transfer
ROOT=logs/avoiding/teacher_generated_abcd
METHODS=(random activation activation_dynamic pca_dynamic)
mkdir -p "$ROOT"
echo "WAITING_BUFFER $(date --iso-8601=seconds)" > "$ROOT/STATUS"
while true; do
  status=$(cat "$BUFFER_ROOT/STATUS" 2>/dev/null || true)
  [[ "$status" == SHARDS_COMPLETE* ]] && break
  [[ "$status" == FAILED* ]] && { echo "BUFFER_FAILED" > "$ROOT/STATUS"; exit 1; }
  sleep 30
done
echo "MERGING_BUFFER $(date --iso-8601=seconds)" > "$ROOT/STATUS"
"$PY" merge_teacher_transfer_buffer.py --root "$BUFFER_ROOT" > "$BUFFER_ROOT/merge.log" 2>&1
"$PY" - <<'PY'
import json
m=json.load(open('logs/avoiding/teacher_generated_transfer/metrics.json'))
assert not m['uses_original_demonstrations'] and not m['uses_expert_actions']
assert m['success_rate'] >= .85, m
assert m['mode_coverage'] >= 18, m
print(m)
PY
run_train_stage() {
  local smoke=$1 failed=0 pids=()
  for gpu in 0 1 2 3; do
    method=${METHODS[$gpu]}; out="$ROOT/$method/model"; [[ "$smoke" == 1 ]] && out="/tmp/tg_abcd_smoke_$method"
    pre=300;ctm=500;maxb=4; [[ "$smoke" == 1 ]] && { pre=1;ctm=1;maxb=1; }
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_teacher_generated_flow_pipeline.py \
      --bundle-dir logs/avoiding/teacher_deployment_bundle --buffer "$BUFFER_ROOT/transfer_buffer.pt" \
      --output-dir "$out" --method "$method" --pretrain-epochs "$pre" --ctm-epochs "$ctm" \
      --batch-size 256 --max-batches "$maxb" --seed 42 > "${out%/model}/train.log" 2>&1 & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  return "$failed"
}
echo "SMOKE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
for method in "${METHODS[@]}"; do mkdir -p "$ROOT/$method" "/tmp/tg_abcd_smoke_$method"; done
run_train_stage 1 || { echo "SMOKE_FAILED" > "$ROOT/STATUS"; exit 1; }
echo "TRAINING_ABCD $(date --iso-8601=seconds)" > "$ROOT/STATUS"
for method in "${METHODS[@]}"; do mkdir -p "$ROOT/$method/model"; done
run_train_stage 0 || { echo "TRAIN_FAILED" > "$ROOT/STATUS"; exit 1; }
echo "EVAL120 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=();failed=0
for gpu in 0 1 2 3; do
  method=${METHODS[$gpu]};out="$ROOT/$method/eval120";mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u evaluate_deployed_flow.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle --model-dir "$ROOT/$method/model" \
    --layers 3 --embed-dim 48 --heads 3 --steps 1 --n-trajectories 120 --seed 42 --output-dir "$out" \
    > "$ROOT/$method/eval120.log" 2>&1 & pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || failed=1; done
[[ $failed -eq 0 ]] || { echo "EVAL_FAILED" > "$ROOT/STATUS"; exit 1; }
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('logs/avoiding/teacher_generated_abcd');summary={}
for m in ('random','activation','activation_dynamic','pca_dynamic'):
 summary[m]=json.load(open(root/m/'eval120/metrics.json'))
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
PY
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
