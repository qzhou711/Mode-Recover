#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/tinysr_depth_recoverability
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt
mkdir -p "$ROOT"

"$PY" - <<'PY' > "$ROOT/audit.json"
import json
print(json.dumps({
 "paper_idea":"select depth masks by post-repair recoverability, not static importance",
 "scope":"FM-4x72-16 -> FM-3x72-16 depth-only structure distillation",
 "layer_subsets":[[0,1,2],[0,1,3],[0,2,3],[1,2,3]],
 "initialization":"bitwise exact copy of shared modules and selected teacher blocks",
 "controlled":"same successful 2400-rollout teacher buffer, seed, optimizer, endpoint weight, updates and eval suite",
 "uses_original_demonstrations":False,"uses_expert_actions":False,
 "checkpoints":[50,100,250,500],"screen":"Standard-120, four workers per GPU",
 "gate":"SR>=80% and coverage>=22; only passing selection receives Standard-480",
 "next_stage":"only after confirmation: selected FM-3x72-16 -> FM-3x48-16 width compression",
 "warning":"entropy/coverage are success-conditional; Standard-120 is screening only"
},indent=2))
PY

lane() {
  local gpu=$1 tag=$2 map=$3 out="$ROOT/$2"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_recoverable_depth_pruning.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$out/model" \
    --layer-map "$map" --epochs 500 --batch-size 256 --max-batches 4 \
    --learning-rate 3e-5 --endpoint-weight 0.03 --save-epochs 50,100,250,500 --seed 42 \
    > "$out/train.log" 2>&1
  for epoch in 50 100 250 500; do
    local e4; e4=$(printf '%04d' "$epoch"); local tmp="$out/epoch${epoch}/model"
    mkdir -p "$tmp"; cp "$out/model/pretrain_epoch_${e4}.pth" "$tmp/eval_best_flow.pth"
    scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$tmp" "$out/epoch${epoch}/eval120" \
      120 0 42 4 16 3 72 4 > "$out/eval_epoch${epoch}.log" 2>&1
  done
  echo "COMPLETE $(date --iso-8601=seconds)" > "$out/STATUS"
}

lane 0 keep012 0,1,2 & P0=$!
lane 1 keep013 0,1,3 & P1=$!
lane 2 keep023 0,2,3 & P2=$!
lane 3 keep123 1,2,3 & P3=$!
failed=0; for p in "$P0" "$P1" "$P2" "$P3"; do wait "$p" || failed=1; done
if (( failed )); then echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
"$PY" scripts/select_depth_recoverability.py --root "$ROOT" > "$ROOT/selection.log"
read -r passed lane_name epoch < <("$PY" - <<'PY'
import json
d=json.load(open('logs/avoiding/tinysr_depth_recoverability/selection.json'));s=d['selected']
print(int(d['any_passed']),s['lane'],s['epoch'])
PY
)
if [[ "$passed" == 1 ]]; then
  e4=$(printf '%04d' "$epoch"); src="$ROOT/$lane_name/model/pretrain_epoch_${e4}.pth"
  mkdir -p "$ROOT/selected_model"; cp "$src" "$ROOT/selected_model/eval_best_flow.pth"
  scripts/run_deployed_flow_parallel_eval.sh 0 "$ROOT/selected_model" "$ROOT/selected_eval480" \
    480 0 42 4 16 3 72 4 > "$ROOT/selected_eval480.log" 2>&1
  echo "COMPLETE passed=1 selected=$lane_name epoch=$epoch $(date --iso-8601=seconds)" > "$ROOT/STATUS"
else
  echo "COMPLETE passed=0 selected_screen_only=$lane_name epoch=$epoch $(date --iso-8601=seconds)" > "$ROOT/STATUS"
fi
