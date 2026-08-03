#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/p8_recovery_curriculum
BUNDLE=logs/avoiding/teacher_deployment_bundle
BASE=logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt
RECOVERY=logs/avoiding/p7_structure_mechanism_audit/teacher_takeover_to_end/intervention_buffer.pt
INIT=logs/avoiding/p6_strong_teacher_structure_repair/velocity_endpoint/selected_model
mkdir -p "$ROOT"

"$PY" - <<'PY' > "$ROOT/audit.json"
import json
print(json.dumps({
 "stage":"16-step structure distillation / recovery curriculum",
 "teacher":"FM-4x72-16-Full","student":"FM-3x48-16","solver_steps":16,
 "uses_original_demonstrations":False,"uses_expert_actions":False,
 "evidence":"teacher takeover-to-end reaches 92.9% SR and 24/24 modes",
 "hypothesis":"the student needs successful post-deviation recovery trajectories, not more ordinary teacher-rollout epochs",
 "controlled":"same init, base buffer, successful recovery buffer, optimizer, endpoint weight, seed and update count",
 "lanes":{"gpu0":"25% recovery","gpu1":"50% recovery","gpu2":"75% recovery","gpu3":"curriculum 75->50->25%"},
 "screen":"checkpoints Standard-120 with 4 workers/GPU",
 "gate":"SR>=75%, coverage>=20; passing candidate proceeds to Standard-480",
 "forbidden":"no CTM/one-step stage and no original demonstrations"
},indent=2))
PY

train_stage() {
  local gpu=$1 ratio=$2 init=$3 out=$4 epochs=$5 saves=$6
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_student_induced_structure_repair.py \
    --bundle-dir "$BUNDLE" --base-buffer "$BASE" --induced-buffer "$RECOVERY" \
    --initial-student "$init" --output-dir "$out" --induced-ratio "$ratio" \
    --induced-recovery-only --induced-success-only --endpoint-weight 0.03 \
    --epochs "$epochs" --save-epochs "$saves" --batch-size 256 --max-batches 4 \
    --learning-rate 3e-5 --seed 42
}

eval_ckpt() {
  local gpu=$1 ckpt=$2 out=$3
  local tmp="$out/model"; mkdir -p "$tmp"; cp "$ckpt" "$tmp/eval_best_flow.pth"
  scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$tmp" "$out/eval120" 120 0 42 4 16 3 48 3
}

fixed_lane() {
  local gpu=$1 ratio=$2 tag=$3 lane="$ROOT/$3"
  mkdir -p "$lane/model"
  train_stage "$gpu" "$ratio" "$INIT" "$lane/model" 250 50,100,250 > "$lane/train.log" 2>&1
  for e in 50 100 250; do
    local e4; e4=$(printf '%04d' "$e")
    eval_ckpt "$gpu" "$lane/model/pretrain_epoch_${e4}.pth" "$lane/epoch${e}" > "$lane/eval_epoch${e}.log" 2>&1
  done
  echo "COMPLETE $(date --iso-8601=seconds)" > "$lane/STATUS"
}

curriculum_lane() {
  local lane="$ROOT/curriculum_75_50_25"; mkdir -p "$lane"
  local init="$INIT"
  local stage=0
  for ratio in 0.75 0.50 0.25; do
    stage=$((stage+1)); local out="$lane/stage${stage}_r${ratio}/model"; mkdir -p "$out"
    train_stage 3 "$ratio" "$init" "$out" 100 100 > "$lane/stage${stage}.log" 2>&1
    init="$out/pretrain_epoch_0100.pth"
    eval_ckpt 3 "$init" "$lane/stage${stage}_r${ratio}" > "$lane/eval_stage${stage}.log" 2>&1
  done
  echo "COMPLETE $(date --iso-8601=seconds)" > "$lane/STATUS"
}

fixed_lane 0 0.25 recovery25 & P0=$!
fixed_lane 1 0.50 recovery50 & P1=$!
fixed_lane 2 0.75 recovery75 & P2=$!
curriculum_lane & P3=$!
failed=0; for p in "$P0" "$P1" "$P2" "$P3"; do wait "$p" || failed=1; done
echo "COMPLETE failed=$failed $(date --iso-8601=seconds)" > "$ROOT/STATUS"
exit "$failed"
