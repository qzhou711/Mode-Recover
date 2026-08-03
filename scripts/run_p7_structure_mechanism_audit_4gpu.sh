#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/p7_structure_mechanism_audit
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt
MINILM=logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/epoch_6000/model/eval_best_flow.pth
BASE_STUDENT=logs/avoiding/p6_strong_teacher_structure_repair/velocity_endpoint/selected_model
mkdir -p "$ROOT"

# Frozen audit contract. This experiment is structure-only (16 steps) and demo-free.
"$PY" - <<'PY' > "$ROOT/audit.json"
import json
print(json.dumps({
  "long_term_goal": "demo-free transferable cross-architecture and cross-step WAM distillation with mode preservation",
  "short_term_question": "is low FM-3x48-16 SR caused by teacher OOD recovery, initialization, or the former four-batch training protocol?",
  "stage": "structure_distillation_only",
  "teacher": "FM-4x72-16-Full",
  "student": "FM-3x48-16",
  "solver_steps": 16,
  "uses_original_demonstrations": False,
  "uses_expert_actions": False,
  "controlled_training_variables": ["teacher rollout buffer", "optimizer", "learning rate", "full-pass count", "seed", "evaluation episodes"],
  "independent_variable": "student initialization",
  "lanes": {
    "gpu0": "first-disagreement teacher takeover until episode termination (recovery ceiling)",
    "gpu1": "random init plus 10 complete successful-teacher-buffer passes",
    "gpu2": "teacher-derived PCA init plus the same 10 complete passes",
    "gpu3": "MiniLMv2-6000 init plus the same 10 complete passes"
  },
  "screen": "Standard-120, four workers per GPU",
  "confirmation": "best meaningful candidates receive Standard-480 later",
  "go_no_go": {
    "teacher_recovery": ">=90% means handoff/student resumption is primary; near 82% means teacher is an unreliable oracle on student states",
    "teacher_buffer_train": ">=85% means old optimization/init protocol is primary; near 60% means teacher-only rollout supervision is insufficient or ambiguous",
    "initialization": "same-data differences isolate teacher-derived initialization effects"
  }
}, indent=2))
PY

run_recovery_ceiling() {
  local lane="$ROOT/teacher_takeover_to_end"
  mkdir -p "$lane/shards"
  local pids=()
  for worker in 0 1 2 3; do
    local start=$((8000 + worker * 120))
    local out="$lane/shards/episodes_${start}_$((start + 119))"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=0 "$PY" -u generate_intervention_dagger_buffer.py \
      --bundle-dir "$BUNDLE" --student "$BASE_STUDENT" --output-dir "$out" \
      --episode-start "$start" --n-episodes 120 --takeover-horizon 1 --takeover-to-end \
      --threshold 0.4180944411691571 --seed 271828 --progress-every 10 \
      > "$out/run.log" 2>&1 & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" merge_intervention_dagger_buffer.py --root "$lane" --episodes 480 --start 8000 --horizon -1 \
    > "$lane/merge.log" 2>&1
  echo "COMPLETE $(date --iso-8601=seconds)" > "$lane/STATUS"
}

run_training_lane() {
  local gpu=$1 tag=$2 init=$3
  local lane="$ROOT/$tag"
  mkdir -p "$lane/model"
  local extra=()
  [[ "$init" == random ]] && extra+=(--random-initialization)
  [[ "$init" == minilm ]] && extra+=(--initial-structure "$MINILM")
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$lane/model" \
    --method pca_dynamic --pretrain-success-only --pretrain-epochs 10 \
    --learning-rate 3e-5 --save-pretrain-epochs 1,3,5,10 \
    --batch-size 256 --max-batches 0 --pretrain-only --seed 42 "${extra[@]}" \
    > "$lane/train.log" 2>&1
  for epoch in 1 3 5 10; do
    local e4; e4=$(printf '%04d' "$epoch")
    local model="$lane/eval_model_epoch${epoch}" eval="$lane/eval120_epoch${epoch}"
    mkdir -p "$model"; cp "$lane/model/pretrain_epoch_${e4}.pth" "$model/eval_best_flow.pth"
    scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$model" "$eval" 120 0 42 4 16 3 48 3 \
      > "$lane/eval120_epoch${epoch}.log" 2>&1
  done
  echo "COMPLETE $(date --iso-8601=seconds)" > "$lane/STATUS"
}

run_recovery_ceiling & P0=$!
run_training_lane 1 fullpass_random random & P1=$!
run_training_lane 2 fullpass_pca pca & P2=$!
run_training_lane 3 fullpass_minilm6000 minilm & P3=$!
failed=0
for pid in "$P0" "$P1" "$P2" "$P3"; do wait "$pid" || failed=1; done
echo "COMPLETE failed=$failed $(date --iso-8601=seconds)" > "$ROOT/STATUS"
exit "$failed"
