#!/usr/bin/env bash
set -euo pipefail
GPU=$1
TAG=$2
RATIO=$3
BALANCED=$4
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/p6_student_induced_repair_round1
LANE="$ROOT/$TAG"
mkdir -p "$LANE/model"
EXTRA=()
if [[ "$BALANCED" == "1" ]]; then
  EXTRA+=(--balance-base-latents --base-latents logs/avoiding/strong_teacher_expanded_2400/hierarchical_k24/basic_nowhite_full_seed42/discovery.npz)
fi
if [[ ! -s "$LANE/model/pretrain_epoch_0250.pth" ]]; then
  echo "TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_student_induced_structure_repair.py \
    --bundle-dir logs/avoiding/teacher_deployment_bundle \
    --base-buffer logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt \
    --induced-buffer logs/avoiding/p6_student_induced_round1/student_induced_buffer.pt \
    --initial-student logs/avoiding/p6_strong_teacher_structure_repair/velocity_endpoint/selected_model \
    --output-dir "$LANE/model" --induced-ratio "$RATIO" --endpoint-weight 0.03 \
    --epochs 250 --save-epochs 50,100,250 --batch-size 256 --max-batches 4 \
    --learning-rate 3e-5 --seed 42 "${EXTRA[@]}" > "$LANE/train.log" 2>&1
fi
for EPOCH in 50 100 250; do
  EVAL="$LANE/eval120_epoch${EPOCH}"
  if [[ ! -s "$EVAL/metrics.json" ]]; then
    EPOCH4=$(printf '%04d' "$EPOCH"); TMP="$LANE/eval_model_epoch${EPOCH}"; mkdir -p "$TMP"
    cp "$LANE/model/pretrain_epoch_${EPOCH4}.pth" "$TMP/eval_best_flow.pth"
    echo "EVALUATING_120 epoch=$EPOCH $(date --iso-8601=seconds)" > "$LANE/STATUS"
    scripts/run_deployed_flow_parallel_eval.sh "$GPU" "$TMP" "$EVAL" 120 0 42 4 16 3 48 3 > "$LANE/eval120_epoch${EPOCH}.log" 2>&1
  fi
done
"$PY" scripts/select_p6_structure_checkpoint.py --lane "$LANE" --epochs 50 100 250 > "$LANE/selection.log" 2>&1
PASSED=$("$PY" -c "import json;print(int(json.load(open('$LANE/selection.json'))['selected']['passed_gate']))")
if [[ "$PASSED" == "1" && ! -s "$LANE/eval480/metrics.json" ]]; then
  echo "EVALUATING_480 $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh "$GPU" "$LANE/selected_model" "$LANE/eval480" 480 0 42 4 16 3 48 3 > "$LANE/eval480.log" 2>&1
fi
echo "COMPLETE passed_gate=$PASSED $(date --iso-8601=seconds)" > "$LANE/STATUS"
