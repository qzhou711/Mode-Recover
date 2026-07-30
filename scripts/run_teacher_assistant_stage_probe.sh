#!/usr/bin/env bash
set -euo pipefail

GPU=$1
TAG=$2
ASSISTANT_EPOCHS=$3
MULTI_NOISE=$4
CROSS_WEIGHT=$5
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/teacher_generated_transfer/transfer_buffer.pt
LANE=logs/avoiding/teacher_generated_structure_wave2_teacher_assistant_stage_probe/"$TAG"
mkdir -p "$LANE/model" "$LANE/assistant_model"

echo "TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_structure_wave2.py \
  --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$LANE/model" \
  --method teacher_assistant --epochs $((ASSISTANT_EPOCHS + 300)) \
  --assistant-epochs "$ASSISTANT_EPOCHS" --student-epochs 300 \
  --assistant-multi-noise "$MULTI_NOISE" \
  --assistant-cross-noise-weight "$CROSS_WEIGHT" \
  --batch-size 256 --max-batches 4 --seed 42 > "$LANE/train.log" 2>&1

cp "$LANE/model/assistant_best_flow.pth" "$LANE/assistant_model/eval_best_flow.pth"
echo "EVALUATING_ASSISTANT_4WORKER $(date --iso-8601=seconds)" > "$LANE/STATUS"
scripts/run_deployed_flow_parallel_eval.sh \
  "$GPU" "$LANE/assistant_model" "$LANE/assistant_eval120" \
  120 0 42 4 16 4 48 4 > "$LANE/assistant_eval120.log" 2>&1

echo "EVALUATING_STUDENT_4WORKER $(date --iso-8601=seconds)" > "$LANE/STATUS"
scripts/run_deployed_flow_parallel_eval.sh \
  "$GPU" "$LANE/model" "$LANE/student_eval120" \
  120 0 42 4 16 3 48 3 > "$LANE/student_eval120.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
