#!/usr/bin/env bash
set -euo pipefail

GPU=$1
ASSISTANT_EPOCHS=$2
STUDENT_EPOCHS=$3
TAG=$4
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/teacher_generated_transfer/transfer_buffer.pt
LANE=logs/avoiding/teacher_generated_structure_wave2_teacher_assistant/"$TAG"
mkdir -p "$LANE/model"

echo "TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_structure_wave2.py \
  --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$LANE/model" \
  --method teacher_assistant \
  --epochs $((ASSISTANT_EPOCHS + STUDENT_EPOCHS)) \
  --assistant-epochs "$ASSISTANT_EPOCHS" --student-epochs "$STUDENT_EPOCHS" \
  --batch-size 256 --max-batches 4 --seed 42 > "$LANE/train.log" 2>&1

echo "EVALUATING_4WORKER $(date --iso-8601=seconds)" > "$LANE/STATUS"
scripts/run_deployed_flow_parallel_eval.sh \
  "$GPU" "$LANE/model" "$LANE/eval120" 120 0 42 4 16 3 48 3 \
  > "$LANE/eval120.log" 2>&1
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
