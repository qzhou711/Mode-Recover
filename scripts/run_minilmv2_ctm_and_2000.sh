#!/usr/bin/env bash
set -euo pipefail

LANE=$1
GPU=$2
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/teacher_generated_transfer/transfer_buffer.pt
ROOT=logs/avoiding/teacher_generated_minilmv2_followup
OUT="$ROOT/$LANE"
mkdir -p "$OUT/model"

if [[ "$LANE" == minilm1000_ctm500 ]]; then
  SOURCE=logs/avoiding/teacher_generated_structure_wave2_1000/minilmv2_relation/model/structure_best_flow.pth
  echo "CTM_TRAINING $(date --iso-8601=seconds)" > "$OUT/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$OUT/model" \
    --method minilmv2_relation --pretrained-structure "$SOURCE" \
    --pretrain-epochs 1 --ctm-epochs 500 --batch-size 256 --max-batches 4 \
    --seed 42 > "$OUT/train.log" 2>&1
  echo "EVALUATING_1STEP_4WORKER $(date --iso-8601=seconds)" > "$OUT/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$OUT/model" "$OUT/eval120" 120 0 42 4 1 3 48 3 \
    > "$OUT/eval120.log" 2>&1
else
  echo "MINILM2000_TRAINING $(date --iso-8601=seconds)" > "$OUT/STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$OUT/model" \
    --method minilmv2_relation --pretrain-epochs 2000 \
    --batch-size 256 --max-batches 4 --pretrain-only --seed 42 \
    > "$OUT/train.log" 2>&1
  cp "$OUT/model/structure_best_flow.pth" "$OUT/model/eval_best_flow.pth"
  echo "EVALUATING_16STEP_4WORKER $(date --iso-8601=seconds)" > "$OUT/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$OUT/model" "$OUT/eval120" 120 0 42 4 16 3 48 3 \
    > "$OUT/eval120.log" 2>&1
fi
echo "COMPLETE $(date --iso-8601=seconds)" > "$OUT/STATUS"
