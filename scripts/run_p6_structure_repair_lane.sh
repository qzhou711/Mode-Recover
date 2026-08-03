#!/usr/bin/env bash
set -euo pipefail

GPU=$1
TAG=$2
METHOD=$3
VELOCITY_WEIGHT=$4
ENDPOINT_WEIGHT=$5

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/p6_strong_teacher_structure_repair
LANE="$ROOT/$TAG"
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt
SOURCE=logs/avoiding/teacher_generated_minilmv2_4000_to_8000_scan/epoch_6000/model/eval_best_flow.pth
mkdir -p "$LANE/model"

if [[ ! -s "$LANE/model/pretrain_epoch_0500.pth" ]]; then
  echo "TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
  EXTRA=()
  if [[ "$METHOD" == "pca_multinoise_endpoint" ]]; then
    EXTRA+=(--endpoint-weight "$ENDPOINT_WEIGHT")
  else
    EXTRA+=(--relation-velocity-weight "$VELOCITY_WEIGHT" --relation-endpoint-weight "$ENDPOINT_WEIGHT")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_teacher_generated_flow_v2.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$LANE/model" \
    --method "$METHOD" --initial-structure "$SOURCE" --pretrain-success-only \
    --pretrain-epochs 500 --learning-rate 3e-5 --endpoint-steps 16 \
    --save-pretrain-epochs 100,250,500 --batch-size 256 --max-batches 4 \
    --pretrain-only --seed 42 "${EXTRA[@]}" > "$LANE/train.log" 2>&1
fi

for EPOCH in 100 250 500; do
  EVAL="$LANE/eval120_epoch${EPOCH}"
  if [[ ! -s "$EVAL/metrics.json" ]]; then
    EPOCH4=$(printf '%04d' "$EPOCH")
    TMP="$LANE/eval_model_epoch${EPOCH}"
    mkdir -p "$TMP"
    cp "$LANE/model/pretrain_epoch_${EPOCH4}.pth" "$TMP/eval_best_flow.pth"
    echo "EVALUATING_120 epoch=$EPOCH $(date --iso-8601=seconds)" > "$LANE/STATUS"
    scripts/run_deployed_flow_parallel_eval.sh \
      "$GPU" "$TMP" "$EVAL" 120 0 42 4 16 3 48 3 \
      > "$LANE/eval120_epoch${EPOCH}.log" 2>&1
  fi
done

"$PY" scripts/select_p6_structure_checkpoint.py --lane "$LANE" \
  > "$LANE/selection.log" 2>&1
PASSED=$("$PY" -c "import json;print(int(json.load(open('$LANE/selection.json'))['selected']['passed_gate']))")
if [[ "$PASSED" == "1" && ! -s "$LANE/eval480/metrics.json" ]]; then
  echo "EVALUATING_480 $(date --iso-8601=seconds)" > "$LANE/STATUS"
  scripts/run_deployed_flow_parallel_eval.sh \
    "$GPU" "$LANE/selected_model" "$LANE/eval480" 480 0 42 4 16 3 48 3 \
    > "$LANE/eval480.log" 2>&1
fi
echo "COMPLETE passed_gate=$PASSED $(date --iso-8601=seconds)" > "$LANE/STATUS"
