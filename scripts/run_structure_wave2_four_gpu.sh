#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/teacher_generated_transfer/transfer_buffer.pt
ROOT=logs/avoiding/teacher_generated_structure_wave2
mkdir -p "$ROOT"

run_lane() {
  local gpu=$1 method=$2 family=$3
  local lane="$ROOT/$method"
  mkdir -p "$lane/model" "$lane/eval120"
  echo "TRAINING $(date --iso-8601=seconds)" > "$lane/STATUS"
  if [[ "$family" == minilm ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_teacher_generated_flow_v2.py \
      --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$lane/model" \
      --method "$method" --pretrain-epochs 300 --batch-size 256 \
      --max-batches 4 --multi-noise 4 --cross-noise-weight 1.0 \
      --pretrain-only --seed 42 > "$lane/train.log" 2>&1
    cp "$lane/model/structure_best_flow.pth" "$lane/model/eval_best_flow.pth"
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_teacher_generated_structure_wave2.py \
      --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$lane/model" \
      --method "$method" --epochs 300 --batch-size 256 --max-batches 4 \
      --seed 42 > "$lane/train.log" 2>&1
  fi
  echo "EVALUATING $(date --iso-8601=seconds)" > "$lane/STATUS"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u evaluate_deployed_flow.py \
    --bundle-dir "$BUNDLE" --model-dir "$lane/model" \
    --layers 3 --embed-dim 48 --heads 3 --steps 16 \
    --n-trajectories 120 --seed 42 --output-dir "$lane/eval120" \
    > "$lane/eval120.log" 2>&1
  echo "COMPLETE $(date --iso-8601=seconds)" > "$lane/STATUS"
}

if [[ $# -eq 3 ]]; then
  run_lane "$1" "$2" "$3"
  exit
fi

echo "Use: $0 GPU METHOD FAMILY"
