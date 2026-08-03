#!/usr/bin/env bash
set -euo pipefail
GPU=$1;TAG=$2;MODE=$3;SW=$4
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python;ROOT=logs/avoiding/bmd_velocity250_expanded/student_induced_experiments;LANE="$ROOT/$TAG";mkdir -p "$LANE/model";echo "TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_student_induced_bmd.py --buffer logs/avoiding/bmd_velocity250_expanded/student_induced/student_induced_buffer.pt --bundle-dir logs/avoiding/teacher_deployment_bundle --student logs/avoiding/velocity250_mode_ctm/baseline/model/checkpoints/epoch_0100/eval_best_flow.pth --classifier logs/avoiding/bmd_velocity250_expanded/inference_models/shape18_transformer/best.pt --output-dir "$LANE/model" --epochs 250 --mode-weight "$MODE" --sw-weight "$SW" > "$LANE/train.log" 2>&1
for epoch in 50 100 250;do padded=$(printf '%04d' "$epoch");echo "EVALUATING epoch=$epoch $(date --iso-8601=seconds)" > "$LANE/STATUS";scripts/run_deployed_flow_parallel_eval.sh "$GPU" "$LANE/model/checkpoints/epoch_$padded" "$LANE/epoch_$padded/eval120" 120 0 42 4 1 3 48 3 > "$LANE/epoch_$padded.log" 2>&1;done;echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
