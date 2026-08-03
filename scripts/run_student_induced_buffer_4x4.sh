#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250_expanded/student_induced
TEACHER=logs/avoiding/teacher_generated_minilmv2_repair_accuracy/velocity_w1/model/pretrain_epoch_0250.pth
STUDENT=logs/avoiding/velocity250_mode_ctm/baseline/model/checkpoints/epoch_0100/eval_best_flow.pth
CLASSIFIER=logs/avoiding/bmd_velocity250_expanded/inference_models/shape18_transformer/best.pt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$ROOT/shards";echo "RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS";pids=()
for gpu in 0 1 2 3; do for worker in 0 1 2 3; do index=$((gpu*4+worker));start=$((3000+index*30));out="$ROOT/shards/episodes_${start}_$((start+29))";mkdir -p "$out";CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u generate_student_induced_buffer.py --bundle-dir logs/avoiding/teacher_deployment_bundle --teacher "$TEACHER" --student "$STUDENT" --classifier "$CLASSIFIER" --output-dir "$out" --episode-start "$start" --n-episodes 30 --progress-every 5 > "$out/run.log" 2>&1 & pids+=("$!");done;done
failed=0;for pid in "${pids[@]}";do wait "$pid"||failed=1;done
if ((failed));then echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS";exit 1;fi
echo "MERGING $(date --iso-8601=seconds)" > "$ROOT/STATUS";"$PY" merge_student_induced_buffer.py --root "$ROOT" > "$ROOT/merge.log" 2>&1;echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
