#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
ROOT=logs/avoiding/flow_teacher_repair_cd
NAMES=(fm_3x48 fm_4x54_head_aligned)
LAYERS=(3 4)
EMBEDS=(48 54)
HEADS=(3 3)

mkdir -p "$ROOT"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -lt 2 ]]; then
  echo "NEED_2_GPUS found=$gpu_count $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

wait_pair() {
  local failed=0
  wait "$1" || failed=1
  wait "$2" || failed=1
  return "$failed"
}

echo "REPAIR_TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1; do
  name=${NAMES[$gpu]}
  out="$ROOT/$name/repair"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_flow_teacher_alignment_repair.py \
    --teacher-dir "$TEACHER" --output-dir "$out/model" \
    --student-layers "${LAYERS[$gpu]}" \
    --student-embed-dim "${EMBEDS[$gpu]}" \
    --student-heads "${HEADS[$gpu]}" \
    --epochs 300 --batch-size 256 --max-batches-per-epoch 4 \
    --calibration-batches 8 --conditional-samples 4 --repair-steps 4 \
    --endpoint-weight 1.0 --geometry-weight 0.1 --flow-weight 0.1 \
    --learning-rate 1e-4 --seed 42 > "$out/train.log" 2>&1 &
  pids+=("$!")
done
if ! wait_pair "${pids[0]}" "${pids[1]}"; then
  echo "REPAIR_TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

echo "REPAIR_EVALUATING_120 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1; do
  name=${NAMES[$gpu]}
  out="$ROOT/$name/repair"
  mkdir -p "$out/eval120_step16"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$out/model" --flow-steps 16 \
    --flow-layers "${LAYERS[$gpu]}" \
    --flow-embed-dim "${EMBEDS[$gpu]}" \
    --flow-heads "${HEADS[$gpu]}" \
    --n-trajectories 120 --progress-every 10 \
    --output-dir "$out/eval120_step16" > "$out/eval120_step16.log" 2>&1 &
  pids+=("$!")
done
if ! wait_pair "${pids[0]}" "${pids[1]}"; then
  echo "REPAIR_EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

echo "CD_TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1; do
  name=${NAMES[$gpu]}
  repair="$ROOT/$name/repair/model"
  out="$ROOT/$name/cd"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_flow_consistency_distillation.py \
    --teacher-dir "$TEACHER" --output-dir "$out/model" \
    --student-init pointwise --pointwise-dir "$repair" \
    --student-layers "${LAYERS[$gpu]}" \
    --student-embed-dim "${EMBEDS[$gpu]}" \
    --student-heads "${HEADS[$gpu]}" \
    --epochs 500 --batch-size 256 --max-batches-per-epoch 4 \
    --time-intervals 16 --flow-weight 0.1 --learning-rate 1e-4 \
    --seed 42 > "$out/train.log" 2>&1 &
  pids+=("$!")
done
if ! wait_pair "${pids[0]}" "${pids[1]}"; then
  echo "CD_TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

echo "CD_EVALUATING_120 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1; do
  name=${NAMES[$gpu]}
  out="$ROOT/$name/cd"
  mkdir -p "$out/eval120_step1"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$out/model" --flow-steps 1 \
    --flow-layers "${LAYERS[$gpu]}" \
    --flow-embed-dim "${EMBEDS[$gpu]}" \
    --flow-heads "${HEADS[$gpu]}" \
    --n-trajectories 120 --progress-every 10 \
    --output-dir "$out/eval120_step1" > "$out/eval120_step1.log" 2>&1 &
  pids+=("$!")
done
if ! wait_pair "${pids[0]}" "${pids[1]}"; then
  echo "CD_EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for name in ("fm_3x48", "fm_4x54_head_aligned"):
    repair = json.loads((root / name / "repair/model/repair_metrics.json").read_text())
    repair_eval = json.loads((root / name / "repair/eval120_step16/metrics.json").read_text())["Flow-Matching"]
    cd = json.loads((root / name / "cd/model/consistency_metrics.json").read_text())
    cd_eval = json.loads((root / name / "cd/eval120_step1/metrics.json").read_text())["Flow-Matching"]
    summary[name] = {
        "architecture": repair["student_architecture"],
        "parameters": repair["student_parameters"],
        "repair_best_epoch": repair["best_epoch"],
        "repair_step16_closed_loop_120": repair_eval,
        "cd_best_epoch": cd["best_epoch"],
        "cd_validation_consistency": cd["best_validation_consistency"],
        "cd_step1_closed_loop_120": cd_eval,
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
