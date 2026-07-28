#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
FULL=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
ROOT=logs/avoiding/flow_repair_anchor_experiments
mkdir -p "$ROOT"

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -lt 2 ]]; then
  echo "NEED_2_GPUS found=$gpu_count $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

lane_repair() {
  local out="$ROOT/fm_4x54_improved_repair"
  mkdir -p "$out/model" "$out/eval120_step16"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u train_flow_teacher_alignment_repair.py \
    --teacher-dir "$TEACHER" --output-dir "$out/model" \
    --student-layers 4 --student-embed-dim 54 --student-heads 3 \
    --epochs 300 --batch-size 256 --max-batches-per-epoch 4 \
    --calibration-batches 8 --conditional-samples 4 \
    --repair-steps 16 --feature-weight 0.1 \
    --endpoint-weight 1.0 --geometry-weight 0.1 --flow-weight 0.1 \
    --learning-rate 1e-4 --seed 42 > "$out/train.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$out/model" --flow-steps 16 \
    --flow-layers 4 --flow-embed-dim 54 --flow-heads 3 \
    --n-trajectories 120 --progress-every 10 \
    --output-dir "$out/eval120_step16" > "$out/eval120_step16.log" 2>&1
}

lane_anchor() {
  local out="$ROOT/fm_3x48_full_init_anchor01_cd"
  mkdir -p "$out/model" "$out/eval120_step1"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u train_flow_consistency_distillation.py \
    --teacher-dir "$TEACHER" --output-dir "$out/model" \
    --student-init full --full-dir "$FULL" \
    --student-layers 3 --student-embed-dim 48 --student-heads 3 \
    --epochs 500 --batch-size 256 --max-batches-per-epoch 4 \
    --time-intervals 16 --flow-weight 0.1 \
    --teacher-anchor-weight 0.1 --learning-rate 1e-4 \
    --seed 42 > "$out/train.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$out/model" --flow-steps 1 \
    --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
    --n-trajectories 120 --progress-every 10 \
    --output-dir "$out/eval120_step1" > "$out/eval120_step1.log" 2>&1
}

echo "RUNNING repair+anchor $(date --iso-8601=seconds)" > "$ROOT/STATUS"
lane_repair & pid0=$!
lane_anchor & pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
repair_train = json.loads(
    (root / "fm_4x54_improved_repair/model/repair_metrics.json").read_text()
)
repair_eval = json.loads(
    (root / "fm_4x54_improved_repair/eval120_step16/metrics.json").read_text()
)["Flow-Matching"]
anchor_train = json.loads(
    (root / "fm_3x48_full_init_anchor01_cd/model/consistency_metrics.json").read_text()
)
anchor_eval = json.loads(
    (root / "fm_3x48_full_init_anchor01_cd/eval120_step1/metrics.json").read_text()
)["Flow-Matching"]
pure_cd = json.loads(
    Path("logs/avoiding/flow_cd_matrix/stage1_init/full/eval120/metrics.json").read_text()
)["Flow-Matching"]
summary = {
    "fm_4x54_improved_repair": {
        "architecture": repair_train["student_architecture"],
        "parameters": repair_train["student_parameters"],
        "best_epoch": repair_train["best_epoch"],
        "closed_loop_120": repair_eval,
    },
    "fm_3x48_full_init_pure_cd_existing": pure_cd,
    "fm_3x48_full_init_anchor01_cd": {
        "teacher_anchor_weight": anchor_train["teacher_anchor_weight"],
        "best_epoch": anchor_train["best_epoch"],
        "validation_consistency": anchor_train["best_validation_consistency"],
        "closed_loop_120": anchor_eval,
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
