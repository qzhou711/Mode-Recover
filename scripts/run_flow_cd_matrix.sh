#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
POINT=logs/avoiding/flow_progressive_compression/intermediate/model
FULL=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
ROOT=logs/avoiding/flow_cd_matrix
INITS=(random teacher_derived pointwise full)

mkdir -p "$ROOT"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -lt 4 ]]; then
  echo "NEED_4_GPUS found=$gpu_count $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

train_lane() {
  local gpu=$1
  local init=$2
  local out="$ROOT/stage1_init/$init"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_flow_consistency_distillation.py \
    --teacher-dir "$TEACHER" --output-dir "$out/model" \
    --student-init "$init" --pointwise-dir "$POINT" --full-dir "$FULL" \
    --epochs 500 --batch-size 256 --max-batches-per-epoch 4 \
    --calibration-batches 8 --time-intervals 16 --flow-weight 0.1 \
    --seed 42 > "$out/train.log" 2>&1
}

evaluate_lane() {
  local gpu=$1
  local weights=$2
  local out=$3
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
    --models flow --flow-weights-dir "$weights" --flow-steps 1 \
    --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 \
    --n-trajectories 120 --progress-every 10 --output-dir "$out" \
    > "${out}.log" 2>&1
}

wait_all() {
  local failed=0
  for pid in "$@"; do wait "$pid" || failed=1; done
  return "$failed"
}

echo "STAGE1_TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1 2 3; do train_lane "$gpu" "${INITS[$gpu]}" & pids+=("$!"); done
if ! wait_all "${pids[@]}"; then
  echo "STAGE1_TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1
fi
for init in "${INITS[@]}"; do
  test -s "$ROOT/stage1_init/$init/model/eval_best_flow.pth"
  test -s "$ROOT/stage1_init/$init/model/consistency_metrics.json"
done

echo "STAGE1_EVALUATING_120 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1 2 3; do
  init=${INITS[$gpu]}
  evaluate_lane "$gpu" "$ROOT/stage1_init/$init/model" \
    "$ROOT/stage1_init/$init/eval120" & pids+=("$!")
done
if ! wait_all "${pids[@]}"; then
  echo "STAGE1_EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1
fi

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for init in ("random", "teacher_derived", "pointwise", "full"):
    train = json.loads((root / "stage1_init" / init / "model/consistency_metrics.json").read_text())
    closed = json.loads((root / "stage1_init" / init / "eval120/metrics.json").read_text())["Flow-Matching"]
    summary[init] = {
        "best_epoch": train["best_epoch"],
        "validation_consistency": train["best_validation_consistency"],
        "closed_loop_120": closed,
    }
(root / "stage1_summary.json").write_text(json.dumps(summary, indent=2))

# Full is an extra-data upper bound. Select the best deployable teacher-only
# candidate between direct structured transfer and pointwise calibration.
def score(name):
    metrics = summary[name]["closed_loop_120"]
    return (
        metrics["success_rate"]
        + 0.30 * metrics["normalized_mode_entropy"]
        + 0.01 * metrics["unique_successful_modes"]
    )
selected = max(("teacher_derived", "pointwise"), key=score)
(root / "SELECTED_INIT").write_text(selected + "\n")
print(json.dumps({"selected": selected, "scores": {x: score(x) for x in ("teacher_derived", "pointwise")}}, indent=2))
PY

SELECTED=$(cat "$ROOT/SELECTED_INIT")
VARIANTS=(pure endpoint_geometry ddil combined)
DIST=(0 0.1 0 0.1)
DDIL=(0 0 0.5 0.5)
SAMPLES=(1 8 1 8)

echo "STAGE2_TRAINING selected_init=$SELECTED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1 2 3; do
  variant=${VARIANTS[$gpu]}
  out="$ROOT/stage2_constraints/$variant"
  mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_flow_consistency_distillation.py \
    --teacher-dir "$TEACHER" --output-dir "$out/model" \
    --student-init "$SELECTED" --pointwise-dir "$POINT" --full-dir "$FULL" \
    --epochs 500 --batch-size 256 --max-batches-per-epoch 4 \
    --calibration-batches 8 --time-intervals 16 --flow-weight 0.1 \
    --conditional-samples "${SAMPLES[$gpu]}" \
    --distribution-weight "${DIST[$gpu]}" --ddil-weight "${DDIL[$gpu]}" \
    --seed 42 > "$out/train.log" 2>&1 & pids+=("$!")
done
if ! wait_all "${pids[@]}"; then
  echo "STAGE2_TRAIN_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1
fi

echo "STAGE2_EVALUATING_120 selected_init=$SELECTED $(date --iso-8601=seconds)" > "$ROOT/STATUS"
pids=()
for gpu in 0 1 2 3; do
  variant=${VARIANTS[$gpu]}
  evaluate_lane "$gpu" "$ROOT/stage2_constraints/$variant/model" \
    "$ROOT/stage2_constraints/$variant/eval120" & pids+=("$!")
done
if ! wait_all "${pids[@]}"; then
  echo "STAGE2_EVAL_FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1
fi

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {"selected_initialization": (root / "SELECTED_INIT").read_text().strip(), "variants": {}}
for variant in ("pure", "endpoint_geometry", "ddil", "combined"):
    train = json.loads((root / "stage2_constraints" / variant / "model/consistency_metrics.json").read_text())
    closed = json.loads((root / "stage2_constraints" / variant / "eval120/metrics.json").read_text())["Flow-Matching"]
    summary["variants"][variant] = {
        "distribution_weight": train["distribution_weight"],
        "ddil_weight": train["ddil_weight"],
        "best_epoch": train["best_epoch"],
        "validation_consistency": train["best_validation_consistency"],
        "closed_loop_120": closed,
        "passes_screen": (
            closed["success_rate"] >= 0.75
            and closed["unique_successful_modes"] >= 18
            and closed["normalized_mode_entropy"] >= 0.75
        ),
    }
(root / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "COMPLETE_STAGE2 $(date --iso-8601=seconds)" > "$ROOT/STATUS"
