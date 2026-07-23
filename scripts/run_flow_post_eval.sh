#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
ROOT=logs/avoiding/flow_auto
mkdir -p "$ROOT"

$PY - <<'GATE'
import json
from pathlib import Path
p=Path('logs/avoiding/eval/flow_matching_teacher5000_step16_120/metrics.json')
m=json.loads(p.read_text())['Flow-Matching']
print('teacher gate:', m, flush=True)
if m['success_rate'] < 0.80 or m['unique_successful_modes'] < 18:
    raise SystemExit('teacher failed gate: require success>=0.80 and modes>=18')
GATE

lane_solver() {
  for steps in 8 4 2; do
    CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$TEACHER" --flow-steps "$steps" --n-trajectories 120 --progress-every 10 --output-dir "$ROOT/teacher_step${steps}_120" > "$ROOT/teacher_step${steps}_120.log" 2>&1
  done
  CUDA_VISIBLE_DEVICES=0 "$PY" -u distill_flow_matching_avoiding.py --teacher-dir "$TEACHER" --output-dir "$ROOT/direct_16to1_same" --teacher-steps 16 --student-steps 1 --epochs 500 --batch-size 256 --max-batches-per-epoch 4 --flow-weight 0.1 --geometry-weight 0.0 > "$ROOT/direct_16to1_same_train.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$ROOT/direct_16to1_same" --flow-steps 1 --n-trajectories 120 --progress-every 10 --output-dir "$ROOT/direct_16to1_same_eval120" > "$ROOT/direct_16to1_same_eval120.log" 2>&1
}

lane_progressive() {
  prev="$TEACHER"
  teacher_steps=16
  for student_steps in 8 4 2 1; do
    out="$ROOT/progressive_${teacher_steps}to${student_steps}"
    CUDA_VISIBLE_DEVICES=1 "$PY" -u distill_flow_matching_avoiding.py --teacher-dir "$prev" --output-dir "$out" --teacher-steps "$teacher_steps" --student-steps "$student_steps" --epochs 500 --batch-size 256 --max-batches-per-epoch 4 --flow-weight 0.1 --geometry-weight 0.0 > "${out}_train.log" 2>&1
    prev="$out"
    teacher_steps="$student_steps"
  done
  CUDA_VISIBLE_DEVICES=1 "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$prev" --flow-steps 1 --n-trajectories 120 --progress-every 10 --output-dir "$ROOT/progressive_final_eval120" > "$ROOT/progressive_final_eval120.log" 2>&1
}

lane_solver & pid0=$!
lane_progressive & pid1=$!
wait "$pid0"
wait "$pid1"

small_pair() {
  gpu=$1
  tag=$2
  geo=$3
  out="$ROOT/$tag"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u distill_flow_matching_avoiding.py --teacher-dir "$TEACHER" --output-dir "$out" --teacher-steps 16 --student-steps 1 --epochs 500 --batch-size 256 --max-batches-per-epoch 4 --flow-weight 0.1 --geometry-weight "$geo" --student-layers 2 --student-embed-dim 36 --student-heads 3 > "${out}_train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py --models flow --flow-weights-dir "$out" --flow-steps 1 --flow-layers 2 --flow-embed-dim 36 --flow-heads 3 --n-trajectories 120 --progress-every 10 --output-dir "${out}_eval120" > "${out}_eval120.log" 2>&1
}
small_pair 0 small_geo0 0.0 & pid0=$!
small_pair 1 small_geo01 0.1 & pid1=$!
wait "$pid0"
wait "$pid1"
echo COMPLETE > "$ROOT/PIPELINE_STATUS"
