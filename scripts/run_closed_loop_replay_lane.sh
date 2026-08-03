#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
TAG="$2"
SAMPLING="$3"
ANCHOR="$4"
CORRECTION="$5"
PCGRAD="$6"
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250_expanded/closed_loop_replay_experiments
LANE="$ROOT/$TAG"
mkdir -p "$LANE/model"
EXTRA=()
if [[ "$PCGRAD" == "1" ]]; then EXTRA+=(--pcgrad); fi
echo "TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_closed_loop_replay.py \
  --teacher-buffer logs/avoiding/bmd_velocity250_expanded/transfer_buffer.pt \
  --discovery logs/avoiding/bmd_velocity250_expanded/hierarchical_k24/shape_nowhite_spectral_seed42/discovery.npz \
  --student-buffer logs/avoiding/bmd_velocity250_expanded/student_induced/student_induced_buffer.pt \
  --student logs/avoiding/velocity250_mode_ctm/baseline/model/checkpoints/epoch_0100/eval_best_flow.pth \
  --output-dir "$LANE/model" --replay-sampling "$SAMPLING" \
  --anchor-weight "$ANCHOR" --correction-weight "$CORRECTION" "${EXTRA[@]}" \
  > "$LANE/train.log" 2>&1
for EPOCH in 50 100 250; do
  E=$(printf '%04d' "$EPOCH")
  echo "EVALUATING epoch=$EPOCH $(date --iso-8601=seconds)" > "$LANE/STATUS"
  bash scripts/run_deployed_flow_parallel_eval.sh "$GPU" \
    "$LANE/model/checkpoints/epoch_$E" "$LANE/epoch_$E/eval120" \
    120 0 42 4 1 3 48 3 > "$LANE/epoch_$E.log" 2>&1
done
echo "COMPLETE $(date --iso-8601=seconds)" > "$LANE/STATUS"
