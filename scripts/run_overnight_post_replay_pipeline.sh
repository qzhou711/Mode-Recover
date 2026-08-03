#!/usr/bin/env bash
set -euo pipefail

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
MULTI=logs/avoiding/bmd_velocity250_expanded/closed_loop_replay_multiseed
ROOT=logs/avoiding/bmd_velocity250_expanded/overnight_post_replay
mkdir -p "$ROOT"
echo "WAITING_MULTI_SEED $(date --iso-8601=seconds)" > "$ROOT/STATUS"

while true; do
  READY=1
  for LANE in natural_seed43 natural_seed44 balanced_seed43 balanced_seed44; do
    grep -q '^COMPLETE' "$MULTI/$LANE/STATUS" 2>/dev/null || READY=0
  done
  [[ "$READY" == "1" ]] && break
  sleep 20
done

AVAILABLE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$AVAILABLE" -lt 4 ]]; then
  echo "BLOCKED expected_4_gpus_found_$AVAILABLE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
  exit 1
fi

MODEL=logs/avoiding/bmd_velocity250_expanded/closed_loop_replay_experiments/natural_replay/model/checkpoints/epoch_0100
echo "RUNNING_SOLVER_AND_STRONG_TEACHER $(date --iso-8601=seconds)" > "$ROOT/STATUS"
for ITEM in 0:2 1:4 2:8; do
  GPU=${ITEM%%:*}; STEPS=${ITEM##*:}
  mkdir -p "$ROOT/solver_steps_$STEPS"
  bash scripts/run_deployed_flow_parallel_eval.sh "$GPU" "$MODEL" \
    "$ROOT/solver_steps_$STEPS/eval480" 480 0 42 4 "$STEPS" 3 48 3 \
    > "$ROOT/solver_steps_$STEPS.log" 2>&1 &
done

STRONG="$ROOT/strong_teacher_natural"
mkdir -p "$STRONG/model"
CUDA_VISIBLE_DEVICES=3 "$PY" -u train_closed_loop_replay.py \
  --teacher-buffer logs/avoiding/teacher_generated_transfer/transfer_buffer.pt \
  --student-buffer logs/avoiding/bmd_velocity250_expanded/student_induced/student_induced_buffer.pt \
  --student logs/avoiding/velocity250_mode_ctm/baseline/model/checkpoints/epoch_0100/eval_best_flow.pth \
  --output-dir "$STRONG/model" --replay-sampling natural --seed 42 \
  > "$STRONG/train.log" 2>&1
for EPOCH in 50 100 250; do
  E=$(printf '%04d' "$EPOCH")
  bash scripts/run_deployed_flow_parallel_eval.sh 3 \
    "$STRONG/model/checkpoints/epoch_$E" "$STRONG/epoch_$E/eval120" \
    120 0 42 4 1 3 48 3 > "$STRONG/epoch_$E.log" 2>&1
done
SELECTED=$("$PY" select_replay_checkpoint.py --lane "$STRONG")
E=$(printf '%04d' "$SELECTED")
bash scripts/run_deployed_flow_parallel_eval.sh 3 \
  "$STRONG/model/checkpoints/epoch_$E" "$STRONG/epoch_$E/eval480" \
  480 0 42 4 1 3 48 3 > "$STRONG/epoch_${E}_eval480.log" 2>&1
wait
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
