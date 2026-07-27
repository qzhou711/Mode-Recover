#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
ROOT=logs/avoiding/flow_collapse_probe
CAPACITY="$ROOT/fm_4x72_1_distill_3x48"
DYNAMICS="$ROOT/fm_3x48_1_distill_3x48_dynamics"
EPOCHS=(10 25 50 100 200 500)

mkdir -p "$CAPACITY/model" "$DYNAMICS/model"
echo TRAINING > "$ROOT/STATUS"

CUDA_VISIBLE_DEVICES=0 "$PY" -u distill_flow_matching_avoiding.py \
  --teacher-dir "$TEACHER" \
  --teacher-steps 16 \
  --teacher-layers 3 \
  --teacher-embed-dim 48 \
  --teacher-heads 3 \
  --student-steps 1 \
  --student-layers 4 \
  --student-embed-dim 72 \
  --student-heads 4 \
  --epochs 500 \
  --batch-size 256 \
  --max-batches-per-epoch 4 \
  --flow-weight 0.1 \
  --geometry-weight 0 \
  --seed 42 \
  --output-dir "$CAPACITY/model" \
  > "$CAPACITY/train.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "$PY" -u distill_flow_matching_avoiding.py \
  --teacher-dir "$TEACHER" \
  --teacher-steps 16 \
  --teacher-layers 3 \
  --teacher-embed-dim 48 \
  --teacher-heads 3 \
  --student-steps 1 \
  --student-layers 3 \
  --student-embed-dim 48 \
  --student-heads 3 \
  --epochs 500 \
  --batch-size 256 \
  --max-batches-per-epoch 4 \
  --flow-weight 0.1 \
  --geometry-weight 0 \
  --seed 42 \
  --save-epochs "${EPOCHS[@]}" \
  --output-dir "$DYNAMICS/model" \
  > "$DYNAMICS/train.log" 2>&1 &
pid1=$!

failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo TRAIN_FAILED > "$ROOT/STATUS"
  exit 1
fi
test -s "$CAPACITY/model/eval_best_flow.pth"
for epoch in "${EPOCHS[@]}"; do
  test -s "$DYNAMICS/model/checkpoints/epoch_$(printf '%04d' "$epoch")/eval_best_flow.pth"
done

echo FIXED_STATE_ANALYSIS > "$ROOT/STATUS"
analyze_epoch() {
  local gpu=$1
  local epoch=$2
  local checkpoint="$DYNAMICS/model/checkpoints/epoch_$(printf '%04d' "$epoch")"
  local output="$DYNAMICS/fixed_state/epoch_$(printf '%04d' "$epoch")"
  mkdir -p "$(dirname "$output")"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u analyze_flow_noise_mapping.py \
    --name "FM-3x48-1-Distill-3x48-epoch${epoch}" \
    --teacher-dir "$TEACHER" \
    --teacher-layers 3 \
    --teacher-embed-dim 48 \
    --teacher-heads 3 \
    --teacher-steps 16 \
    --student-dir "$checkpoint" \
    --student-layers 3 \
    --student-embed-dim 48 \
    --student-heads 3 \
    --student-steps 1 \
    --state-count 12 \
    --samples-per-state 2048 \
    --chunk-size 1024 \
    --seed 42 \
    --output-dir "$output" \
    > "${output}.log" 2>&1
}

for i in "${!EPOCHS[@]}"; do
  gpu=$((i % 2))
  analyze_epoch "$gpu" "${EPOCHS[$i]}" &
  pids[$i]=$!
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

eval_two_gpu() {
  local model=$1
  local layers=$2
  local embed=$3
  local heads=$4
  local output=$5
  local shards="$output/shards"
  mkdir -p "$shards"
  for gpu in 0 1; do
    start=$((gpu * 240))
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u visualize_avoiding.py \
      --models flow \
      --flow-weights-dir "$model" \
      --flow-steps 1 \
      --flow-layers "$layers" \
      --flow-embed-dim "$embed" \
      --flow-heads "$heads" \
      --episode-start "$start" \
      --n-trajectories 240 \
      --progress-every 10 \
      --output-dir "$shards/gpu${gpu}" \
      > "$output/gpu${gpu}.log" 2>&1 &
    eval_pids[$gpu]=$!
  done
  for pid in "${eval_pids[@]}"; do
    wait "$pid"
  done
  "$PY" scripts/merge_avoiding_eval_shards.py \
    --shards "$shards/gpu0" "$shards/gpu1" \
    --output-dir "$output" \
    > "$output/merge.log" 2>&1
}

echo EVALUATING_CAPACITY > "$ROOT/STATUS"
eval_two_gpu "$CAPACITY/model" 4 72 4 "$CAPACITY/eval480"

for epoch in 10 100 500; do
  echo "EVALUATING_DYNAMICS_EPOCH_${epoch}" > "$ROOT/STATUS"
  checkpoint="$DYNAMICS/model/checkpoints/epoch_$(printf '%04d' "$epoch")"
  output="$DYNAMICS/eval480/epoch_$(printf '%04d' "$epoch")"
  eval_two_gpu "$checkpoint" 3 48 3 "$output"
done

echo COMPLETE > "$ROOT/STATUS"
