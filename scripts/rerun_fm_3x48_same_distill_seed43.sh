#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
TEACHER=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
BLOCKING_STATUS=logs/avoiding/flow_solver_baselines/fm_4x72_step1_eval480/STATUS
ROOT=logs/avoiding/flow_capacity_scan/same_3x48_rerun_seed43
MODEL="$ROOT/model"
EVAL="$ROOT/eval480"
SHARDS="$ROOT/eval_shards"

mkdir -p "$MODEL" "$SHARDS"
echo WAITING_FOR_GPU > "$ROOT/STATUS"

while [[ ! -f "$BLOCKING_STATUS" ]] || [[ "$(cat "$BLOCKING_STATUS")" == "RUNNING" ]]; do
  sleep 30
done
if [[ "$(cat "$BLOCKING_STATUS")" != "COMPLETE" ]]; then
  echo BLOCKING_EVAL_FAILED > "$ROOT/STATUS"
  exit 1
fi

echo TRAINING > "$ROOT/STATUS"
CUDA_VISIBLE_DEVICES=0 "$PY" -u distill_flow_matching_avoiding.py \
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
  --seed 43 \
  --output-dir "$MODEL" \
  > "$ROOT/train.log" 2>&1
test -s "$MODEL/eval_best_flow.pth"

echo EVALUATING > "$ROOT/STATUS"
CUDA_VISIBLE_DEVICES=0 "$PY" -u visualize_avoiding.py \
  --models flow \
  --flow-weights-dir "$MODEL" \
  --flow-steps 1 \
  --flow-layers 3 \
  --flow-embed-dim 48 \
  --flow-heads 3 \
  --episode-start 0 \
  --n-trajectories 240 \
  --progress-every 10 \
  --output-dir "$SHARDS/episodes_000_239" \
  > "$ROOT/eval_shard0.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "$PY" -u visualize_avoiding.py \
  --models flow \
  --flow-weights-dir "$MODEL" \
  --flow-steps 1 \
  --flow-layers 3 \
  --flow-embed-dim 48 \
  --flow-heads 3 \
  --episode-start 240 \
  --n-trajectories 240 \
  --progress-every 10 \
  --output-dir "$SHARDS/episodes_240_479" \
  > "$ROOT/eval_shard1.log" 2>&1 &
pid1=$!

failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo EVAL_FAILED > "$ROOT/STATUS"
  exit 1
fi

"$PY" scripts/merge_avoiding_eval_shards.py \
  --shards "$SHARDS/episodes_000_239" "$SHARDS/episodes_240_479" \
  --output-dir "$EVAL" \
  > "$ROOT/merge.log" 2>&1

echo COMPLETE > "$ROOT/STATUS"
