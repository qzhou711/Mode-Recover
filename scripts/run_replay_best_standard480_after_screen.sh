#!/usr/bin/env bash
set -euo pipefail

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/bmd_velocity250_expanded/closed_loop_replay_experiments
LANES=(natural_replay balanced_replay balanced_anchor balanced_anchor_pcgrad)

while true; do
  READY=1
  for LANE in "${LANES[@]}"; do
    if ! grep -q '^COMPLETE' "$ROOT/$LANE/STATUS" 2>/dev/null; then READY=0; fi
  done
  [[ "$READY" == "1" ]] && break
  sleep 20
done

for GPU in 0 1 2 3; do
  LANE="${LANES[$GPU]}"
  EPOCH=$("$PY" select_replay_checkpoint.py --lane "$ROOT/$LANE")
  E=$(printf '%04d' "$EPOCH")
  echo "EVALUATING_STANDARD480 selected_epoch=$EPOCH $(date --iso-8601=seconds)" \
    > "$ROOT/$LANE/STATUS"
  bash scripts/run_deployed_flow_parallel_eval.sh "$GPU" \
    "$ROOT/$LANE/model/checkpoints/epoch_$E" \
    "$ROOT/$LANE/epoch_$E/eval480" 480 0 42 4 1 3 48 3 \
    > "$ROOT/$LANE/epoch_${E}_eval480.log" 2>&1 &
done
wait
for LANE in "${LANES[@]}"; do
  echo "COMPLETE_STANDARD480 $(date --iso-8601=seconds)" > "$ROOT/$LANE/STATUS"
done
