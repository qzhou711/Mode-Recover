#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
ROOT=logs/avoiding/tinysr_depth_recoverability/keep013/epoch0
MODEL="$ROOT/model"
OUT="$ROOT/eval120"
rm -rf "$OUT"
mkdir -p "$OUT"

cat > "$ROOT/audit.txt" <<'EOF'
Long-term objective: demo-free, mode-preserving architecture compression followed by step distillation.
Short-term bottleneck: quantify the exact keep013 post-deletion, pre-repair closed-loop baseline.
Stage targeted: architecture compression initialization diagnostic (repair epoch 0).
Mechanism hypothesis: exact subset copying retains part of Teacher behavior, while rollout repair restores closed-loop executability and mode support.
Causal intervention and controls: same initial_flow checkpoint, 16-step solver, seed 42, episodes 0-119, and trusted evaluator as repaired checkpoints.
Data-access status: evaluation only; no demonstrations, expert actions, or training updates.
Metrics and go/no-go: Standard-120 SR, successful-mode coverage, auxiliary entropy, raw trajectories and plot.
Resource/time check: two V100; two 60-episode halves with four workers per GPU; expected under 10 minutes.
Expected interpretations: low epoch-0 SR isolates transfer-map damage; strong epoch-0 metrics would reduce the claimed contribution of repair.
EOF

scripts/run_deployed_flow_parallel_eval.sh 0 "$MODEL" "$OUT/gpu0_ep000_059" 60 0 42 4 16 3 72 4 & p0=$!
scripts/run_deployed_flow_parallel_eval.sh 1 "$MODEL" "$OUT/gpu1_ep060_119" 60 60 42 4 16 3 72 4 & p1=$!
wait "$p0"; wait "$p1"

PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
mapfile -t SHARDS < <(find "$OUT" -path '*/shards/worker_*' -type d | sort)
"$PY" scripts/merge_deployed_flow_eval_shards.py --shards "${SHARDS[@]}" --output-dir "$OUT"
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
