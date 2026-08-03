#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/keep013_step_distillation
TEACHER=logs/avoiding/tinysr_depth_recoverability/selected_model
BUFFER=logs/avoiding/keep013_width_compression/teacher_buffer_2400/transfer_buffer.pt
BUNDLE=logs/avoiding/teacher_deployment_bundle
mkdir -p "$ROOT"; echo "TRAINING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
cat > "$ROOT/audit.txt" <<'EOF'
Long-term objective: demo-free transferable structure+step WAM distillation with mode preservation.
Short-term bottleneck: test whether the successful keep013 depth-compressed model survives 16->1 step distillation.
Stage targeted: step distillation only; 3x72 architecture fixed.
Mechanism hypothesis: endpoint fidelity preserves teacher basins better than self-bootstrapped CTM, while both reduce latency.
Causal intervention and controls: same keep013 teacher/init/buffer/seed; endpoint vs Boundary-CTM+DSM; 16-step and solver-only controls.
Data-access status: teacher rollout states plus online keep013 queries; no original demonstrations or expert actions.
Metrics and go/no-go: SR, coverage, entropy, trajectories, CUDA latency; Standard120 screen then Standard480.
Resource/time check: two V100, about 3h40 remaining; tmux, resumable checkpoints.
Expected interpretations: good 16-step but bad 1-step isolates step collapse; endpoint>CTM implicates CTM self-bootstrapping.
EOF
train_lane(){ gpu=$1; method=$2; out="$ROOT/$method/model"; mkdir -p "$out"; CUDA_VISIBLE_DEVICES=$gpu "$PY" -u train_deployed_flow_step_distillation.py --teacher-dir "$TEACHER" --buffer "$BUFFER" --output-dir "$out" --method "$method" --epochs 500 --save-epochs 100,250,500 > "$ROOT/$method/train.log" 2>&1; }
mkdir -p "$ROOT/endpoint" "$ROOT/boundary_ctm"; train_lane 0 endpoint & p0=$!; train_lane 1 boundary_ctm & p1=$!; wait $p0; wait $p1
echo "EVALUATING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
eval_lane(){ gpu=$1; method=$2; for tag in 100 250 500; do src="$ROOT/$method/model/checkpoints/epoch_$(printf '%04d' $tag)"; out="$ROOT/$method/epoch_$tag/eval120"; scripts/run_deployed_flow_parallel_eval.sh $gpu "$src" "$out" 120 0 42 4 1 3 72 4 "" --inference-mode boundary > "$ROOT/$method/eval_$tag.log" 2>&1; done; }
eval_lane 0 endpoint & e0=$!; eval_lane 1 boundary_ctm & e1=$!; wait $e0; wait $e1
CUDA_VISIBLE_DEVICES=0 "$PY" benchmark_flow_latency.py --model-dir "$TEACHER" --steps 16 --inference-mode integrate --output "$ROOT/latency_keep013_16step.json"
CUDA_VISIBLE_DEVICES=0 "$PY" benchmark_flow_latency.py --model-dir "$ROOT/endpoint/model/checkpoints/epoch_0500" --steps 1 --inference-mode boundary --output "$ROOT/latency_endpoint_1step.json"
CUDA_VISIBLE_DEVICES=1 "$PY" benchmark_flow_latency.py --model-dir "$ROOT/boundary_ctm/model/checkpoints/epoch_0500" --steps 1 --inference-mode boundary --output "$ROOT/latency_ctm_1step.json"
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
