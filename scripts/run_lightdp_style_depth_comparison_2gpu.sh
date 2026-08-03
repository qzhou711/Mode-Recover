#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/lightdp_style_depth_comparison
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt
mkdir -p "$ROOT"

cat > "$ROOT/audit.json" <<'EOF'
{
  "long_term_objective": "transferable demo-free architecture compression plus step distillation for robot/WAM policies while preserving multimodality",
  "short_term_bottleneck": "determine whether static SVD importance and learned gates select mode-preserving layers as reliably as closed-loop recoverability",
  "stage_targeted": "architecture compression / cross-architecture distillation",
  "mechanism_hypothesis": "LightDP-style learned layer gates optimize average fidelity but may miss noise-to-mode basin geometry",
  "causal_control": "same FM-4x72-16 teacher, successful rollout buffer, 3x72 target, endpoint/velocity repair, seed and Standard-120 evaluator as keep013",
  "data_access": "teacher rollouts only; no original demonstrations or expert actions",
  "screen": "Standard-120 SR, coverage and entropy; confirm Standard-480 only if competitive",
  "go": "SR>=88%, coverage>=22; compare selected mask and Pareto point with keep013",
  "official_reproduction": false,
  "reason": "LightDP has no accessible official code and its published setup is demonstration-based DDPM, so this is a controlled LightDP-style Flow adaptation"
}
EOF

CUDA_VISIBLE_DEVICES=0 "$PY" -u train_lightdp_style_depth_pruning.py \
  --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$ROOT/model" \
  --gate-epochs 250 --repair-epochs 500 --seed 42 > "$ROOT/train.log" 2>&1

for epoch in 50 100 250 500; do
  e4=$(printf '%04d' "$epoch")
  mkdir -p "$ROOT/epoch${epoch}/model"
  cp "$ROOT/model/pretrain_epoch_${e4}.pth" "$ROOT/epoch${epoch}/model/eval_best_flow.pth"
  scripts/run_deployed_flow_parallel_eval.sh 0 "$ROOT/epoch${epoch}/model" \
    "$ROOT/epoch${epoch}/eval120" 120 0 42 4 16 3 72 4 > "$ROOT/eval_${epoch}.log" 2>&1
done
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
