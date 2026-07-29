#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
source /jet/home/qzhou7/workspace/anaconda3/etc/profile.d/conda.sh
conda activate d3il
export PYTHONPATH=.
ROOT=logs/avoiding/flow_endpoint_bctm_anchor
T=logs/avoiding/trained/flow_matching_transformer_5000_seed42
F=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
mkdir -p "$ROOT"
train_endpoint() {
  mkdir -p "$ROOT/teacher_endpoint/model"
  CUDA_VISIBLE_DEVICES=0 python -u distill_flow_matching_avoiding.py     --teacher-dir "$T" --output-dir "$ROOT/teacher_endpoint/model"     --teacher-steps 16 --student-steps 1 --student-layers 3 --student-embed-dim 48 --student-heads 3     --student-init checkpoint --student-init-dir "$F" --epochs 500 --batch-size 256     --max-batches-per-epoch 4 --flow-weight 0.1 > "$ROOT/teacher_endpoint/train.log" 2>&1
}
train_bctm() {
  local gpu=$1 name=$2 anchor=$3
  mkdir -p "$ROOT/$name/model"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train_flow_ctm_avoiding.py     --teacher-dir "$T" --output-dir "$ROOT/$name/model"     --student-layers 3 --student-embed-dim 48 --student-heads 3     --student-init checkpoint --init-dir "$F" --epochs 500 --batch-size 256     --max-batches-per-epoch 4 --dsm-weight 0.1 --endpoint-anchor-weight "$anchor"     > "$ROOT/$name/train.log" 2>&1
}
train_endpoint & p0=$!
train_bctm 1 boundary_ctm 0 & p1=$!
train_bctm 2 boundary_ctm_anchor005 0.05 & p2=$!
train_bctm 3 boundary_ctm_anchor01 0.1 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
eval_one() {
  local gpu=$1 name=$2
  CUDA_VISIBLE_DEVICES="$gpu" python -u visualize_avoiding.py --models flow     --flow-weights-dir "$ROOT/$name/model" --flow-steps 1     --flow-layers 3 --flow-embed-dim 48 --flow-heads 3 --n-trajectories 120     --progress-every 10 --output-dir "$ROOT/$name/eval120" > "$ROOT/$name/eval120.log" 2>&1
}
eval_one 0 teacher_endpoint & e0=$!
eval_one 1 boundary_ctm & e1=$!
eval_one 2 boundary_ctm_anchor005 & e2=$!
eval_one 3 boundary_ctm_anchor01 & e3=$!
wait "$e0" "$e1" "$e2" "$e3"
touch "$ROOT/COMPLETE"
