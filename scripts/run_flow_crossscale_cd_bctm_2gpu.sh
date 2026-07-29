#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
source /jet/home/qzhou7/workspace/anaconda3/etc/profile.d/conda.sh
conda activate d3il
export PYTHONPATH=.
ROOT=logs/avoiding/flow_crossscale_cd_bctm
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
FULL3=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
mkdir -p "$ROOT"
train_cd() {
  local gpu=$1 name=$2 init=$3 extra=$4
  mkdir -p "$ROOT/$name/model"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train_flow_consistency_distillation.py     --teacher-dir "$TEACHER" --output-dir "$ROOT/$name/model"     --student-init "$init" --student-layers 3 --student-embed-dim 48 --student-heads 3     --epochs 500 --batch-size 256 --max-batches-per-epoch 4 --flow-weight 0.1 $extra     > "$ROOT/$name/train.log" 2>&1
  touch "$ROOT/$name/TRAIN_COMPLETE"
}
train_bctm() {
  local gpu=$1 name=$2 init=$3 extra=$4
  mkdir -p "$ROOT/$name/model"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train_flow_ctm_avoiding.py     --teacher-dir "$TEACHER" --output-dir "$ROOT/$name/model"     --student-layers 3 --student-embed-dim 48 --student-heads 3 --student-init "$init"     --epochs 500 --batch-size 256 --max-batches-per-epoch 4 --dsm-weight 0.1 $extra     > "$ROOT/$name/train.log" 2>&1
  touch "$ROOT/$name/TRAIN_COMPLETE"
}
lane0() {
  train_cd 0 cd_random random ""
  train_bctm 0 bctm_random random ""
}
lane1() {
  train_cd 1 cd_full full "--full-dir $FULL3"
  train_bctm 1 bctm_full checkpoint "--init-dir $FULL3"
}
lane0 & p0=$!
lane1 & p1=$!
wait "$p0" "$p1"
eval_one() {
  local gpu=$1 name=$2
  CUDA_VISIBLE_DEVICES="$gpu" python -u visualize_avoiding.py --models flow     --flow-weights-dir "$ROOT/$name/model" --flow-steps 1     --flow-layers 3 --flow-embed-dim 48 --flow-heads 3     --n-trajectories 120 --progress-every 10 --output-dir "$ROOT/$name/eval120"     > "$ROOT/$name/eval120.log" 2>&1
}
eval_lane0() { eval_one 0 cd_random; eval_one 0 bctm_random; }
eval_lane1() { eval_one 1 cd_full; eval_one 1 bctm_full; }
eval_lane0 & e0=$!
eval_lane1 & e1=$!
wait "$e0" "$e1"
touch "$ROOT/COMPLETE"
