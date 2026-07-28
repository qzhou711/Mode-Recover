#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
source /jet/home/qzhou7/workspace/anaconda3/etc/profile.d/conda.sh
conda activate d3il
export PYTHONPATH=.
ROOT=logs/avoiding/flow_boundary_ctm_validation
TEACHER=logs/avoiding/trained/flow_matching_transformer_5000_seed42
mkdir -p "$ROOT"
rm -f "$ROOT/COMPLETE"
run_train() {
  local gpu=$1 name=$2 extra=$3
  mkdir -p "$ROOT/$name/model"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train_flow_ctm_avoiding.py     --teacher-dir "$TEACHER" --output-dir "$ROOT/$name/model"     --epochs 500 --batch-size 256 --max-batches-per-epoch 4     --time-bins 16 --dsm-weight 0.1 $extra > "$ROOT/$name/train.log" 2>&1
}
run_train 0 uniform "--endpoint-probability 0" & p0=$!
run_train 1 endpoint50 "--endpoint-probability 0.5" & p1=$!
run_train 2 endpoint_anchor "--endpoint-probability 0.5 --endpoint-anchor-weight 0.1" & p2=$!
run_train 3 endpoint_anchor_distribution "--endpoint-probability 0.5 --endpoint-anchor-weight 0.1 --distribution-weight 0.1 --conditional-samples 8" & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
run_eval() {
  local gpu=$1 name=$2
  CUDA_VISIBLE_DEVICES="$gpu" python -u visualize_avoiding.py --models flow     --flow-weights-dir "$ROOT/$name/model" --flow-steps 1     --flow-layers 4 --flow-embed-dim 72 --flow-heads 4     --n-trajectories 120 --progress-every 10 --output-dir "$ROOT/$name/eval120"     > "$ROOT/$name/eval120.log" 2>&1
}
run_eval 0 uniform & e0=$!
run_eval 1 endpoint50 & e1=$!
run_eval 2 endpoint_anchor & e2=$!
run_eval 3 endpoint_anchor_distribution & e3=$!
wait "$e0" "$e1" "$e2" "$e3"
touch "$ROOT/COMPLETE"
