#!/usr/bin/env bash
set -euo pipefail
source /jet/home/qzhou7/workspace/anaconda3/etc/profile.d/conda.sh
conda activate d3il
export PYTHONPATH=.
ROOT=logs/avoiding/flow_ctm_validation
T4=logs/avoiding/trained/flow_matching_transformer_5000_seed42
T3=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
mkdir -p "$ROOT"
run_train() {
  local gpu=$1 name=$2 teacher=$3 layers=$4 dim=$5 heads=$6 dsm=$7
  mkdir -p "$ROOT/$name/model"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train_flow_ctm_avoiding.py     --teacher-dir "$teacher" --output-dir "$ROOT/$name/model"     --student-layers "$layers" --student-embed-dim "$dim" --student-heads "$heads"     --teacher-layers "$layers" --teacher-embed-dim "$dim" --teacher-heads "$heads" \
    --epochs 500 --batch-size 256 --max-batches-per-epoch 4 --time-bins 16     --dsm-weight "$dsm" > "$ROOT/$name/train.log" 2>&1
}
run_train 0 fm_4x72_ctm_dsm01 "$T4" 4 72 4 0.1 & p0=$!
run_train 1 fm_4x72_ctm_dsm0  "$T4" 4 72 4 0.0 & p1=$!
run_train 2 fm_3x48_ctm_dsm01 "$T3" 3 48 3 0.1 & p2=$!
run_train 3 fm_3x48_ctm_dsm0  "$T3" 3 48 3 0.0 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
run_eval() {
  local gpu=$1 name=$2 layers=$3 dim=$4 heads=$5
  CUDA_VISIBLE_DEVICES="$gpu" python -u visualize_avoiding.py --models flow     --flow-weights-dir "$ROOT/$name/model" --flow-steps 1     --flow-layers "$layers" --flow-embed-dim "$dim" --flow-heads "$heads"     --n-trajectories 120 --progress-every 10 --output-dir "$ROOT/$name/eval120"     > "$ROOT/$name/eval120.log" 2>&1
}
run_eval 0 fm_4x72_ctm_dsm01 4 72 4 & e0=$!
run_eval 1 fm_4x72_ctm_dsm0  4 72 4 & e1=$!
run_eval 2 fm_3x48_ctm_dsm01 3 48 3 & e2=$!
run_eval 3 fm_3x48_ctm_dsm0  3 48 3 & e3=$!
wait "$e0" "$e1" "$e2" "$e3"
touch "$ROOT/COMPLETE"
