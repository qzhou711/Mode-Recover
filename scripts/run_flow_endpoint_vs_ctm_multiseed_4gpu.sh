#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
source /jet/home/qzhou7/workspace/anaconda3/etc/profile.d/conda.sh
conda activate d3il
export PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=logs/avoiding/flow_endpoint_vs_ctm_multiseed
T=logs/avoiding/trained/flow_matching_transformer_5000_seed42
F=logs/avoiding/trained/flow_matching_small_struct_3x48_5000_seed42
mkdir -p "$ROOT"
train_endpoint() {
  local gpu=$1 seed=$2 name=$3
  local out="$ROOT/$name/model"
  mkdir -p "$out/checkpoints/epoch_0000"
  cp "$F/eval_best_flow.pth" "$out/checkpoints/epoch_0000/eval_best_flow.pth"
  CUDA_VISIBLE_DEVICES="$gpu" python -u distill_flow_matching_avoiding.py     --teacher-dir "$T" --output-dir "$out" --teacher-steps 16 --student-steps 1     --student-layers 3 --student-embed-dim 48 --student-heads 3     --student-init checkpoint --student-init-dir "$F" --epochs 500 --batch-size 256     --max-batches-per-epoch 4 --flow-weight 0.1 --seed "$seed"     --save-epochs 10 25 50 100 250 500 > "$ROOT/$name/train.log" 2>&1
}
train_ctm() {
  local gpu=$1 seed=$2 name=$3
  local out="$ROOT/$name/model"
  mkdir -p "$out/checkpoints/epoch_0000"
  cp "$F/eval_best_flow.pth" "$out/checkpoints/epoch_0000/eval_best_flow.pth"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train_flow_ctm_avoiding.py     --teacher-dir "$T" --output-dir "$out" --student-layers 3 --student-embed-dim 48     --student-heads 3 --student-init checkpoint --init-dir "$F" --epochs 500     --batch-size 256 --max-batches-per-epoch 4 --dsm-weight 0.1 --seed "$seed"     --save-epochs 10 25 50 100 250 500 > "$ROOT/$name/train.log" 2>&1
}
train_endpoint 0 43 endpoint_seed43 & p0=$!
train_ctm 1 43 ctm_seed43 & p1=$!
train_endpoint 2 44 endpoint_seed44 & p2=$!
train_ctm 3 44 ctm_seed44 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
merge_standard() {
  local out=$1; shift
  python scripts/merge_avoiding_eval_shards.py --shards "$@" --output-dir "$out" > "$out/merge.log" 2>&1
}
eval_standard() {
  local gpu=$1 name=$2 model=$3
  local out="$ROOT/$name/standard480"
  mkdir -p "$out/shards"
  local pids=() shards=()
  for w in 0 1 2 3; do
    local start=$((w*120)) shard="$out/shards/episodes_$((w*120))_$((w*120+119))"
    shards+=("$shard")
    CUDA_VISIBLE_DEVICES="$gpu" python -u visualize_avoiding.py --models flow       --flow-weights-dir "$model" --flow-steps 1 --flow-layers 3 --flow-embed-dim 48       --flow-heads 3 --episode-start "$start" --n-trajectories 120 --progress-every 10       --output-dir "$shard" > "$shard.log" 2>&1 & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  merge_standard "$out" "${shards[@]}"
}
eval_paired() {
  local gpu=$1 name=$2 model=$3
  local out="$ROOT/$name/paired1000"
  mkdir -p "$out/shards"
  local pids=() shards=()
  for w in 0 1 2 3; do
    local start=$((w*250)) shard="$out/shards/episodes_$((w*250))_$((w*250+249))"
    shards+=("$shard")
    CUDA_VISIBLE_DEVICES="$gpu" python -u analyze_flow_paired_rollouts.py       --teacher-dir "$T" --teacher-layers 4 --teacher-embed-dim 72 --teacher-heads 4       --student-dir "$model" --student-layers 3 --student-embed-dim 48 --student-heads 3       --episode-start "$start" --n-episodes 250 --seed 2027 --progress-every 10       --output-dir "$shard" > "$shard.log" 2>&1 & pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python merge_flow_paired_rollouts.py --shards "${shards[@]}" --expected-episodes 1000     --output-dir "$out/merged" > "$out/merge.log" 2>&1
}
eval_lane() {
  local gpu=$1 name=$2
  local model="$ROOT/$name/model"
  eval_standard "$gpu" "$name" "$model"
  eval_paired "$gpu" "$name" "$model"
}
eval_lane 0 endpoint_seed43 & e0=$!
eval_lane 1 ctm_seed43 & e1=$!
eval_lane 2 endpoint_seed44 & e2=$!
eval_lane 3 ctm_seed44 & e3=$!
wait "$e0" "$e1" "$e2" "$e3"
# Reuse completed seed42 training and evaluate it with the identical long protocols.
eval_standard 0 endpoint_seed42 logs/avoiding/flow_endpoint_bctm_anchor/teacher_endpoint/model & q0=$!
eval_standard 1 ctm_seed42 logs/avoiding/flow_endpoint_bctm_anchor/boundary_ctm/model & q1=$!
wait "$q0" "$q1"
eval_paired 0 endpoint_seed42 logs/avoiding/flow_endpoint_bctm_anchor/teacher_endpoint/model & r0=$!
eval_paired 1 ctm_seed42 logs/avoiding/flow_endpoint_bctm_anchor/boundary_ctm/model & r1=$!
wait "$r0" "$r1"
touch "$ROOT/COMPLETE"
