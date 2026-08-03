#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/keep013_width_compression
BUFFER="$ROOT/teacher_buffer_2400"
TEACHER=logs/avoiding/tinysr_depth_recoverability/selected_model
BUNDLE=logs/avoiding/teacher_deployment_bundle
mkdir -p "$BUFFER/shards"

# Eight complete shards replace the interrupted, non-checkpointed partial shards.
pids=()
for gpu in 0 1; do
  for worker in 0 1 2 3; do
    idx=$((gpu*4+worker)); start=$((13000+idx*300)); out="$BUFFER/shards/resume_${start}_$((start+299))"; mkdir -p "$out"
    if [[ ! -s "$out/transfer_buffer.pt" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u generate_teacher_transfer_buffer.py --bundle-dir "$BUNDLE" \
        --model-dir "$TEACHER" --layers 3 --embed-dim 72 --heads 4 --steps 16 \
        --output-dir "$out" --episode-start "$start" --n-episodes 300 --seed 2027 --progress-every 15 \
        > "$out/run.log" 2>&1 & pids+=("$!")
    fi
  done
done
for p in "${pids[@]}"; do wait "$p"; done
"$PY" merge_teacher_transfer_buffer.py --root "$BUFFER" > "$BUFFER/merge.log" 2>&1
"$PY" -c "import json;d=json.load(open('$BUFFER/metrics.json'));assert d['episodes']==2400 and d['success_rate']>=.85 and d['mode_coverage']>=22 and not d['uses_original_demonstrations'] and not d['uses_expert_actions'],d"

lane() {
  local gpu=$1 method=$2 out="$ROOT/width_$2"; mkdir -p "$out/model"
  if [[ ! -s "$out/model/pretrain_epoch_0500.pth" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_recoverable_width_compression.py \
      --bundle-dir "$BUNDLE" --teacher "$TEACHER" --buffer "$BUFFER/transfer_buffer.pt" \
      --output-dir "$out/model" --method "$method" --epochs 500 --batch-size 256 --max-batches 4 \
      --learning-rate 3e-5 --endpoint-weight .03 --save-epochs 50,100,250,500 --seed 42 > "$out/train.log" 2>&1
  fi
  for e in 50 100 250 500; do
    e4=$(printf '%04d' "$e"); tmp="$out/epoch${e}/model"; mkdir -p "$tmp"
    cp "$out/model/pretrain_epoch_${e4}.pth" "$tmp/eval_best_flow.pth"
    if [[ ! -s "$out/epoch${e}/eval120/metrics.json" ]]; then
      scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$tmp" "$out/epoch${e}/eval120" \
        120 0 42 4 16 3 48 3 > "$out/eval_epoch${e}.log" 2>&1
    fi
  done
  echo "COMPLETE $(date --iso-8601=seconds)" > "$out/STATUS"
}

lane 0 random_orthogonal & P0=$!; lane 1 pca & P1=$!; wait "$P0"; wait "$P1"
lane 0 activation_coordinate & P2=$!; lane 1 qkv_sensitivity & P3=$!; wait "$P2"; wait "$P3"
"$PY" scripts/select_width_recoverability.py --root "$ROOT" > "$ROOT/selection.log"
read -r passed lane_name epoch < <("$PY" -c "import json;d=json.load(open('$ROOT/selection.json'));s=d['selected'];print(int(d['any_passed']),s['lane'],s['epoch'])")
if [[ "$passed" == 1 && ! -s "$ROOT/selected_eval480/metrics.json" ]]; then
  e4=$(printf '%04d' "$epoch"); mkdir -p "$ROOT/selected_model"
  cp "$ROOT/$lane_name/model/pretrain_epoch_${e4}.pth" "$ROOT/selected_model/eval_best_flow.pth"
  scripts/run_deployed_flow_parallel_eval.sh 0 "$ROOT/selected_model" "$ROOT/selected_eval480" \
    480 1000 4242 4 16 3 48 3 > "$ROOT/selected_eval480.log" 2>&1
fi
echo "COMPLETE passed=$passed selected=$lane_name epoch=$epoch $(date --iso-8601=seconds)" > "$ROOT/STATUS"
