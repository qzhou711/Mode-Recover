#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/keep013_width_compression
BUFFER="$ROOT/teacher_buffer_2400"
TEACHER=logs/avoiding/tinysr_depth_recoverability/selected_model
BUNDLE=logs/avoiding/teacher_deployment_bundle
mkdir -p "$ROOT" "$BUFFER/shards"
"$PY" - <<'PY' > "$ROOT/audit.json"
import json
print(json.dumps({'scope':'FM-3x72-16 keep013 -> FM-3x48-16 width-only structure distillation','teacher_steps':16,'student_steps':16,'depth_fixed':3,'teacher_rollout_episodes':2400,'methods':['random_orthogonal','pca','activation_coordinate','qkv_sensitivity'],'repair':'identical velocity + 0.03 endpoint','uses_original_demonstrations':False,'uses_expert_actions':False,'gate':'Standard-120 SR>=80%, coverage>=22; passing best gets independent Standard-480'},indent=2))
PY

pids=()
for gpu in 0 1 2 3; do
  for worker in 0 1 2 3; do
    idx=$((gpu*4+worker)); start=$((10000+idx*150)); out="$BUFFER/shards/episodes_${start}_$((start+149))"; mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u generate_teacher_transfer_buffer.py --bundle-dir "$BUNDLE" \
      --model-dir "$TEACHER" --layers 3 --embed-dim 72 --heads 4 --steps 16 \
      --output-dir "$out" --episode-start "$start" --n-episodes 150 --seed 2027 --progress-every 15 \
      > "$out/run.log" 2>&1 & pids+=("$!")
  done
done
for p in "${pids[@]}"; do wait "$p"; done
"$PY" merge_teacher_transfer_buffer.py --root "$BUFFER" > "$BUFFER/merge.log" 2>&1
"$PY" -c "import json;d=json.load(open('$BUFFER/metrics.json'));assert d['episodes']==2400 and d['success_rate']>=.85 and d['mode_coverage']>=22 and not d['uses_original_demonstrations'] and not d['uses_expert_actions'],d"

lane() {
  local gpu=$1 method=$2 out="$ROOT/width_$2"; mkdir -p "$out/model"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u train_recoverable_width_compression.py \
    --bundle-dir "$BUNDLE" --teacher "$TEACHER" --buffer "$BUFFER/transfer_buffer.pt" \
    --output-dir "$out/model" --method "$method" --epochs 500 --batch-size 256 --max-batches 4 \
    --learning-rate 3e-5 --endpoint-weight .03 --save-epochs 50,100,250,500 --seed 42 > "$out/train.log" 2>&1
  for e in 50 100 250 500; do
    e4=$(printf '%04d' "$e"); tmp="$out/epoch${e}/model"; mkdir -p "$tmp"
    cp "$out/model/pretrain_epoch_${e4}.pth" "$tmp/eval_best_flow.pth"
    scripts/run_deployed_flow_parallel_eval.sh "$gpu" "$tmp" "$out/epoch${e}/eval120" \
      120 0 42 4 16 3 48 3 > "$out/eval_epoch${e}.log" 2>&1
  done
  echo "COMPLETE $(date --iso-8601=seconds)" > "$out/STATUS"
}

lane 0 random_orthogonal & P0=$!
lane 1 pca & P1=$!
lane 2 activation_coordinate & P2=$!
lane 3 qkv_sensitivity & P3=$!
failed=0
for p in "$P0" "$P1" "$P2" "$P3"; do wait "$p" || failed=1; done
((failed==0)) || { echo FAILED > "$ROOT/STATUS"; exit 1; }
"$PY" scripts/select_width_recoverability.py --root "$ROOT" > "$ROOT/selection.log"
read -r passed lane_name epoch < <("$PY" -c "import json;d=json.load(open('$ROOT/selection.json'));s=d['selected'];print(int(d['any_passed']),s['lane'],s['epoch'])")
if [[ "$passed" == 1 ]]; then
  e4=$(printf '%04d' "$epoch"); mkdir -p "$ROOT/selected_model"
  cp "$ROOT/$lane_name/model/pretrain_epoch_${e4}.pth" "$ROOT/selected_model/eval_best_flow.pth"
  scripts/run_deployed_flow_parallel_eval.sh 0 "$ROOT/selected_model" "$ROOT/selected_eval480" \
    480 1000 4242 4 16 3 48 3 > "$ROOT/selected_eval480.log" 2>&1
fi
echo "COMPLETE passed=$passed selected=$lane_name epoch=$epoch $(date --iso-8601=seconds)" > "$ROOT/STATUS"
