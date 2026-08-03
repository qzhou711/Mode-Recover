#!/usr/bin/env bash
set -euo pipefail
GPU=$1;TAG=$2;H=$3;RECOVERY=$4
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python;ROOT=logs/avoiding/p6_intervention_dagger_repair;LANE="$ROOT/$TAG";mkdir -p "$LANE/model";EXTRA=()
[[ "$RECOVERY" == 1 ]]&&EXTRA+=(--induced-recovery-only)
if [[ ! -s "$LANE/model/pretrain_epoch_0250.pth" ]];then echo "TRAINING $(date --iso-8601=seconds)" > "$LANE/STATUS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u train_student_induced_structure_repair.py --bundle-dir logs/avoiding/teacher_deployment_bundle --base-buffer logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt --induced-buffer "logs/avoiding/p6_intervention_dagger_round2/h${H}/intervention_buffer.pt" --initial-student logs/avoiding/p6_strong_teacher_structure_repair/velocity_endpoint/selected_model --output-dir "$LANE/model" --induced-ratio 0.25 --endpoint-weight 0.03 --epochs 250 --save-epochs 50,100,250 --batch-size 256 --max-batches 4 --learning-rate 3e-5 --seed 42 "${EXTRA[@]}" > "$LANE/train.log" 2>&1;fi
for E in 50 100 250;do OUT="$LANE/eval120_epoch${E}";if [[ ! -s "$OUT/metrics.json" ]];then E4=$(printf '%04d' "$E");TMP="$LANE/eval_model_epoch${E}";mkdir -p "$TMP";cp "$LANE/model/pretrain_epoch_${E4}.pth" "$TMP/eval_best_flow.pth";echo "EVALUATING_120 epoch=$E $(date --iso-8601=seconds)" > "$LANE/STATUS";scripts/run_deployed_flow_parallel_eval.sh "$GPU" "$TMP" "$OUT" 120 0 42 4 16 3 48 3 > "$LANE/eval120_epoch${E}.log" 2>&1;fi;done
"$PY" scripts/select_p6_structure_checkpoint.py --lane "$LANE" --epochs 50 100 250 > "$LANE/selection.log" 2>&1
PASS=$("$PY" -c "import json;print(int(json.load(open('$LANE/selection.json'))['selected']['passed_gate']))")
if [[ "$PASS" == 1 && ! -s "$LANE/eval480/metrics.json" ]];then echo "EVALUATING_480 $(date --iso-8601=seconds)" > "$LANE/STATUS";scripts/run_deployed_flow_parallel_eval.sh "$GPU" "$LANE/selected_model" "$LANE/eval480" 480 0 42 4 16 3 48 3 > "$LANE/eval480.log" 2>&1;fi
echo "COMPLETE passed_gate=$PASS $(date --iso-8601=seconds)" > "$LANE/STATUS"
