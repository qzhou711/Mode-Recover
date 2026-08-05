#!/usr/bin/env bash
set -euo pipefail

cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/biretopk_v1
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt

# P1 is launched separately on four GPUs. Do not start P2 until all replicas pass.
while [ "$(find "$ROOT" -maxdepth 2 -path '*/k3_seed_*/summary.json' | wc -l)" -lt 4 ]; do
  sleep 30
done

$PY - <<'PY'
import json
from pathlib import Path
root=Path('logs/avoiding/biretopk_v1')
rows=[]
for seed in (42,43,44,45):
    x=json.loads((root/f'k3_seed_{seed}/summary.json').read_text())
    rows.append({'seed':seed,'selected_layers':x['selected_layers'],
                 'inclusion_probabilities':x['final_inclusion_probabilities']})
passed=sum(r['selected_layers']==[0,1,3] for r in rows)
result={'phase':'P1 L4K3 generic BiReTopK','rows':rows,'passed_seeds':passed,'go':passed>=3}
(root/'p1_k3_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if passed < 3: raise SystemExit(2)
PY

# P2a: all six Top-2 masks are calibration truth only, run in two GPU waves.
run_mask () {
  local gpu=$1 mask=$2
  local name=${mask//,/}
  local out=$ROOT/k2_calibration/keep${name}
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu $PY train_recoverable_depth_pruning.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$out" \
    --layer-map "$mask" --epochs 5 --batch-size 256 --max-batches 4 \
    --save-epochs 5 --all-rollouts --holdout-residue 2 --seed 42 \
    > "$out/train.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY audit_depth_recoverability_proxies.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --student-layers 2 \
    --student-checkpoint "$out/pretrain_epoch_0005.pth" \
    --output "$out/proxy.json" --mask "keep${name}" --stage epoch_0005 \
    --samples 512 --groups 48 --samples-per-state 4 --batch-size 128 \
    --seed 42 --episode-residue 2 > "$out/proxy.log" 2>&1
}

run_mask 0 0,1 & run_mask 1 0,2 & run_mask 2 0,3 & run_mask 3 1,2 & wait
run_mask 0 1,3 & run_mask 1 2,3 & wait

BEST=$($PY - <<'PY'
import json
from pathlib import Path
root=Path('logs/avoiding/biretopk_v1/k2_calibration')
metrics=['endpoint_mse_cvar20','velocity_mse_cvar20','multi_noise_trajectory_paired_mse',
         'teacher_to_student_set_coverage','student_to_teacher_set_precision']
rows={p.parent.name.replace('keep',''):json.loads(p.read_text()) for p in root.glob('keep*/proxy.json')}
best={m:min(rows,key=lambda k:rows[k][m]) for m in metrics}
unanimous=len(set(best.values()))==1
result={'phase':'P2a L4K2 six-mask calibration','metrics':metrics,'best_by_metric':best,
        'unanimous':unanimous,
        'values':{k:{m:v[m] for m in metrics} for k,v in rows.items()}}
Path('logs/avoiding/biretopk_v1/p2a_k2_calibration.json').write_text(json.dumps(result,indent=2))
print(next(iter(best.values())) if unanimous else 'BLOCKED')
PY
)
if [ "$BEST" = BLOCKED ]; then
  echo "P2 stopped: proxy metrics do not agree on a unique Top-2 ground truth."
  exit 3
fi

# P2b: generic Top-2 search, four independent seeds.
run_search () {
  local gpu=$1 seed=$2 out=$ROOT/k2_seed_${seed}
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu $PY train_discrete_bilevel_depth_search.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$out" \
    --rounds 40 --inner-epochs 5 --inner-batches 4 --batch-size 256 \
    --outer-groups 32 --samples-per-state 4 --architecture-lr 0.15 \
    --target-k 2 --seed "$seed" > "$out/train.log" 2>&1
}
run_search 0 42 & run_search 1 43 & run_search 2 44 & run_search 3 45 & wait

$PY - "$BEST" <<'PY'
import json,sys
from pathlib import Path
best=[int(x) for x in sys.argv[1]]
root=Path('logs/avoiding/biretopk_v1')
rows=[]
for seed in (42,43,44,45):
    x=json.loads((root/f'k2_seed_{seed}/summary.json').read_text())
    rows.append({'seed':seed,'selected_layers':x['selected_layers'],
                 'inclusion_probabilities':x['final_inclusion_probabilities']})
passed=sum(r['selected_layers']==best for r in rows)
result={'phase':'P2b L4K2 generic BiReTopK','calibrated_best':best,'rows':rows,
        'passed_seeds':passed,'go':passed>=3}
(root/'p2b_k2_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
PY
