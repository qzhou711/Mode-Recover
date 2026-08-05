#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/biretopk_v1
EXP=$ROOT/k2_disambiguation
BUNDLE=logs/avoiding/teacher_deployment_bundle
BUFFER=logs/avoiding/strong_teacher_expanded_2400/transfer_buffer.pt

run_case () {
  local gpu=$1
  local seed=$2
  local mask=$3
  local name=${mask//,/}
  local out=$EXP/seed_${seed}/keep${name}
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu $PY train_recoverable_depth_pruning.py \
    --bundle-dir "$BUNDLE" --buffer "$BUFFER" --output-dir "$out" \
    --layer-map "$mask" --epochs 25 --batch-size 256 --max-batches 4 \
    --save-epochs 10,25 --all-rollouts --holdout-residue 2 --seed "$seed" \
    > "$out/train.log" 2>&1
  for ep in 0010 0025; do
    CUDA_VISIBLE_DEVICES=$gpu $PY audit_depth_recoverability_proxies.py \
      --bundle-dir "$BUNDLE" --buffer "$BUFFER" --student-layers 2 \
      --student-checkpoint "$out/pretrain_epoch_${ep}.pth" \
      --output "$out/proxy_epoch_${ep}.json" --mask "keep${name}" --stage "epoch_${ep}" \
      --samples 512 --groups 48 --samples-per-state 4 --batch-size 128 \
      --seed "$seed" --episode-residue 2 > "$out/proxy_epoch_${ep}.log" 2>&1
  done
}

# Two paired waves: each seed's two masks run simultaneously on separate GPUs.
run_case 0 42 0,2 & run_case 1 42 0,3 & run_case 2 43 0,2 & run_case 3 43 0,3 & wait
run_case 0 44 0,2 & run_case 1 44 0,3 & run_case 2 45 0,2 & run_case 3 45 0,3 & wait

BEST=$($PY - <<'PY'
import json
from pathlib import Path
root=Path('logs/avoiding/biretopk_v1/k2_disambiguation')
behavior=['endpoint_mse_cvar20','multi_noise_trajectory_paired_mse',
          'teacher_to_student_set_coverage','student_to_teacher_set_precision']
velocity='velocity_mse_cvar20'
comparisons=[]
for seed in (42,43,44,45):
  for ep in ('0010','0025'):
    a=json.loads((root/f'seed_{seed}/keep02/proxy_epoch_{ep}.json').read_text())
    b=json.loads((root/f'seed_{seed}/keep03/proxy_epoch_{ep}.json').read_text())
    comparisons.append({'seed':seed,'epoch':int(ep),
      'winner_by_metric':{m:('02' if a[m]<b[m] else '03') for m in behavior+[velocity]},
      'keep02':{m:a[m] for m in behavior+[velocity]},
      'keep03':{m:b[m] for m in behavior+[velocity]}})
wins={m:{x:sum(c['winner_by_metric'][m]==x for c in comparisons) for x in ('02','03')}
      for m in behavior+[velocity]}
eligible=[]
for candidate in ('02','03'):
  if all(wins[m][candidate]>=7 for m in behavior): eligible.append(candidate)
best=eligible[0] if len(eligible)==1 else None
result={'comparisons':comparisons,'wins':wins,'behavior_metrics':behavior,
        'confirmed_best':best,'go':best is not None,
        'velocity_conflict':best is not None and wins[velocity][best]<4}
(root/'summary.json').write_text(json.dumps(result,indent=2))
print(best or 'BLOCKED')
PY
)
if [ "$BEST" = BLOCKED ]; then
  echo "Top-2 search blocked: behavior/distribution metrics are not stable."
  exit 3
fi

run_search () {
  local gpu=$1
  local seed=$2
  local out=$ROOT/k2_confirmed_seed_${seed}
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
 x=json.loads((root/f'k2_confirmed_seed_{seed}/summary.json').read_text())
 rows.append({'seed':seed,'selected_layers':x['selected_layers'],
              'inclusion_probabilities':x['final_inclusion_probabilities']})
passed=sum(x['selected_layers']==best for x in rows)
result={'calibrated_best':best,'rows':rows,'passed_seeds':passed,'go':passed>=3}
(root/'p2b_k2_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
PY
