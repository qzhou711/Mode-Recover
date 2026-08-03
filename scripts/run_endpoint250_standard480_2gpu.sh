#!/usr/bin/env bash
set -euo pipefail
cd /ocean/projects/ccr200024p/qzhou7/projects/d3il
PY=/jet/home/qzhou7/workspace/anaconda3/envs/d3il/bin/python
ROOT=logs/avoiding/keep013_step_distillation/endpoint/epoch_250/eval480
MODEL=logs/avoiding/keep013_step_distillation/endpoint/model/checkpoints/epoch_0250
mkdir -p "$ROOT"
echo "RUNNING $(date --iso-8601=seconds)" > "$ROOT/STATUS"
cat > "$ROOT/audit.txt" <<'EOF'
Long-term objective: demonstration-free structure+step distillation with mode preservation.
Short-term bottleneck: confirm the Endpoint-250 Standard-120 Pareto point at publication-scale evaluation.
Stage targeted: evaluation of step distillation; no training changes.
Mechanism hypothesis: Endpoint fidelity preserves most but not all keep013 noise-to-mode basins.
Causal intervention and controls: exact epoch-250 checkpoint, boundary 0->1 inference, episodes 0..479, seed42.
Data-access status: evaluation only; no demonstrations or expert actions.
Metrics and go/no-go: SR, successful-mode coverage, entropy, raw trajectories; Standard-480.
Resource/time check: two V100, four workers/GPU, more than two hours remaining.
Expected interpretations: stable >=20 coverage supports endpoint as Pareto baseline; substantial drop falsifies the Standard-120 ranking.
EOF
scripts/run_deployed_flow_parallel_eval.sh 0 "$MODEL" "$ROOT/gpu0_ep000_239" 240 0 42 4 1 3 72 4 "" --inference-mode boundary > "$ROOT/gpu0.log" 2>&1 & p0=$!
scripts/run_deployed_flow_parallel_eval.sh 1 "$MODEL" "$ROOT/gpu1_ep240_479" 240 240 42 4 1 3 72 4 "" --inference-mode boundary > "$ROOT/gpu1.log" 2>&1 & p1=$!
failed=0; wait "$p0" || failed=1; wait "$p1" || failed=1
if ((failed)); then echo "FAILED $(date --iso-8601=seconds)" > "$ROOT/STATUS"; exit 1; fi
"$PY" scripts/merge_deployed_flow_eval_shards.py --shards "$ROOT/gpu0_ep000_239" "$ROOT/gpu1_ep240_479" --output-dir "$ROOT"
"$PY" - <<'PY'
import json, numpy as np
from pathlib import Path
root=Path('logs/avoiding/keep013_step_distillation/endpoint/epoch_250/eval480')
d=np.load(root/'trajectories.npz',allow_pickle=True)
assert len(d['successes'])==480
m=json.loads((root/'metrics.json').read_text())
assert m['n_trajectories']==480
(root/'protocol.json').write_text(json.dumps({'checkpoint':'endpoint epoch-250','episodes':[0,479],'seed':42,'inference_mode':'boundary','gpus':2,'workers_per_gpu':4,'uses_original_demonstrations':False},indent=2))
PY
echo "COMPLETE $(date --iso-8601=seconds)" > "$ROOT/STATUS"
