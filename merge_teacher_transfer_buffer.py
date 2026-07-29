import argparse,json,math
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
files=sorted(a.root.glob('shards/*/transfer_buffer.pt'))
if not files: raise SystemExit('no shard buffers')
parts=[torch.load(f,map_location='cpu') for f in files]
tensor_keys=['states','noises','teacher_endpoints','episode_ids','control_steps','successes','modes']
merged={k:torch.cat([x[k] for x in parts],dim=0) for k in tensor_keys}
merged['paths']=sum([x['paths'] for x in parts],[])
merged['metadata']={'source':'deployed_teacher_plus_environment','uses_original_demonstrations':False,'uses_expert_actions':False,'shards':[str(f) for f in files]}
success=merged['successes'].numpy().astype(bool); modes=merged['modes'].numpy()[success]
if len(modes):
 enc=modes.dot(1<<np.arange(modes.shape[1])); _,counts=np.unique(enc,return_counts=True); pmode=counts/counts.sum()
 coverage=len(counts); entropy=float(-(pmode*np.log(pmode)/np.log(24)).sum())
else: coverage=0; entropy=0.0
metrics={'episodes':int(len(success)),'samples':int(len(merged['states'])),'success_rate':float(success.mean()),'successful_episodes':int(success.sum()),'mode_coverage':int(coverage),'normalized_mode_entropy':entropy,'uses_original_demonstrations':False,'uses_expert_actions':False}
torch.save(merged,a.root/'transfer_buffer.pt');(a.root/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
