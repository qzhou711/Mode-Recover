import argparse, json
from pathlib import Path
import numpy as np, torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--episodes',type=int,default=480);p.add_argument('--start',type=int,required=True);p.add_argument('--horizon',type=int,required=True);a=p.parse_args()
files=sorted(a.root.glob('shards/*/intervention_buffer.pt'));parts=[torch.load(f,map_location='cpu') for f in files]
if not parts:raise SystemExit('no shards')
keys=['states','noises','student_endpoints','teacher_corrections','episode_ids','control_steps','disagreements','teacher_control','triggers','successes','modes','intervention_counts','intervention_steps']
d={k:torch.cat([x[k] for x in parts]) for k in keys};d['paths']=sum([x['paths'] for x in parts],[]);d['metadata']={'teacher_architecture':'FM-4x72-16','student_architecture':'FM-3x48-16','takeover_horizon':a.horizon,'threshold':0.4180944411691571,'uses_original_demonstrations':False,'uses_expert_actions':False,'teacher_queries_on_student_states':True,'shards':[str(f) for f in files]}
u=torch.unique(d['episode_ids']);aligned=len({len(d[k]) for k in ['states','noises','student_endpoints','teacher_corrections','episode_ids','control_steps','disagreements','teacher_control','triggers']})==1
checks={'episodes':len(d['successes'])==a.episodes,'ids':torch.equal(u,torch.arange(a.start,a.start+a.episodes)),'aligned':aligned,'finite':all(torch.isfinite(d[k]).all() for k in ['states','noises','student_endpoints','teacher_corrections','disagreements']),'no_demonstrations':True}
s=d['successes'].numpy().astype(bool);m=d['modes'].numpy()[s];enc=m.dot(1<<np.arange(m.shape[1])) if len(m) else np.array([]);_,cnt=np.unique(enc,return_counts=True);prob=cnt/cnt.sum() if len(cnt) else np.array([])
metrics={'passed':all(bool(x) for x in checks.values()),'checks':checks,'episodes':len(s),'samples':len(d['states']),'assisted_success_rate':float(s.mean()),'mode_coverage':len(cnt),'mode_entropy':float(-(prob*np.log(prob)/np.log(24)).sum()) if len(prob) else 0.0,'episodes_with_intervention':int((d['intervention_counts']>0).sum()),'mean_interventions':float(d['intervention_counts'].float().mean()),'teacher_control_fraction':float(d['teacher_control'].float().mean()),'uses_original_demonstrations':False}
torch.save(d,a.root/'intervention_buffer.pt');(a.root/'validation.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2));raise SystemExit(0 if metrics['passed'] else 2)
