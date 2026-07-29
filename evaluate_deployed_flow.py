import argparse,json,time
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt,numpy as np,torch
from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from envs.gym_avoiding_env.gym_avoiding.envs.objects.avoiding_objects import get_obj_xy_list
from teacher_flow_deployment import build_flow,DeploymentScaler

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--layers',type=int,default=3);p.add_argument('--embed-dim',type=int,default=48);p.add_argument('--heads',type=int,default=3);p.add_argument('--steps',type=int,default=1);p.add_argument('--n-trajectories',type=int,required=True);p.add_argument('--episode-start',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 meta=torch.load(a.bundle_dir/'deployment_metadata.pt',map_location='cpu');scaler=DeploymentScaler(meta,'cuda');model=build_flow(a.layers,a.embed_dim,a.heads,'cuda',a.steps);model.load_state_dict(torch.load(a.model_dir/'eval_best_flow.pth',map_location='cuda'),strict=True);model.min_action=meta['y_bounds_tensor'][0].cuda();model.max_action=meta['y_bounds_tensor'][1].cuda();model.eval();env=ObstacleAvoidanceEnv(render=False);env.start();paths=[];successes=[];modes=[];start=time.monotonic()
 for ep in range(a.n_trajectories):
  global_ep=a.episode_start+ep;np.random.seed(a.seed+global_ep);torch.manual_seed(a.seed+global_ep);ctx=deque(maxlen=meta['window_size']);obs=env.reset();pred=env.robot_state();fixed_z=pred[2:];path=[env.robot.current_c_pos[:2].copy()];done=False;info=(np.zeros(9),False)
  while not done:
   raw=np.concatenate((pred[:2],obs));s=scaler.scale_input(torch.from_numpy(raw).float().view(1,4));ctx.append(s);state=torch.stack(tuple(ctx),dim=1)
   with torch.no_grad():out=model.sample(state,steps=a.steps)
   action=scaler.inverse_scale_output(out[:,-1])[0].cpu().numpy();pred=action+raw[:2];command=np.concatenate((pred,fixed_z,[0,1,0,0]));obs,_,done,info=env.step(command);path.append(env.robot.current_c_pos[:2].copy())
  paths.append(np.asarray(path));successes.append(bool(info[1]));modes.append(np.asarray(info[0],dtype=np.int8))
  if (ep+1)%10==0:print(json.dumps({'progress':{'completed':ep+1,'total':a.n_trajectories,'successes':int(sum(successes))}}),flush=True)
 successes=np.asarray(successes);modes=np.stack(modes);sm=modes[successes]
 if len(sm): enc=sm.dot(1<<np.arange(sm.shape[1]));_,counts=np.unique(enc,return_counts=True);q=counts/counts.sum();coverage=len(counts);entropy=float(-(q*np.log(q)/np.log(24)).sum())
 else:coverage=0;entropy=0.
 metrics={'n_trajectories':len(successes),'success_rate':float(successes.mean()),'successful_trajectories':int(successes.sum()),'unique_successful_modes':int(coverage),'normalized_mode_entropy':entropy,'uses_original_demonstrations':False}
 (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(a.output_dir/'trajectories.npz',trajectories=np.asarray(paths,dtype=object),successes=successes,modes=modes)
 fig,ax=plt.subplots(figsize=(6,6));
 for path,ok in zip(paths,successes):ax.plot(path[:,0],path[:,1],color='#1976d2' if ok else '#d32f2f',alpha=.45,lw=1.2)
 for x,y in get_obj_xy_list():ax.add_patch(plt.Circle((x,y),.03,color='black',alpha=.8))
 ax.axhline(.35,color='#2e7d32',ls='--');ax.set(xlim=(.25,.75),ylim=(-.32,.42),xlabel='x [m]',ylabel='y [m]',title=f"SR={metrics['success_rate']:.1%}, modes={coverage}/24, H={entropy:.3f}");ax.set_aspect('equal');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(a.output_dir/'trajectory_comparison.png',dpi=220);print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
