"""Generate demonstration-free transfer data from deployed teacher + environment."""
import argparse, json
from collections import deque
from pathlib import Path
import numpy as np
import torch
from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from teacher_flow_deployment import DeploymentScaler,build_flow,load_deployed_teacher


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--episode-start',type=int,default=0)
    p.add_argument('--n-episodes',type=int,required=True)
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--progress-every',type=int,default=10)
    p.add_argument('--model-dir',type=Path,default=None)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--embed-dim',type=int,default=48)
    p.add_argument('--heads',type=int,default=3)
    p.add_argument('--steps',type=int,default=16)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    if a.model_dir is None:
        teacher,scaler,meta=load_deployed_teacher(a.bundle_dir)
        source='deployed_teacher_plus_environment'
    else:
        meta=torch.load(a.bundle_dir/'deployment_metadata.pt',map_location='cpu')
        scaler=DeploymentScaler(meta,'cuda')
        teacher=build_flow(a.layers,a.embed_dim,a.heads,'cuda',a.steps)
        checkpoint=a.model_dir if a.model_dir.is_file() else a.model_dir/'eval_best_flow.pth'
        teacher.load_state_dict(torch.load(checkpoint,map_location='cuda'),strict=True)
        teacher.min_action=meta['y_bounds_tensor'][0].cuda()
        teacher.max_action=meta['y_bounds_tensor'][1].cuda()
        teacher.eval()
        for parameter in teacher.parameters(): parameter.requires_grad_(False)
        source='repair_checkpoint_plus_environment'
    env=ObstacleAvoidanceEnv(render=False); env.start()
    states=[]; noises=[]; endpoints=[]; episode_ids=[]; control_steps=[]
    paths=[]; successes=[]; modes=[]
    for local in range(a.n_episodes):
        eid=a.episode_start+local
        np.random.seed(a.seed+eid); torch.manual_seed(a.seed+eid)
        context=deque(maxlen=meta['window_size'])
        obs=env.reset(); pred_action=env.robot_state(); fixed_z=pred_action[2:]
        path=[env.robot.current_c_pos[:2].copy()]; done=False; info=(np.zeros(9),False); step=0
        while not done:
            raw=np.concatenate((pred_action[:2],obs))
            scaled=scaler.scale_input(torch.from_numpy(raw).float().view(1,4))
            context.append(scaled)
            state=torch.stack(tuple(context),dim=1)
            gen=torch.Generator(device='cpu').manual_seed(a.seed+eid*100003+step*1009)
            noise=torch.randn(1,state.shape[1],2,generator=gen).to(state.device)
            with torch.no_grad(): endpoint=teacher.sample(state,initial_noise=noise,steps=a.steps if a.model_dir is not None else meta['teacher_steps'])
            if len(context)==meta['window_size']:
                states.append(state[0].cpu()); noises.append(noise[0].cpu()); endpoints.append(endpoint[0].cpu())
                episode_ids.append(eid); control_steps.append(step)
            action=scaler.inverse_scale_output(endpoint[:,-1])[0].cpu().numpy()
            pred_action=action+raw[:2]
            command=np.concatenate((pred_action,fixed_z,[0,1,0,0]))
            obs,_,done,info=env.step(command); path.append(env.robot.current_c_pos[:2].copy()); step+=1
        paths.append(np.asarray(path)); successes.append(bool(info[1])); modes.append(np.asarray(info[0],dtype=np.int8))
        done_count=local+1
        if done_count%a.progress_every==0 or done_count==a.n_episodes:
            print(json.dumps({'buffer_progress':{'completed':done_count,'total':a.n_episodes,'successes':int(sum(successes))}}),flush=True)
    payload={'states':torch.stack(states),'noises':torch.stack(noises),'teacher_endpoints':torch.stack(endpoints),
             'episode_ids':torch.tensor(episode_ids),'control_steps':torch.tensor(control_steps),
             'successes':torch.tensor(successes),'modes':torch.from_numpy(np.stack(modes)),
             'paths':paths,'metadata':{'source':source,'source_model':str(a.model_dir) if a.model_dir is not None else str(meta['source_checkpoint']),
             'uses_original_demonstrations':False,'uses_expert_actions':False,'episode_start':a.episode_start,
             'n_episodes':a.n_episodes,'seed':a.seed,'solver_steps':a.steps if a.model_dir is not None else meta['teacher_steps']}}
    torch.save(payload,a.output_dir/'transfer_buffer.pt')
    print(json.dumps({'samples':len(states),'episodes':a.n_episodes,'success_rate':float(np.mean(successes))}),flush=True)
if __name__=='__main__': main()
