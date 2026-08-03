"""Generate demonstration-free paired Teacher/Student closed-loop transfer data."""
import argparse,json
from collections import deque
from pathlib import Path

import numpy as np,torch
from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
from teacher_flow_deployment import DeploymentScaler,build_flow
from train_bmd_inference_model import InferenceModel,resample_prefix


def load_policy(path,layers,embed_dim,heads,steps,meta):
    model=build_flow(layers,embed_dim,heads,'cuda',steps)
    checkpoint=path if path.is_file() else path/'eval_best_flow.pth'
    model.load_state_dict(torch.load(checkpoint,map_location='cuda'),strict=True)
    model.min_action=meta['y_bounds_tensor'][0].cuda();model.max_action=meta['y_bounds_tensor'][1].cuda();model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model


def classify_path(model,checkpoint,path):
    feature=resample_prefix(path,1.0);feature=(feature-checkpoint['mean'])/checkpoint['std']
    with torch.no_grad():return int(model(torch.from_numpy(feature).float().unsqueeze(0).cuda()).argmax(1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle-dir',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--student',type=Path,required=True);p.add_argument('--classifier',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--episode-start',type=int,required=True);p.add_argument('--n-episodes',type=int,required=True);p.add_argument('--seed',type=int,default=27182);p.add_argument('--progress-every',type=int,default=10);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    meta=torch.load(a.bundle_dir/'deployment_metadata.pt',map_location='cpu');scaler=DeploymentScaler(meta,'cuda')
    teacher=load_policy(a.teacher,3,48,3,16,meta);student=load_policy(a.student,3,48,3,1,meta)
    qckpt=torch.load(a.classifier,map_location='cpu');classifier=InferenceModel(qckpt['model'],24).cuda();classifier.load_state_dict(qckpt['state_dict']);classifier.eval()
    env=ObstacleAvoidanceEnv(render=False);env.start()
    states=[];noises=[];student_endpoints=[];teacher_corrections=[];episode_ids=[];control_steps=[]
    teacher_paths=[];student_paths=[];teacher_successes=[];student_successes=[];teacher_modes=[];student_modes=[];latents=[]
    def execute(policy,eid,collect=False):
        np.random.seed(a.seed+eid);torch.manual_seed(a.seed+eid);context=deque(maxlen=meta['window_size']);obs=env.reset();pred_action=env.robot_state();fixed_z=pred_action[2:];path=[env.robot.current_c_pos[:2].copy()];done=False;info=(np.zeros(9),False);step=0
        while not done:
            raw=np.concatenate((pred_action[:2],obs));scaled=scaler.scale_input(torch.from_numpy(raw).float().view(1,4));context.append(scaled);state=torch.stack(tuple(context),dim=1)
            gen=torch.Generator(device='cpu').manual_seed(a.seed+eid*100003+step*1009);noise=torch.randn(1,state.shape[1],2,generator=gen).to(state.device)
            with torch.no_grad():endpoint=policy.sample(state,initial_noise=noise,steps=policy.solver_steps)
            if collect and len(context)==meta['window_size']:
                with torch.no_grad():correction=teacher.sample(state,initial_noise=noise,steps=16)
                states.append(state[0].cpu());noises.append(noise[0].cpu());student_endpoints.append(endpoint[0].cpu());teacher_corrections.append(correction[0].cpu());episode_ids.append(eid);control_steps.append(step)
            action=scaler.inverse_scale_output(endpoint[:,-1])[0].cpu().numpy();pred_action=action+raw[:2];command=np.concatenate((pred_action,fixed_z,[0,1,0,0]));obs,_,done,info=env.step(command);path.append(env.robot.current_c_pos[:2].copy());step+=1
        return np.asarray(path),bool(info[1]),np.asarray(info[0],dtype=np.int8)
    for local in range(a.n_episodes):
        eid=a.episode_start+local;teacher_path,teacher_success,teacher_mode=execute(teacher,eid,False);latent=classify_path(classifier,qckpt,teacher_path);student_path,student_success,student_mode=execute(student,eid,True)
        teacher_paths.append(teacher_path);student_paths.append(student_path);teacher_successes.append(teacher_success);student_successes.append(student_success);teacher_modes.append(teacher_mode);student_modes.append(student_mode);latents.append(latent)
        if (local+1)%a.progress_every==0 or local+1==a.n_episodes:print(json.dumps({'paired_progress':{'completed':local+1,'total':a.n_episodes,'teacher_successes':int(sum(teacher_successes)),'student_successes':int(sum(student_successes))}}),flush=True)
    payload={'states':torch.stack(states),'noises':torch.stack(noises),'student_endpoints':torch.stack(student_endpoints),'teacher_corrections':torch.stack(teacher_corrections),'episode_ids':torch.tensor(episode_ids),'control_steps':torch.tensor(control_steps),'teacher_paths':teacher_paths,'student_paths':student_paths,'teacher_successes':torch.tensor(teacher_successes),'student_successes':torch.tensor(student_successes),'teacher_modes':torch.from_numpy(np.stack(teacher_modes)),'student_modes':torch.from_numpy(np.stack(student_modes)),'teacher_latents':torch.tensor(latents),'metadata':{'uses_original_demonstrations':False,'uses_expert_actions':False,'teacher':str(a.teacher),'student':str(a.student),'classifier':str(a.classifier),'episode_start':a.episode_start,'n_episodes':a.n_episodes,'workers_per_gpu':4}}
    torch.save(payload,a.output_dir/'student_induced_buffer.pt')


if __name__=='__main__':main()
