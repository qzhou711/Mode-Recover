"""Demonstration-free structure transfer followed by Flow-CTM on teacher-generated data."""
import argparse,copy,json,math
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from agents.models.flow_matching.ctm import centered_gram_loss,ctm_paths,pseudo_huber,freeze,update_ema
from agents.models.diffusion.ema import ExponentialMovingAverage
from environments.dataset.avoiding_dataset import Avoiding_Dataset
from teacher_flow_deployment import build_flow,load_deployed_teacher
from train_flow_progressive_compression import initialize_student,selection_basis


def activation_matrix(teacher,states,noises,batches,batch_size):
    got=[]
    def hook(_m,inputs): got.append(inputs[0].detach().reshape(-1,inputs[0].shape[-1]).cpu())
    handle=teacher.model.blocks[0].register_forward_pre_hook(hook)
    with torch.no_grad():
        for i in range(min(batches,math.ceil(len(states)/batch_size))):
            s=states[i*batch_size:(i+1)*batch_size].to(teacher.device); n=noises[i*batch_size:(i+1)*batch_size].to(teacher.device)
            t=float((i+1)/(batches+1)); x=teacher.integrate(n,s,start_time=0.0,end_time=t,steps=max(1,round(16*t)))
            teacher.velocity(x,torch.full((len(s),),t,device=s.device),s)
    handle.remove(); return torch.cat(got)


def init_student(teacher,student,method,acts,layers,heads):
    if method=='random': return {'method':'random'}
    if method in {'activation','early','width'}:
        basis,meta=selection_basis(acts,layers,heads)
        if method=='early': meta['teacher_layers']=[0,1,2]
    elif method=='pca':
        centered=acts-acts.mean(0,keepdim=True); _,sv,vh=torch.linalg.svd(centered,full_matrices=False); basis=vh[:48].T.contiguous()
        signs=torch.sign(basis[torch.argmax(basis.abs(),dim=0),torch.arange(48)]); basis*=torch.where(signs==0,torch.ones_like(signs),signs)
        meta={'teacher_layers':[0,2,3],'explained_variance':float(sv[:48].square().sum()/sv.square().sum())}
    else: raise ValueError(method)
    meta['method']=method; meta['load_max_abs_diff']=initialize_student(teacher,student,basis,meta['teacher_layers']); return meta


def save_ema(model,ema,path):
    ema.store(model.get_params());ema.copy_to(model.get_params());torch.save(model.state_dict(),path);ema.restore(model.get_params())


def relation_forward(model,x,t,state,layers,full_minilm=False,relation_heads=3):
    captured=[];handles=[]
    for index in layers:
        handles.append(model.model.blocks[index].register_forward_pre_hook(lambda _m,inputs: captured.append(inputs[0])))
    output=model.velocity(x,t,state)
    for handle in handles: handle.remove()
    relations=[]
    for index,hidden in zip(layers,captured):
        block=model.model.blocks[index];normalized=block.ln1(hidden);attn=block.attn;batch,tokens,_=normalized.shape
        q=attn.query(normalized).reshape(batch,tokens,relation_heads,-1).transpose(1,2);k=attn.key(normalized).reshape(batch,tokens,relation_heads,-1).transpose(1,2);v=attn.value(normalized).reshape(batch,tokens,relation_heads,-1).transpose(1,2)
        qk=torch.softmax((q@k.transpose(-2,-1))/math.sqrt(q.shape[-1]),dim=-1)
        qn=F.normalize(q,dim=-1);kn=F.normalize(k,dim=-1);vn=F.normalize(v,dim=-1)
        qq=qn@qn.transpose(-2,-1);kk=kn@kn.transpose(-2,-1);vv=vn@vn.transpose(-2,-1)
        relations.append({'qk':qk,'qq':qq,'kk':kk,'vv':vv,'q':q,'k':k,'v':v} if full_minilm else {'qk':qk,'vv':vv})
    return output,relations

def relation_loss(student_rel,teacher_rel,full_minilm=False):
    values=[]
    keys=('qk','qq','kk','vv') if full_minilm else ('qk','vv')
    for student_layer,teacher_layer in zip(student_rel,teacher_rel):
        values.append(torch.stack([F.mse_loss(student_layer[key],teacher_layer[key]) for key in keys]).mean())
    return torch.stack(values).mean()

def cross_noise_relation_loss(student_rel,teacher_rel,multi_noise):
    """Match teacher geometry among K noise outcomes sharing one state."""
    values=[]
    for student_layer,teacher_layer in zip(student_rel,teacher_rel):
        for key in ('q','k','v'):
            sf=student_layer[key].flatten(1);tf=teacher_layer[key].flatten(1)
            sf=F.normalize(sf.reshape(-1,multi_noise,sf.shape[-1]),dim=-1)
            tf=F.normalize(tf.reshape(-1,multi_noise,tf.shape[-1]),dim=-1)
            values.append(F.mse_loss(sf@sf.transpose(-2,-1),tf@tf.transpose(-2,-1)))
    return torch.stack(values).mean()

def differentiable_integrate(model,x,state,steps=16,start_time=0.0,end_time=1.0):
    """Heun integration without the deployment-time no_grad decorator."""
    dt=(end_time-start_time)/steps
    for index in range(steps):
        current=start_time+index*dt
        t=torch.full((len(x),),current,device=x.device,dtype=x.dtype)
        velocity=model.velocity(x,t,state)
        if index+1==steps:
            x=x+dt*velocity
        else:
            predictor=x+dt*velocity
            next_t=torch.full_like(t,current+dt)
            next_velocity=model.velocity(predictor,next_t,state)
            x=x+0.5*dt*(velocity+next_velocity)
    return x

def demonstration_dataset(data_dir,metadata):
    """Load the original expert windows using the deployed teacher's scaling."""
    raw=Avoiding_Dataset(data_dir,device='cpu',obs_dim=4,action_dim=2,max_len_data=200,window_size=5)
    states=[];actions=[]
    for trajectory,start,end in raw.slices:
        states.append(raw.observations[trajectory,start:end])
        actions.append(raw.actions[trajectory,start:end])
    states=torch.stack(states).float();actions=torch.stack(actions).float()
    states=(states-metadata['x_mean'].cpu())/(metadata['x_std'].cpu()+1e-12)
    actions=(actions-metadata['y_mean'].cpu())/(metadata['y_std'].cpu()+1e-12)
    return TensorDataset(states.float(),torch.zeros_like(actions),actions.float())

def make_ctm_loader(rollout_dataset,demo_dataset,source,batch_size,seed):
    if source=='rollout':
        return DataLoader(rollout_dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
    if source=='demonstration':
        return DataLoader(demo_dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
    states=torch.cat((rollout_dataset.tensors[0],demo_dataset.tensors[0]))
    noises=torch.cat((rollout_dataset.tensors[1],demo_dataset.tensors[1]))
    actions=torch.cat((rollout_dataset.tensors[2].float(),demo_dataset.tensors[2]))
    combined=TensorDataset(states,noises,actions)
    n_rollout=len(rollout_dataset);n_demo=len(demo_dataset)
    weights=torch.cat((torch.full((n_rollout,),0.5/n_rollout),torch.full((n_demo,),0.5/n_demo)))
    generator=torch.Generator().manual_seed(seed)
    sampler=WeightedRandomSampler(weights,n_rollout+n_demo,replacement=True,generator=generator)
    return DataLoader(combined,batch_size=batch_size,sampler=sampler,num_workers=0,drop_last=True)

def make_balanced_ctm_loader(dataset,labels,episode_ids,batch_size,seed,keep=None):
    labels=torch.as_tensor(labels).long()
    episode_ids=torch.as_tensor(episode_ids).long()
    if len(labels)!=len(dataset):
        raise ValueError(f'expected {len(dataset)} CTM labels, got {len(labels)}')
    if len(episode_ids)!=len(dataset):
        raise ValueError(f'expected {len(dataset)} episode ids, got {len(episode_ids)}')
    keep=torch.ones(len(dataset),dtype=torch.bool) if keep is None else torch.as_tensor(keep).bool()
    if not bool(keep.any()):
        raise ValueError('balanced CTM selection removed every sample')
    tensors=tuple(tensor[keep] for tensor in dataset.tensors)
    selected_labels=labels[keep]
    selected_episodes=episode_ids[keep]
    # Hierarchical target: z uniformly, then episode uniformly within z, then
    # timestep uniformly within episode. This prevents long trajectories from
    # receiving more total probability mass.
    episode_pairs=torch.stack((selected_labels,selected_episodes),dim=1)
    unique_pairs,pair_inverse,pair_counts=torch.unique(
        episode_pairs,dim=0,return_inverse=True,return_counts=True
    )
    latent_values,latent_episode_inverse,latent_episode_counts=torch.unique(
        unique_pairs[:,0],return_inverse=True,return_counts=True
    )
    pair_latent_episode_count=latent_episode_counts[latent_episode_inverse]
    weights=(pair_latent_episode_count[pair_inverse]*pair_counts[pair_inverse]).float().reciprocal()
    generator=torch.Generator().manual_seed(seed)
    sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=generator)
    return DataLoader(TensorDataset(*tensors,selected_labels),batch_size=batch_size,sampler=sampler,num_workers=0,drop_last=True)

def conditional_rbf_mmd(student_samples,teacher_samples,labels):
    """Biased, stable RBF-MMD averaged over latent groups in the minibatch."""
    student_samples=student_samples.flatten(1)
    teacher_samples=teacher_samples.flatten(1)
    losses=[]
    for label in torch.unique(labels):
        keep=labels==label
        if int(keep.sum()) < 2:
            continue
        x=student_samples[keep]
        y=teacher_samples[keep]
        joined=torch.cat((x.detach(),y.detach()))
        with torch.no_grad():
            distances=torch.pdist(joined).square()
            bandwidth=distances.median().clamp_min(1e-6) if len(distances) else joined.new_tensor(1.0)
        kxx=torch.exp(-torch.cdist(x,x).square()/(2*bandwidth))
        kyy=torch.exp(-torch.cdist(y,y).square()/(2*bandwidth))
        kxy=torch.exp(-torch.cdist(x,y).square()/(2*bandwidth))
        losses.append(kxx.mean()+kyy.mean()-2*kxy.mean())
    return torch.stack(losses).mean() if losses else student_samples.sum()*0.0

def make_native_trajectory_loader(data,labels,horizon,batch_size,seed):
    """Consecutive Teacher-rollout windows preserving state/noise/endpoint identity."""
    episode_ids=data['episode_ids'].long()
    control_steps=data['control_steps'].long()
    labels=torch.as_tensor(labels).long()
    window_states=[];window_noises=[];window_endpoints=[];window_labels=[];window_episodes=[]
    for episode_id in torch.unique(episode_ids):
        indices=torch.where(episode_ids==episode_id)[0]
        indices=indices[torch.argsort(control_steps[indices])]
        if len(indices)<horizon:
            continue
        episode_labels=torch.unique(labels[indices])
        if len(episode_labels)!=1:
            raise ValueError(f'episode {int(episode_id)} has multiple latent labels')
        if int(episode_labels[0]) < 0:
            continue
        for start in range(len(indices)-horizon+1):
            chosen=indices[start:start+horizon]
            if not bool(torch.all(torch.diff(control_steps[chosen])==1)):
                continue
            window_states.append(data['states'][chosen])
            window_noises.append(data['noises'][chosen])
            window_endpoints.append(data['teacher_endpoints'][chosen])
            window_labels.append(episode_labels[0])
            window_episodes.append(episode_id)
    if not window_states:
        raise ValueError('no consecutive native trajectory windows available')
    dataset=TensorDataset(
        torch.stack(window_states).float(),
        torch.stack(window_noises).float(),
        torch.stack(window_endpoints).float(),
    )
    return make_balanced_ctm_loader(
        dataset,torch.stack(window_labels),torch.stack(window_episodes),batch_size,seed
    )

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle-dir',type=Path,required=True);p.add_argument('--buffer',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--method',choices=['activation_dynamic','pca_dynamic','early_dynamic','width_dynamic','pca_multinoise_endpoint','pca_balanced_multitime','pca_combined','early_multinoise_endpoint','early_combined','width_multinoise_endpoint','width_combined','attention_relation','minilmv2_relation','minilmv2_multinoise_relation'],required=True);p.add_argument('--pretrain-epochs',type=int,default=300);p.add_argument('--ctm-epochs',type=int,default=500);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--max-batches',type=int,default=4,help='Batches per epoch; 0 means a complete DataLoader pass.');p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--multi-noise',type=int,default=4);p.add_argument('--cross-noise-weight',type=float,default=1.0);p.add_argument('--relation-velocity-weight',type=float,default=.1);p.add_argument('--velocity-ramp-epochs',type=int,default=0);p.add_argument('--mode-balanced-sampling',action='store_true');p.add_argument('--pretrain-success-only',action='store_true');p.add_argument('--relation-endpoint-weight',type=float,default=0.0);p.add_argument('--student-induced-weight',type=float,default=0.0);p.add_argument('--endpoint-steps',type=int,default=16);p.add_argument('--endpoint-weight',type=float,default=.1);p.add_argument('--save-pretrain-epochs',type=str,default='');p.add_argument('--ctm-dsm-weight',type=float,default=.1);p.add_argument('--ctm-endpoint-anchor-weight',type=float,default=0.0);p.add_argument('--ctm-mode-weight',type=float,default=0.0);p.add_argument('--ctm-endpoint-mmd-weight',type=float,default=0.0);p.add_argument('--ctm-trajectory-mmd-weight',type=float,default=0.0);p.add_argument('--ctm-trajectory-horizon',type=int,default=4);p.add_argument('--ctm-conditional-samples',type=int,default=4);p.add_argument('--save-ctm-epochs',type=str,default='');p.add_argument('--ctm-data-source',choices=['rollout','demonstration','mixed'],default='rollout');p.add_argument('--ctm-latent-labels',type=Path,default=None);p.add_argument('--ctm-labels-for-loss-only',action='store_true');p.add_argument('--ctm-ground-truth-mode-balanced',action='store_true');p.add_argument('--ctm-success-only',action='store_true');p.add_argument('--demonstration-dir',type=Path,default=Path('environments/dataset/data/avoiding/data'));p.add_argument('--pretrain-only',action='store_true');p.add_argument('--initialization-only',action='store_true');p.add_argument('--random-initialization',action='store_true');p.add_argument('--initial-structure',type=Path,default=None);p.add_argument('--pretrained-structure',type=Path,default=None);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.ctm_conditional_samples < 2 and a.ctm_mode_weight > 0:
        p.error('--ctm-conditional-samples must be at least 2 when --ctm-mode-weight > 0')
    if a.ctm_endpoint_mmd_weight > 0 and a.ctm_latent_labels is None and not a.ctm_ground_truth_mode_balanced:
        p.error('--ctm-endpoint-mmd-weight requires latent labels or ground-truth mode labels')
    if a.ctm_trajectory_mmd_weight > 0 and a.ctm_latent_labels is None and not a.ctm_ground_truth_mode_balanced:
        p.error('--ctm-trajectory-mmd-weight requires latent labels or ground-truth mode labels')
    torch.manual_seed(a.seed);np.random.seed(a.seed);a.output_dir.mkdir(parents=True,exist_ok=True)
    data=torch.load(a.buffer,map_location='cpu'); assert not data['metadata']['uses_original_demonstrations']; assert not data['metadata']['uses_expert_actions']
    states=data['states'].float(); source_noises=data['noises'].float(); pseudo_actions=data['teacher_endpoints'].float()
    if a.pretrain_success_only:
        sample_success=data['successes'].bool()[data['episode_ids'].long()]
        states=states[sample_success];source_noises=source_noises[sample_success];pseudo_actions=pseudo_actions[sample_success]
    dataset=TensorDataset(states,source_noises,pseudo_actions)
    sampler=None
    if a.mode_balanced_sampling or ('balanced_multitime' in a.method or 'combined' in a.method):
        episode_ids=data['episode_ids'].long();mode_codes=(data['modes'].long()*(1<<torch.arange(data['modes'].shape[1]))).sum(1);sample_codes=mode_codes[episode_ids]
        _,inverse,counts=torch.unique(sample_codes,return_inverse=True,return_counts=True);weights=counts[inverse].float().reciprocal();sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=0,drop_last=True)
    teacher,_,meta=load_deployed_teacher(a.bundle_dir);layers,heads=(4,4) if a.method.startswith('width') else (3,3);student=build_flow(layers,48,heads,'cuda',16).train()
    demo_dataset=demonstration_dataset(a.demonstration_dir,meta) if a.ctm_data_source!='rollout' else None
    ctm_loader=make_ctm_loader(dataset,demo_dataset,a.ctm_data_source,a.batch_size,a.seed)
    ctm_balance_kind='none'
    if a.ctm_latent_labels is not None or a.ctm_ground_truth_mode_balanced:
        if a.ctm_data_source!='rollout':
            p.error('latent/mode-balanced CTM currently requires --ctm-data-source rollout')
        if a.ctm_latent_labels is not None:
            discovered=np.load(a.ctm_latent_labels)
            labels=discovered['sample_latents']
            ctm_balance_kind='bmd_latent'
        else:
            mode_codes=(data['modes'].long()*(1<<torch.arange(data['modes'].shape[1]))).sum(1)
            labels=mode_codes[data['episode_ids'].long()]
            ctm_balance_kind='ground_truth_mode_oracle'
        sample_success=data['successes'].bool()[data['episode_ids'].long()]
        keep=sample_success if a.ctm_success_only else None
        if not a.ctm_labels_for_loss_only:
            ctm_loader=make_balanced_ctm_loader(
                dataset,labels,data['episode_ids'].long(),a.batch_size,a.seed,keep
            )
    trajectory_loader=None
    if a.ctm_trajectory_mmd_weight>0:
        trajectory_loader=make_native_trajectory_loader(
            data,labels,a.ctm_trajectory_horizon,a.batch_size,a.seed+1000003
        )
    acts=activation_matrix(teacher,states,source_noises,8,a.batch_size)
    relation_method=a.method in {'attention_relation','minilmv2_relation','minilmv2_multinoise_relation'}
    full_minilm=a.method in {'minilmv2_relation','minilmv2_multinoise_relation'}
    relation_multinoise=a.method=='minilmv2_multinoise_relation'
    init_kind='pca' if a.method.startswith('pca') else 'early' if a.method.startswith('early') else 'activation' if a.method.startswith('activation') else 'early' if relation_method else 'width'
    initialization={'kind':'random','teacher_derived':False} if a.random_initialization else init_student(teacher,student,init_kind,acts,layers,heads)
    if a.initial_structure is not None:
        student.load_state_dict(torch.load(a.initial_structure,map_location='cuda'),strict=True)
        initialization['continued_from']=str(a.initial_structure)
    torch.save(student.state_dict(),a.output_dir/'initial_flow.pth')
    if a.initialization_only: torch.save(student.state_dict(),a.output_dir/'eval_best_flow.pth');(a.output_dir/'metrics.json').write_text(json.dumps({'method':a.method,'initialization':initialization,'uses_original_demonstrations':False,'uses_expert_actions':False},indent=2));return
    if a.pretrained_structure is not None:
        student.load_state_dict(torch.load(a.pretrained_structure,map_location='cuda'),strict=True)
        initialization['pretrained_structure']=str(a.pretrained_structure)
    milestones={int(x) for x in a.save_pretrain_epochs.split(',') if x};dynamic=a.pretrained_structure is None; prehistory=[]
    if dynamic:
        opt=torch.optim.Adam(student.get_params(),lr=a.learning_rate); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.pretrain_epochs,eta_min=1e-6); ema=ExponentialMovingAverage(student.get_params(),0.995,'cuda'); best=math.inf
        for epoch in range(a.pretrain_epochs):
            velocity_weight=a.relation_velocity_weight
            if a.velocity_ramp_epochs>0:
                velocity_weight*=min(1.0,(epoch+1)/a.velocity_ramp_epochs)
            vals=[];endpoint_vals=[];relation_vals=[];cross_noise_vals=[];induced_vals=[]
            effective_batches=len(loader) if a.max_batches == 0 else a.max_batches
            for bi,(state,noise,_action) in enumerate(loader):
                if a.max_batches > 0 and bi>=a.max_batches: break
                state=state.cuda();noise=noise.cuda()
                if ('multinoise_endpoint' in a.method or 'combined' in a.method or relation_multinoise):
                    base=max(1,len(state)//a.multi_noise);state=state[:base].repeat_interleave(a.multi_noise,0);noise=torch.randn(len(state),*noise.shape[1:],device='cuda')
                t=[.125,.375,.625,.875][(epoch*effective_batches+bi)%4] if ('balanced_multitime' in a.method or 'combined' in a.method) else float(torch.rand(()).clamp(.02,.98));tv=torch.full((len(state),),t,device='cuda')
                with torch.no_grad(): x=teacher.integrate(noise,state,start_time=0.0,end_time=t,steps=max(1,round(16*t)))
                if relation_method:
                    with torch.no_grad(): target,teacher_rel=relation_forward(teacher,x,tv,state,[0,1,2],full_minilm)
                    student_output,student_rel=relation_forward(student,x,tv,state,[0,1,2],full_minilm);velocity_loss=F.mse_loss(student_output,target);relation=relation_loss(student_rel,teacher_rel,full_minilm)
                    cross_noise=cross_noise_relation_loss(student_rel,teacher_rel,a.multi_noise) if relation_multinoise else velocity_loss*0
                else:
                    with torch.no_grad(): target=teacher.velocity(x,tv,state)
                    velocity_loss=F.mse_loss(student.velocity(x,tv,state),target);relation=velocity_loss*0;cross_noise=velocity_loss*0
                endpoint_loss=velocity_loss*0
                if ('multinoise_endpoint' in a.method or 'combined' in a.method):
                    with torch.no_grad(): teacher_endpoint=teacher.integrate(noise,state,start_time=0.0,end_time=1.0,steps=16)
                    zeros=torch.zeros(len(state),device='cuda');ones=torch.ones_like(zeros);student_endpoint=student.boundary_transition(noise,zeros,ones,state);endpoint_loss=F.mse_loss(student_endpoint,teacher_endpoint)
                if relation_method and a.relation_endpoint_weight>0:
                    with torch.no_grad(): teacher_endpoint=teacher.integrate(noise,state,start_time=0.0,end_time=1.0,steps=a.endpoint_steps)
                    student_endpoint=differentiable_integrate(student,noise,state,steps=a.endpoint_steps)
                    endpoint_loss=F.mse_loss(student_endpoint,teacher_endpoint)
                induced_loss=velocity_loss*0
                if relation_method and a.student_induced_weight>0:
                    with torch.no_grad():
                        student_x=student.integrate(noise,state,start_time=0.0,end_time=t,steps=max(1,round(16*t)))
                        induced_target=teacher.velocity(student_x,tv,state)
                    induced_loss=F.mse_loss(student.velocity(student_x,tv,state),induced_target)
                loss=(relation+a.cross_noise_weight*cross_noise+velocity_weight*velocity_loss+a.relation_endpoint_weight*endpoint_loss+a.student_induced_weight*induced_loss) if relation_method else velocity_loss+a.endpoint_weight*endpoint_loss;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.get_params(),1);opt.step();ema.update(student.get_params());vals.append(float(velocity_loss.detach()));endpoint_vals.append(float(endpoint_loss.detach()));relation_vals.append(float(relation.detach()));cross_noise_vals.append(float(cross_noise.detach()));induced_vals.append(float(induced_loss.detach()))
            sched.step();score=float(np.mean(relation_vals)+a.cross_noise_weight*np.mean(cross_noise_vals)+velocity_weight*np.mean(vals)+a.relation_endpoint_weight*np.mean(endpoint_vals)+a.student_induced_weight*np.mean(induced_vals)) if relation_method else float(np.mean(vals)+a.endpoint_weight*np.mean(endpoint_vals));prehistory.append({'epoch':epoch,'velocity_weight':velocity_weight,'velocity_loss':float(np.mean(vals)),'endpoint_loss':float(np.mean(endpoint_vals)),'student_induced_loss':float(np.mean(induced_vals)),'relation_loss':float(np.mean(relation_vals)),'cross_noise_relation_loss':float(np.mean(cross_noise_vals)),'selection_loss':score})
            if score<best:best=score;save_ema(student,ema,a.output_dir/'structure_best_flow.pth')
            if epoch+1 in milestones: save_ema(student,ema,a.output_dir/f'pretrain_epoch_{epoch+1:04d}.pth')
            if epoch%25==0: print(json.dumps({'stage':'structure','method':a.method,'epoch':epoch,'loss':score}),flush=True)
        student.load_state_dict(torch.load(a.output_dir/'structure_best_flow.pth',map_location='cuda'),strict=True)
    else: torch.save(student.state_dict(),a.output_dir/'structure_best_flow.pth')
    if a.pretrain_only:
        (a.output_dir/'pretrain_metrics.json').write_text(json.dumps({'method':a.method,'milestones':sorted(milestones),'relation_velocity_weight':a.relation_velocity_weight,'velocity_ramp_epochs':a.velocity_ramp_epochs,'mode_balanced_sampling':a.mode_balanced_sampling,'pretrain_success_only':a.pretrain_success_only,'pretrain_samples':len(dataset),'pretrain_history':prehistory,'uses_original_demonstrations':False,'uses_expert_actions':False},indent=2));return
    student.solver_steps=1;target=freeze(copy.deepcopy(student));student.train();opt=torch.optim.Adam(student.get_params(),lr=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.ctm_epochs,eta_min=1e-6);best=math.inf;best_epoch=-1;history=[];ctm_milestones={int(x) for x in a.save_ctm_epochs.split(',') if x}
    trajectory_iterator=iter(trajectory_loader) if trajectory_loader is not None else None
    x_mean=meta['x_mean'].cuda();x_std=meta['x_std'].cuda()
    y_mean=meta['y_mean'].cuda();y_std=meta['y_std'].cuda()
    for epoch in range(a.ctm_epochs):
        cv=[];dv=[];av=[];mv=[];mmdv=[];trajectory_mmdv=[]
        for bi,batch in enumerate(ctm_loader):
            if bi>=a.max_batches: break
            state,_stored_noise,action=batch[:3]
            conditional_labels=batch[3] if len(batch)==4 else None
            state=state.cuda();action=action.cuda();noise=torch.randn_like(action);ti=int(torch.randint(0,15,()).item());si=int(torch.randint(ti+2,17,()).item())
            prediction,reference,x_t,t=ctm_paths(student,target,teacher,action,state,noise,(ti,si),16);cl=pseudo_huber(prediction,reference,.01);one=torch.ones_like(t);den=student.boundary_transition(x_t,t,one,state);dl=pseudo_huber(den,action,.01);loss=cl+.1*dl
            anchor_loss=den.sum()*0.0
            if a.ctm_endpoint_anchor_weight>0:
                zeros=torch.zeros_like(t);student_endpoint=student.boundary_transition(noise,zeros,one,state)
                with torch.no_grad(): teacher_endpoint=teacher.integrate(noise,state,start_time=0.0,end_time=1.0,steps=16)
                anchor_loss=pseudo_huber(student_endpoint,teacher_endpoint,.01)
            mode_loss=den.sum()*0.0
            if a.ctm_mode_weight>0:
                k=a.ctm_conditional_samples;base=max(1,len(state)//k);mode_state=state[:base].repeat_interleave(k,0);mode_noise=torch.randn(base*k,*noise.shape[1:],device=noise.device,dtype=noise.dtype);zeros=torch.zeros(base*k,device=noise.device,dtype=noise.dtype);ones=torch.ones_like(zeros)
                student_endpoints=student.boundary_transition(mode_noise,zeros,ones,mode_state)
                with torch.no_grad(): teacher_endpoints=teacher.integrate(mode_noise,mode_state,start_time=0.0,end_time=1.0,steps=16)
                mode_loss=centered_gram_loss(student_endpoints,teacher_endpoints,k)
            mmd_loss=den.sum()*0.0
            if a.ctm_endpoint_mmd_weight>0:
                zeros=torch.zeros_like(t)
                student_endpoints=student.boundary_transition(noise,zeros,one,state)
                with torch.no_grad():
                    teacher_endpoints=teacher.integrate(noise,state,start_time=0.0,end_time=1.0,steps=16)
                mmd_loss=conditional_rbf_mmd(student_endpoints,teacher_endpoints,conditional_labels.cuda())
            trajectory_mmd_loss=den.sum()*0.0
            if trajectory_loader is not None:
                try:
                    trajectory_batch=next(trajectory_iterator)
                except StopIteration:
                    trajectory_iterator=iter(trajectory_loader)
                    trajectory_batch=next(trajectory_iterator)
                native_states,native_noises,native_endpoints,native_labels=trajectory_batch
                shape=native_noises.shape
                flat_states=native_states.flatten(0,1).cuda()
                flat_noises=native_noises.flatten(0,1).cuda()
                flat_teacher=native_endpoints.flatten(0,1).cuda()
                zeros=torch.zeros(len(flat_noises),device='cuda');ones=torch.ones_like(zeros)
                flat_student=student.boundary_transition(flat_noises,zeros,ones,flat_states)
                current_xy=flat_states[:,-1,:2]*x_std[:2]+x_mean[:2]
                student_delta=flat_student[:,-1]*y_std+y_mean
                teacher_delta=flat_teacher[:,-1]*y_std+y_mean
                student_waypoints=(current_xy+student_delta).reshape(shape[0],shape[1],2)
                teacher_waypoints=(current_xy+teacher_delta).reshape(shape[0],shape[1],2)
                trajectory_mmd_loss=conditional_rbf_mmd(
                    student_waypoints,teacher_waypoints,native_labels.cuda()
                )
            loss=cl+a.ctm_dsm_weight*dl+a.ctm_endpoint_anchor_weight*anchor_loss+a.ctm_mode_weight*mode_loss+a.ctm_endpoint_mmd_weight*mmd_loss+a.ctm_trajectory_mmd_weight*trajectory_mmd_loss
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.get_params(),1);opt.step();update_ema(target,student,.995);cv.append(float(cl.detach()));dv.append(float(dl.detach()))
            av.append(float(anchor_loss.detach()));mv.append(float(mode_loss.detach()));mmdv.append(float(mmd_loss.detach()));trajectory_mmdv.append(float(trajectory_mmd_loss.detach()))
        sched.step();score=float(np.mean(cv)+a.ctm_dsm_weight*np.mean(dv)+a.ctm_endpoint_anchor_weight*np.mean(av)+a.ctm_mode_weight*np.mean(mv)+a.ctm_endpoint_mmd_weight*np.mean(mmdv)+a.ctm_trajectory_mmd_weight*np.mean(trajectory_mmdv));rec={'epoch':epoch,'ctm_loss':float(np.mean(cv)),'pseudo_dsm_loss':float(np.mean(dv)),'endpoint_anchor_loss':float(np.mean(av)),'mode_gram_loss':float(np.mean(mv)),'endpoint_mmd_loss':float(np.mean(mmdv)),'trajectory_mmd_loss':float(np.mean(trajectory_mmdv)),'selection_loss':score};history.append(rec)
        if score<best:best=score;best_epoch=epoch;torch.save(target.state_dict(),a.output_dir/'eval_best_flow.pth')
        if epoch+1 in ctm_milestones:
            checkpoint_dir=a.output_dir/'checkpoints'/f'epoch_{epoch+1:04d}';checkpoint_dir.mkdir(parents=True,exist_ok=True);torch.save(target.state_dict(),checkpoint_dir/'eval_best_flow.pth')
        if epoch%25==0:print(json.dumps({'stage':'ctm','method':a.method,**rec}),flush=True)
    torch.save(target.state_dict(),a.output_dir/'last_flow.pth')
    uses_demo=a.ctm_data_source!='rollout'
    summary={'method':a.method,'experiment_label':'demonstration_assisted_ctm_oracle' if uses_demo else 'demonstration_free_ctm','demonstration_free':not uses_demo,'ctm_data_source':a.ctm_data_source,'ctm_balance_kind':ctm_balance_kind,'ctm_labels_for_loss_only':a.ctm_labels_for_loss_only,'ctm_balancing_protocol':'none_labels_used_only_by_auxiliary_loss' if a.ctm_labels_for_loss_only else 'uniform_latent_then_episode_then_timestep' if ctm_balance_kind!='none' else 'none','ctm_latent_labels':str(a.ctm_latent_labels) if a.ctm_latent_labels is not None else None,'ctm_success_only':a.ctm_success_only,'demonstration_mixture_probability':0.5 if a.ctm_data_source=='mixed' else 1.0 if a.ctm_data_source=='demonstration' else 0.0,'student_architecture':{'layers':layers,'embed_dim':48,'heads':heads},'multi_noise':a.multi_noise if ('multinoise_endpoint' in a.method or 'combined' in a.method) else 1,'endpoint_weight':a.endpoint_weight if ('multinoise_endpoint' in a.method or 'combined' in a.method) else 0.0,'mode_balanced_sampling':('balanced_multitime' in a.method or 'combined' in a.method),'stratified_multitime':('balanced_multitime' in a.method or 'combined' in a.method),'initialization':initialization,'uses_original_demonstrations':uses_demo,'uses_expert_actions':uses_demo,'demonstration_dir':str(a.demonstration_dir) if uses_demo else None,'buffer':str(a.buffer),'structure_dynamic_pretraining':dynamic,'pretrain_epochs':a.pretrain_epochs if dynamic else 0,'ctm_epochs':a.ctm_epochs,'ctm_dsm_weight':a.ctm_dsm_weight,'ctm_endpoint_anchor_weight':a.ctm_endpoint_anchor_weight,'ctm_mode_weight':a.ctm_mode_weight,'ctm_endpoint_mmd_weight':a.ctm_endpoint_mmd_weight,'ctm_trajectory_mmd_weight':a.ctm_trajectory_mmd_weight,'ctm_trajectory_horizon':a.ctm_trajectory_horizon,'ctm_conditional_samples':a.ctm_conditional_samples,'ctm_milestones':sorted(ctm_milestones),'best_epoch':best_epoch,'best_selection_loss':best,'pretrain_history':prehistory,'ctm_history':history}
    (a.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if not k.endswith('history')},indent=2))
if __name__=='__main__':main()
