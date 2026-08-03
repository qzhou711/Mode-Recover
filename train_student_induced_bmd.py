"""Demonstration-free Student-induced repair with frozen BMD mode constraints."""
import argparse,json,math
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from agents.models.diffusion.ema import ExponentialMovingAverage
from agents.models.flow_matching.ctm import pseudo_huber
from teacher_flow_deployment import build_flow
from train_bmd_inference_model import InferenceModel,resample_prefix


def save_ema(model,ema,path):
    ema.store(model.get_params());ema.copy_to(model.get_params());torch.save(model.state_dict(),path);ema.restore(model.get_params())


def sliced_wasserstein(prediction,target,directions):
    p=prediction@directions.T;t=target@directions.T
    return (p.sort(dim=1).values-t.sort(dim=1).values).abs().mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--buffer',type=Path,required=True);p.add_argument('--bundle-dir',type=Path,required=True);p.add_argument('--student',type=Path,required=True);p.add_argument('--classifier',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=250);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--max-batches',type=int,default=4);p.add_argument('--mode-weight',type=float,default=0);p.add_argument('--sw-weight',type=float,default=0);p.add_argument('--failure-anchor-weight',type=float,default=.1);p.add_argument('--gate',choices=['teacher','student','conservative'],default='teacher');p.add_argument('--balance-student-latents',action='store_true');p.add_argument('--save-epochs',default='50,100,250');p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);torch.manual_seed(a.seed);np.random.seed(a.seed)
    data=torch.load(a.buffer,map_location='cpu');assert not data['metadata']['uses_original_demonstrations'];meta=torch.load(a.bundle_dir/'deployment_metadata.pt',map_location='cpu')
    model=build_flow(3,48,3,'cuda',1);model.load_state_dict(torch.load(a.student,map_location='cuda'));model.train()
    qckpt=torch.load(a.classifier,map_location='cpu');classifier=InferenceModel(qckpt['model'],24).cuda();classifier.load_state_dict(qckpt['state_dict']);classifier.eval()
    for parameter in classifier.parameters():parameter.requires_grad_(False)
    episode_ids=data['episode_ids'].long();unique_episodes=torch.unique(episode_ids);episode_to_row={int(e):i for i,e in enumerate(unique_episodes)}
    sample_rows=torch.tensor([episode_to_row[int(e)] for e in episode_ids]);teacher_success=data['teacher_successes'][sample_rows].bool();student_success=data['student_successes'][sample_rows].bool()
    if a.gate=='teacher':
        correction_mask=teacher_success;sample_weights=torch.where(teacher_success,torch.ones_like(teacher_success,dtype=torch.float),torch.full_like(teacher_success,a.failure_anchor_weight,dtype=torch.float))
    elif a.gate=='student':
        correction_mask=~student_success;sample_weights=torch.ones_like(student_success,dtype=torch.float)
    else:
        correction_mask=(~student_success)&teacher_success;sample_weights=(student_success|correction_mask).float()
    targets=torch.where(correction_mask[:,None,None],data['teacher_corrections'],data['student_endpoints'])
    sample_data=TensorDataset(data['states'].float(),data['noises'].float(),targets.float(),sample_weights)
    sampler=None
    student_latents=[]
    if a.balance_student_latents:
        with torch.no_grad():
            for start in range(0,len(data['student_paths']),128):
                features=np.stack([resample_prefix(path,1.0) for path in data['student_paths'][start:start+128]])
                features=(features-qckpt['mean'])/qckpt['std'];student_latents.extend(classifier(torch.from_numpy(features).float().cuda()).argmax(1).cpu().tolist())
        student_latents=torch.tensor(student_latents);episode_student_success=data['student_successes'].bool();counts=torch.bincount(student_latents[episode_student_success],minlength=24).float();nonempty=(counts>0).sum().clamp_min(1);balanced=torch.ones(len(unique_episodes));balanced[episode_student_success]=(episode_student_success.sum()/nonempty)/counts[student_latents[episode_student_success]].clamp_min(1);sampling_weights=balanced[sample_rows];sampler=WeightedRandomSampler(sampling_weights.double(),len(sample_data),replacement=True,generator=torch.Generator().manual_seed(a.seed))
    sample_loader=DataLoader(sample_data,batch_size=a.batch_size,shuffle=sampler is None,sampler=sampler,drop_last=True)
    # Fixed differentiable trajectory support on Teacher-success episodes only.
    e_states=[];e_noises=[];e_targets=[];e_labels=[]
    for row,eid in enumerate(unique_episodes):
        if not bool(data['teacher_successes'][row]):continue
        idx=torch.where(episode_ids==eid)[0];idx=idx[torch.argsort(data['control_steps'][idx])];chosen=idx[torch.linspace(0,len(idx)-1,32).round().long()]
        e_states.append(data['states'][chosen]);e_noises.append(data['noises'][chosen]);e_targets.append(torch.from_numpy(resample_prefix(data['teacher_paths'][row],1.0)));e_labels.append(data['teacher_latents'][row])
    episode_loader=DataLoader(TensorDataset(torch.stack(e_states).float(),torch.stack(e_noises).float(),torch.stack(e_targets).float(),torch.stack(e_labels).long()),batch_size=16,shuffle=True,drop_last=True)
    episode_iter=iter(episode_loader);x_mean=meta['x_mean'].cuda();x_std=meta['x_std'].cuda();y_mean=meta['y_mean'].cuda();y_std=meta['y_std'].cuda();q_mean=torch.as_tensor(qckpt['mean'],device='cuda');q_std=torch.as_tensor(qckpt['std'],device='cuda');angles=torch.linspace(0,math.pi,16,device='cuda');directions=torch.stack((angles.cos(),angles.sin()),1)
    opt=torch.optim.AdamW(model.get_params(),lr=1e-4,weight_decay=1e-5);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-6);ema=ExponentialMovingAverage(model.get_params(),.995,'cuda');milestones={int(x) for x in a.save_epochs.split(',')};history=[]
    for epoch in range(1,a.epochs+1):
        rv=[];mv=[];sv=[]
        for bi,(state,noise,target,weight) in enumerate(sample_loader):
            if bi>=a.max_batches:break
            state=state.cuda();noise=noise.cuda();target=target.cuda();zero=torch.zeros(len(state),device='cuda');one=torch.ones_like(zero);prediction=model.boundary_transition(noise,zero,one,state)
            per_sample=torch.sqrt((prediction-target).square()+.01**2)-.01;repair=(per_sample.flatten(1).mean(1)*weight.cuda()).sum()/weight.cuda().sum().clamp_min(1);mode_loss=repair*0;sw_loss=repair*0
            if a.mode_weight>0 or a.sw_weight>0:
                try:es,en,teacher_path,z=next(episode_iter)
                except StopIteration:episode_iter=iter(episode_loader);es,en,teacher_path,z=next(episode_iter)
                shape=en.shape;es=es.flatten(0,1).cuda();en=en.flatten(0,1).cuda();zero=torch.zeros(len(en),device='cuda');one=torch.ones_like(zero);endpoint=model.boundary_transition(en,zero,one,es);current=es[:,-1,:2]*x_std[:2]+x_mean[:2];waypoints=(current+endpoint[:,-1]*y_std+y_mean).reshape(shape[0],shape[1],2);velocity=torch.diff(waypoints,dim=1,prepend=waypoints[:,:1]);q_input=(torch.cat((waypoints,velocity),2)-q_mean)/q_std;mode_loss=F.cross_entropy(classifier(q_input),z.cuda());sw_loss=sliced_wasserstein(waypoints,teacher_path.cuda()[:,:,:2],directions)
            loss=repair+a.mode_weight*mode_loss+a.sw_weight*sw_loss;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.get_params(),1);opt.step();ema.update(model.get_params());rv.append(float(repair.detach()));mv.append(float(mode_loss.detach()));sv.append(float(sw_loss.detach()))
        sched.step();record={'epoch':epoch,'repair_loss':float(np.mean(rv)),'mode_loss':float(np.mean(mv)),'sw_loss':float(np.mean(sv))};history.append(record)
        if epoch in milestones:
            out=a.output_dir/'checkpoints'/f'epoch_{epoch:04d}';out.mkdir(parents=True,exist_ok=True);save_ema(model,ema,out/'eval_best_flow.pth')
        if epoch%25==0:print(json.dumps(record),flush=True)
    save_ema(model,ema,a.output_dir/'eval_best_flow.pth');summary={'demonstration_free':True,'uses_original_demonstrations':False,'mode_weight':a.mode_weight,'sw_weight':a.sw_weight,'failure_anchor_weight':a.failure_anchor_weight,'gate':a.gate,'balance_student_latents':a.balance_student_latents,'mask_counts':{'teacher_and_student_success':int((teacher_success&student_success).sum()),'teacher_only_success':int((teacher_success&~student_success).sum()),'student_only_success':int((~teacher_success&student_success).sum()),'both_failure':int((~teacher_success&~student_success).sum()),'correction_samples':int(correction_mask.sum()),'anchored_samples':int((~correction_mask&(sample_weights>0)).sum()),'ignored_samples':int((sample_weights==0).sum())},'student_latent_counts':torch.bincount(student_latents,minlength=24).tolist() if a.balance_student_latents else None,'epochs':a.epochs,'history':history};(a.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
