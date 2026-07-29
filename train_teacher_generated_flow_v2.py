"""Demonstration-free structure transfer followed by Flow-CTM on teacher-generated data."""
import argparse,copy,json,math
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from agents.models.flow_matching.ctm import ctm_paths,pseudo_huber,freeze,update_ema
from agents.models.diffusion.ema import ExponentialMovingAverage
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle-dir',type=Path,required=True);p.add_argument('--buffer',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--method',choices=['activation_dynamic','pca_dynamic','early_dynamic','width_dynamic','pca_multinoise_endpoint','pca_balanced_multitime','pca_combined'],required=True);p.add_argument('--pretrain-epochs',type=int,default=300);p.add_argument('--ctm-epochs',type=int,default=500);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--max-batches',type=int,default=4);p.add_argument('--multi-noise',type=int,default=4);p.add_argument('--endpoint-weight',type=float,default=.1);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    torch.manual_seed(a.seed);np.random.seed(a.seed);a.output_dir.mkdir(parents=True,exist_ok=True)
    data=torch.load(a.buffer,map_location='cpu'); assert not data['metadata']['uses_original_demonstrations']; assert not data['metadata']['uses_expert_actions']
    states=data['states'].float(); source_noises=data['noises'].float(); pseudo_actions=data['teacher_endpoints'].float()
    dataset=TensorDataset(states,source_noises,pseudo_actions)
    sampler=None
    if a.method in {'pca_balanced_multitime','pca_combined'}:
        episode_ids=data['episode_ids'].long();mode_codes=(data['modes'].long()*(1<<torch.arange(data['modes'].shape[1]))).sum(1);sample_codes=mode_codes[episode_ids]
        _,inverse,counts=torch.unique(sample_codes,return_inverse=True,return_counts=True);weights=counts[inverse].float().reciprocal();sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=0,drop_last=True)
    teacher,_,meta=load_deployed_teacher(a.bundle_dir);layers,heads=(4,4) if a.method=='width_dynamic' else (3,3);student=build_flow(layers,48,heads,'cuda',16).train()
    acts=activation_matrix(teacher,states,source_noises,8,a.batch_size)
    init_kind='pca' if a.method.startswith('pca') else 'early' if a.method.startswith('early') else 'activation' if a.method.startswith('activation') else 'width'
    initialization=init_student(teacher,student,init_kind,acts,layers,heads); torch.save(student.state_dict(),a.output_dir/'initial_flow.pth')
    dynamic=True; prehistory=[]
    if dynamic:
        opt=torch.optim.Adam(student.get_params(),lr=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.pretrain_epochs,eta_min=1e-6); ema=ExponentialMovingAverage(student.get_params(),0.995,'cuda'); best=math.inf
        for epoch in range(a.pretrain_epochs):
            vals=[];endpoint_vals=[]
            for bi,(state,noise,_action) in enumerate(loader):
                if bi>=a.max_batches: break
                state=state.cuda();noise=noise.cuda()
                if a.method in {'pca_multinoise_endpoint','pca_combined'}:
                    base=max(1,len(state)//a.multi_noise);state=state[:base].repeat_interleave(a.multi_noise,0);noise=torch.randn(len(state),*noise.shape[1:],device='cuda')
                t=[.125,.375,.625,.875][(epoch*a.max_batches+bi)%4] if a.method in {'pca_balanced_multitime','pca_combined'} else float(torch.rand(()).clamp(.02,.98));tv=torch.full((len(state),),t,device='cuda')
                with torch.no_grad(): x=teacher.integrate(noise,state,start_time=0.0,end_time=t,steps=max(1,round(16*t)));target=teacher.velocity(x,tv,state)
                velocity_loss=F.mse_loss(student.velocity(x,tv,state),target);endpoint_loss=velocity_loss*0
                if a.method in {'pca_multinoise_endpoint','pca_combined'}:
                    with torch.no_grad(): teacher_endpoint=teacher.integrate(noise,state,start_time=0.0,end_time=1.0,steps=16)
                    zeros=torch.zeros(len(state),device='cuda');ones=torch.ones_like(zeros);student_endpoint=student.boundary_transition(noise,zeros,ones,state);endpoint_loss=F.mse_loss(student_endpoint,teacher_endpoint)
                loss=velocity_loss+a.endpoint_weight*endpoint_loss;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.get_params(),1);opt.step();ema.update(student.get_params());vals.append(float(velocity_loss.detach()));endpoint_vals.append(float(endpoint_loss.detach()))
            sched.step();score=float(np.mean(vals)+a.endpoint_weight*np.mean(endpoint_vals));prehistory.append({'epoch':epoch,'velocity_loss':float(np.mean(vals)),'endpoint_loss':float(np.mean(endpoint_vals)),'selection_loss':score})
            if score<best:best=score;save_ema(student,ema,a.output_dir/'structure_best_flow.pth')
            if epoch%25==0: print(json.dumps({'stage':'structure','method':a.method,'epoch':epoch,'loss':score}),flush=True)
        student.load_state_dict(torch.load(a.output_dir/'structure_best_flow.pth',map_location='cuda'),strict=True)
    else: torch.save(student.state_dict(),a.output_dir/'structure_best_flow.pth')
    student.solver_steps=1;target=freeze(copy.deepcopy(student));student.train();opt=torch.optim.Adam(student.get_params(),lr=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.ctm_epochs,eta_min=1e-6);best=math.inf;best_epoch=-1;history=[]
    for epoch in range(a.ctm_epochs):
        cv=[];dv=[]
        for bi,(state,_stored_noise,action) in enumerate(loader):
            if bi>=a.max_batches: break
            state=state.cuda();action=action.cuda();noise=torch.randn_like(action);ti=int(torch.randint(0,15,()).item());si=int(torch.randint(ti+2,17,()).item())
            prediction,reference,x_t,t=ctm_paths(student,target,teacher,action,state,noise,(ti,si),16);cl=pseudo_huber(prediction,reference,.01);one=torch.ones_like(t);den=student.boundary_transition(x_t,t,one,state);dl=pseudo_huber(den,action,.01);loss=cl+.1*dl
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.get_params(),1);opt.step();update_ema(target,student,.995);cv.append(float(cl.detach()));dv.append(float(dl.detach()))
        sched.step();score=float(np.mean(cv)+.1*np.mean(dv));rec={'epoch':epoch,'ctm_loss':float(np.mean(cv)),'pseudo_dsm_loss':float(np.mean(dv)),'selection_loss':score};history.append(rec)
        if score<best:best=score;best_epoch=epoch;torch.save(target.state_dict(),a.output_dir/'eval_best_flow.pth')
        if epoch%25==0:print(json.dumps({'stage':'ctm','method':a.method,**rec}),flush=True)
    torch.save(target.state_dict(),a.output_dir/'last_flow.pth')
    summary={'method':a.method,'student_architecture':{'layers':layers,'embed_dim':48,'heads':heads},'multi_noise':a.multi_noise if a.method in {'pca_multinoise_endpoint','pca_combined'} else 1,'endpoint_weight':a.endpoint_weight if a.method in {'pca_multinoise_endpoint','pca_combined'} else 0.0,'mode_balanced_sampling':a.method in {'pca_balanced_multitime','pca_combined'},'stratified_multitime':a.method in {'pca_balanced_multitime','pca_combined'},'initialization':initialization,'uses_original_demonstrations':False,'uses_expert_actions':False,'buffer':str(a.buffer),'structure_dynamic_pretraining':dynamic,'pretrain_epochs':a.pretrain_epochs if dynamic else 0,'ctm_epochs':a.ctm_epochs,'best_epoch':best_epoch,'best_selection_loss':best,'pretrain_history':prehistory,'ctm_history':history}
    (a.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if not k.endswith('history')},indent=2))
if __name__=='__main__':main()
