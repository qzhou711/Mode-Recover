"""Trajectory Consistency Distillation (CTM) for Avoiding Flow policies."""
import argparse, copy, json, math
from pathlib import Path
import numpy as np
import torch
from tqdm import trange
import wandb
from distill_flow_matching_avoiding import make_agent, make_student, repeat_conditions
from agents.models.flow_matching.ctm import (
    conditional_distance_loss, ctm_paths, pseudo_huber, freeze, update_ema,
)


def initialize_student(student, teacher, init_kind, init_dir):
    if init_kind == "random":
        return "random"
    if init_kind == "teacher":
        student.load_state_dict(teacher.state_dict(), strict=True)
        return "teacher"
    if init_dir is None:
        raise ValueError("checkpoint initialization requires --init-dir")
    checkpoint = init_dir/'eval_best_flow.pth'
    student.load_state_dict(torch.load(checkpoint, map_location=student.device), strict=True)
    return str(checkpoint)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--teacher-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--student-init',choices=['random','teacher','checkpoint'],default='teacher')
    parser.add_argument('--init-dir',type=Path)
    parser.add_argument('--student-layers',type=int,default=4)
    parser.add_argument('--student-embed-dim',type=int,default=72)
    parser.add_argument('--student-heads',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--max-batches-per-epoch',type=int,default=4)
    parser.add_argument('--teacher-steps',type=int,default=16)
    parser.add_argument('--teacher-layers',type=int,default=4)
    parser.add_argument('--teacher-embed-dim',type=int,default=72)
    parser.add_argument('--teacher-heads',type=int,default=4)
    parser.add_argument('--time-bins',type=int,default=16)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--ema-decay',type=float,default=0.995)
    parser.add_argument('--dsm-weight',type=float,default=0.1)
    parser.add_argument('--endpoint-probability',type=float,default=0.0)
    parser.add_argument('--endpoint-anchor-weight',type=float,default=0.0)
    parser.add_argument('--distribution-weight',type=float,default=0.0)
    parser.add_argument('--conditional-samples',type=int,default=1)
    parser.add_argument('--delta',type=float,default=0.01)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.time_bins < 2: parser.error('--time-bins must be at least 2')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    wandb.init(mode='disabled')
    agent=make_agent(args.teacher_dir,args.batch_size,args.teacher_steps, args.teacher_layers,args.teacher_embed_dim,args.teacher_heads)
    teacher=freeze(agent.model)
    student=make_student(agent,1,args.student_layers,args.student_embed_dim,args.student_heads,'random').to(agent.device)
    initialization=initialize_student(student,teacher,args.student_init,args.init_dir)
    target=freeze(copy.deepcopy(student)); student.train()
    optimizer=torch.optim.Adam(student.get_params(),lr=args.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    best,best_epoch,history=math.inf,-1,[]
    for epoch in trange(args.epochs,desc='Flow CTM'):
        epoch_ctm,epoch_dsm,epoch_anchor,epoch_distribution=[],[],[],[]
        for batch_index,(state,action,_) in enumerate(agent.train_dataloader):
            if args.max_batches_per_epoch and batch_index>=args.max_batches_per_epoch: break
            state=agent.scaler.scale_input(state).float(); action=agent.scaler.scale_output(action).float()
            state,action=repeat_conditions(state,action,args.conditional_samples)
            noise=torch.randn_like(action)
            if torch.rand(()).item() < args.endpoint_probability:
                ti,si=0,args.time_bins
            else:
                ti=int(torch.randint(0,args.time_bins-1,()).item())
                si=int(torch.randint(ti+2,args.time_bins+1,()).item())
            prediction,reference,x_t,t=ctm_paths(student,target,teacher,action,state,noise,(ti,si),args.time_bins)
            ctm_loss=pseudo_huber(prediction,reference,args.delta)
            one=torch.ones_like(t)
            denoised=student.boundary_transition(x_t,t,one,state)
            dsm_loss=pseudo_huber(denoised,action,args.delta)
            anchor_loss=denoised.sum()*0.0
            distribution_loss=denoised.sum()*0.0
            if args.endpoint_anchor_weight>0 or args.distribution_weight>0:
                zeros=torch.zeros_like(t)
                student_endpoint=student.boundary_transition(noise,zeros,one,state)
                with torch.no_grad():
                    teacher_endpoint=teacher.integrate(noise,state,start_time=0.0,end_time=1.0,steps=args.teacher_steps)
                anchor_loss=pseudo_huber(student_endpoint,teacher_endpoint,args.delta)
                distribution_loss=conditional_distance_loss(student_endpoint,teacher_endpoint,args.conditional_samples)
            loss=(ctm_loss+args.dsm_weight*dsm_loss
                  +args.endpoint_anchor_weight*anchor_loss
                  +args.distribution_weight*distribution_loss)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.get_params(),1.0)
            optimizer.step(); update_ema(target,student,args.ema_decay)
            epoch_ctm.append(float(ctm_loss.detach())); epoch_dsm.append(float(dsm_loss.detach()))
            epoch_anchor.append(float(anchor_loss.detach())); epoch_distribution.append(float(distribution_loss.detach()))
        scheduler.step()
        score=float(np.mean(epoch_ctm)+args.dsm_weight*np.mean(epoch_dsm)+args.endpoint_anchor_weight*np.mean(epoch_anchor)+args.distribution_weight*np.mean(epoch_distribution))
        record={'epoch':epoch,'ctm_loss':float(np.mean(epoch_ctm)),'dsm_loss':float(np.mean(epoch_dsm)),'endpoint_anchor_loss':float(np.mean(epoch_anchor)),'distribution_loss':float(np.mean(epoch_distribution)),'selection_loss':score,'learning_rate':optimizer.param_groups[0]['lr']}
        history.append(record)
        if score<best:
            best,best_epoch=score,epoch; torch.save(target.state_dict(),args.output_dir/'eval_best_flow.pth')
        if epoch%10==0 or epoch+1==args.epochs: print(json.dumps(record),flush=True)
    torch.save(target.state_dict(),args.output_dir/'last_flow.pth')
    summary={'method':'flow_boundary_ctm','time_orientation':'noise_0_to_data_1','teacher_checkpoint':str(args.teacher_dir/'eval_best_flow.pth'),'student_architecture':{'layers':args.student_layers,'embed_dim':args.student_embed_dim,'heads':args.student_heads},'initialization':initialization,'epochs':args.epochs,'time_bins':args.time_bins,'dsm_weight':args.dsm_weight,'endpoint_probability':args.endpoint_probability,'endpoint_anchor_weight':args.endpoint_anchor_weight,'distribution_weight':args.distribution_weight,'conditional_samples':args.conditional_samples,'best_epoch':best_epoch,'best_selection_loss':best,'history':history}
    (args.output_dir/'ctm_metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='history'},indent=2)); wandb.finish()


if __name__=='__main__': main()
