"""CUDA-event latency benchmark for deployed Flow policy generation."""
import argparse, json
from pathlib import Path
import torch
from teacher_flow_deployment import build_flow

p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--layers',type=int,default=3);p.add_argument('--embed-dim',type=int,default=72);p.add_argument('--heads',type=int,default=4);p.add_argument('--steps',type=int,default=16);p.add_argument('--inference-mode',choices=['integrate','boundary'],default='integrate');p.add_argument('--warmup',type=int,default=100);p.add_argument('--repeats',type=int,default=1000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
model=build_flow(a.layers,a.embed_dim,a.heads,'cuda',a.steps);model.load_state_dict(torch.load(a.model_dir/'eval_best_flow.pth',map_location='cuda'),strict=True);model.eval();state=torch.randn(1,5,4,device='cuda');noise=torch.randn(1,5,2,device='cuda');zero=torch.zeros(1,device='cuda');one=torch.ones_like(zero)
def run():
 with torch.no_grad():
  return model.boundary_transition(noise,zero,one,state) if a.inference_mode=='boundary' else model.sample(state,initial_noise=noise,steps=a.steps)
for _ in range(a.warmup):run()
torch.cuda.synchronize();times=[]
for _ in range(a.repeats):
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();run();end.record();torch.cuda.synchronize();times.append(start.elapsed_time(end))
t=torch.tensor(times);result={'model_dir':str(a.model_dir),'steps':a.steps,'inference_mode':a.inference_mode,'batch_size':1,'warmup':a.warmup,'repeats':a.repeats,'mean_ms':float(t.mean()),'median_ms':float(t.median()),'p95_ms':float(torch.quantile(t,.95))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
