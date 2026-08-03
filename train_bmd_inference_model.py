"""Train a frozen BMD-inspired q(z|trajectory prefix) without true mode labels."""
import argparse,json,random
from pathlib import Path

import numpy as np,torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler


def resample_prefix(path,fraction,n=32):
    path=np.asarray(path,dtype=np.float32)[:,:2]
    end=max(2,int(np.ceil(len(path)*fraction)));path=path[:end]
    arc=np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    if arc[-1]<=1e-8: position=np.repeat(path[:1],n,axis=0)
    else:
        target=np.linspace(0,arc[-1],n)
        position=np.stack([np.interp(target,arc,path[:,axis]) for axis in range(2)],1)
    velocity=np.diff(position,axis=0,prepend=position[:1])
    return np.concatenate((position,velocity),axis=1).astype(np.float32)


class PrefixDataset(Dataset):
    def __init__(self,paths,labels,indices,mean,std,fixed_fraction=None):
        self.paths=paths;self.labels=labels;self.indices=np.asarray(indices)
        self.mean=mean;self.std=std;self.fixed_fraction=fixed_fraction
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        episode=self.indices[index]
        fraction=self.fixed_fraction if self.fixed_fraction is not None else random.choice((.25,.5,.75,1.0))
        x=(resample_prefix(self.paths[episode],fraction)-self.mean)/self.std
        return torch.from_numpy(x),torch.tensor(self.labels[episode]).long()


class InferenceModel(nn.Module):
    def __init__(self,kind,classes,hidden=128):
        super().__init__();self.kind=kind
        if kind=='gru': self.encoder=nn.GRU(4,hidden,2,batch_first=True,dropout=.1)
        else:
            self.input=nn.Linear(4,hidden);self.position=nn.Parameter(torch.randn(1,32,hidden)*.02)
            layer=nn.TransformerEncoderLayer(hidden,4,hidden*4,.1,batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,3,norm=nn.LayerNorm(hidden))
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,classes))
    def forward(self,x):
        if self.kind=='gru': _,h=self.encoder(x);feature=h[-1]
        else: feature=self.encoder(self.input(x)+self.position).mean(1)
        return self.head(feature)


@torch.no_grad()
def evaluate(model,paths,labels,indices,mean,std,batch_size):
    result={};model.eval()
    for fraction in (.25,.5,.75,1.0):
        loader=DataLoader(PrefixDataset(paths,labels,indices,mean,std,fraction),batch_size=batch_size)
        truth=[];prediction=[]
        for x,y in loader:
            prediction.extend(model(x.cuda()).argmax(1).cpu().tolist());truth.extend(y.tolist())
        result[str(fraction)]={
            'macro_f1':float(f1_score(truth,prediction,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(truth,prediction)),
            'per_class_recall':recall_score(truth,prediction,labels=list(range(24)),average=None,zero_division=0).tolist(),
        }
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--buffer',type=Path,required=True);p.add_argument('--discovery',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--model',choices=['gru','transformer'],required=True);p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    data=torch.load(a.buffer,map_location='cpu');discovery=np.load(a.discovery)
    labels=discovery['episode_latents'].astype(np.int64);paths=np.asarray(data['paths'],dtype=object)
    eligible=np.where(labels>=0)[0];train,temp=train_test_split(eligible,test_size=.3,random_state=a.seed,stratify=labels[eligible]);valid,test=train_test_split(temp,test_size=.5,random_state=a.seed,stratify=labels[temp])
    full=np.stack([resample_prefix(paths[index],1.0) for index in train]);mean=full.mean((0,1),keepdims=False);std=full.std((0,1),keepdims=False).clip(1e-6)
    train_set=PrefixDataset(paths,labels,train,mean,std);counts=np.bincount(labels[train],minlength=24);weights=1.0/counts[labels[train]];sampler=WeightedRandomSampler(torch.tensor(weights).double(),len(train),replacement=True,generator=torch.Generator().manual_seed(a.seed))
    loader=DataLoader(train_set,batch_size=a.batch_size,sampler=sampler,num_workers=0,drop_last=True)
    model=InferenceModel(a.model,24).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-6);best=-1;best_epoch=-1;history=[]
    for epoch in range(a.epochs):
        model.train();losses=[]
        for x,y in loader:
            loss=nn.functional.cross_entropy(model(x.cuda()),y.cuda());opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        sched.step()
        if epoch%10==9 or epoch==a.epochs-1:
            metrics=evaluate(model,paths,labels,valid,mean,std,a.batch_size);score=metrics['1.0']['macro_f1']+metrics['0.5']['macro_f1'];history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':metrics,'selection_score':score});print(json.dumps(history[-1]),flush=True)
            if score>best:best=score;best_epoch=epoch+1;torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std,'model':a.model},a.output_dir/'best.pt')
    checkpoint=torch.load(a.output_dir/'best.pt',map_location='cuda');model.load_state_dict(checkpoint['state_dict']);test_metrics=evaluate(model,paths,labels,test,mean,std,a.batch_size)
    summary={'model':a.model,'seed':a.seed,'labels_source':str(a.discovery),'uses_true_modes':False,'uses_binary_success_partition':True,'uses_original_demonstrations':False,'episodes':{'train':len(train),'validation':len(valid),'test':len(test)},'best_epoch':best_epoch,'validation_selection_score':best,'test':test_metrics,'history':history}
    (a.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='history'},indent=2))


if __name__=='__main__':main()
