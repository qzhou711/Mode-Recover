import argparse,json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_score


def resample(path,n=32):
    path=np.asarray(path,dtype=np.float64)[:,:2]
    arc=np.r_[0.0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    if arc[-1] <= 1e-12:
        return np.repeat(path[:1],n,axis=0)
    target=np.linspace(0.0,arc[-1],n)
    return np.stack([np.interp(target,arc,path[:,axis]) for axis in range(2)],axis=1)


def pair_distances(values,labels,same):
    distances=np.linalg.norm(values[:,None]-values[None,:],axis=-1)
    upper=np.triu(np.ones_like(distances,dtype=bool),1)
    mask=upper & ((labels[:,None]==labels[None,:]) if same else (labels[:,None]!=labels[None,:]))
    return distances[mask]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--buffer',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    data=torch.load(a.buffer,map_location='cpu')
    success=data['successes'].numpy().astype(bool)
    modes=data['modes'].numpy().dot(1<<np.arange(data['modes'].shape[1]))
    paths=np.asarray(data['paths'],dtype=object)[success]
    labels=modes[success]
    endpoints=np.stack([np.asarray(path)[-1,:2] for path in paths])
    trajectories=np.stack([resample(path).reshape(-1) for path in paths])
    within=pair_distances(endpoints,labels,True)
    between=pair_distances(endpoints,labels,False)
    metrics={
        'successful_trajectories':len(paths),
        'successful_modes':int(len(np.unique(labels))),
        'endpoint_within_mode_pair_distance_mean':float(within.mean()),
        'endpoint_between_mode_pair_distance_mean':float(between.mean()),
        'endpoint_between_over_within':float(between.mean()/max(within.mean(),1e-12)),
        'endpoint_mode_silhouette':float(silhouette_score(endpoints,labels)),
        'full_trajectory_mode_silhouette':float(silhouette_score(trajectories,labels)),
        'conclusion':'endpoint_is_mode_insufficient' if silhouette_score(endpoints,labels)<.1 else 'endpoint_has_mode_signal',
    }
    (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2))
    fig,ax=plt.subplots(figsize=(7,6))
    for mode in np.unique(labels):
        points=endpoints[labels==mode]
        ax.scatter(points[:,0],points[:,1],s=15,alpha=.65,label=str(mode))
    ax.set(title='Successful trajectory endpoints by mode',xlabel='x',ylabel='y')
    ax.legend(ncol=3,fontsize=6,title='mode code')
    fig.tight_layout();fig.savefig(a.output_dir/'endpoint_by_mode.png',dpi=180);plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
