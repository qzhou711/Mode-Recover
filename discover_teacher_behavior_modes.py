"""Unsupervised trajectory-mode discovery gate for Teacher-relative BMD.

This is an offline discovery proxy, not the paper's full PPO steering stage.
Ground-truth D3IL modes are used only for post-hoc evaluation.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler


def resample_path(path,n):
    path=np.asarray(path,dtype=np.float64)
    if len(path)==1:
        return np.repeat(path,n,axis=0)
    delta=np.linalg.norm(np.diff(path,axis=0),axis=1)
    arc=np.concatenate(([0.0],np.cumsum(delta)))
    if arc[-1]<=1e-12:
        return np.repeat(path[:1],n,axis=0)
    target=np.linspace(0.0,arc[-1],n)
    return np.stack([np.interp(target,arc,path[:,axis]) for axis in range(2)],axis=1)


def purity_score(labels,truth):
    total=0
    for cluster in np.unique(labels):
        values=truth[labels==cluster]
        if len(values):
            total+=np.unique(values,return_counts=True)[1].max()
    return float(total/len(labels))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--buffer',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--clusters',type=int,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resample-points',type=int,default=32)
    p.add_argument('--pca-components',type=int,default=16)
    p.add_argument('--feature-kind',choices=['basic','kinematic','shape'],default='basic')
    p.add_argument('--pca-whiten',type=int,choices=[0,1],default=1)
    p.add_argument('--clusterer',choices=['gmm_diag','gmm_full','spectral'],default='gmm_diag')
    p.add_argument('--hierarchical-success',action='store_true')
    p.add_argument('--failure-clusters',type=int,default=4)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    data=torch.load(a.buffer,map_location='cpu')
    paths=data['paths'];success=data['successes'].numpy().astype(bool)
    modes=data['modes'].numpy().astype(np.int64)
    trajectories=np.stack([resample_path(path,a.resample_points) for path in paths])
    velocity=np.diff(trajectories,axis=1,prepend=trajectories[:,:1])
    if a.feature_kind=='basic':
        parts=(trajectories,velocity)
    else:
        acceleration=np.diff(velocity,axis=1,prepend=velocity[:,:1])
        speed=np.linalg.norm(velocity,axis=-1,keepdims=True)
        tangent=velocity/np.maximum(speed,1e-8)
        curvature=np.linalg.norm(np.diff(tangent,axis=1,prepend=tangent[:,:1]),axis=-1,keepdims=True)
        if a.feature_kind=='kinematic':
            parts=(trajectories,velocity,acceleration,tangent,curvature)
        else:
            centered=trajectories-trajectories[:,:1]
            scale=np.maximum(np.linalg.norm(np.diff(trajectories,axis=1),axis=-1).sum(1)[:,None,None],1e-8)
            parts=(centered/scale,tangent,curvature)
    features=np.concatenate(tuple(part.reshape(len(paths),-1) for part in parts),axis=1)
    scaled=StandardScaler().fit_transform(features)
    n_components=min(a.pca_components,len(paths)-1,scaled.shape[1])
    embedding=PCA(n_components=n_components,whiten=bool(a.pca_whiten),random_state=a.seed).fit_transform(scaled)
    def fit_cluster(values,n_clusters):
        if a.clusterer.startswith('gmm_'):
            covariance=a.clusterer.split('_',1)[1]
            model=GaussianMixture(n_components=n_clusters,covariance_type=covariance,n_init=10,random_state=a.seed,reg_covar=1e-5)
            fitted=model.fit_predict(values)
            return fitted,float(model.bic(values))
        model=SpectralClustering(
            n_clusters=n_clusters,affinity='nearest_neighbors',
            n_neighbors=min(20,len(values)-1),assign_labels='cluster_qr',random_state=a.seed,
        )
        return model.fit_predict(values),None
    if a.hierarchical_success:
        labels=np.full(len(paths),-1,dtype=np.int64)
        labels[success],bic=fit_cluster(embedding[success],a.clusters)
        failure_labels=np.full(len(paths),-1,dtype=np.int64)
        failure_labels[~success],failure_bic=fit_cluster(embedding[~success],a.failure_clusters)
        metric_embedding=embedding[success]
        metric_labels=labels[success]
    elif a.clusterer.startswith('gmm_'):
        covariance=a.clusterer.split('_',1)[1]
        cluster_model=GaussianMixture(n_components=a.clusters,covariance_type=covariance,n_init=10,random_state=a.seed,reg_covar=1e-5)
        labels=cluster_model.fit_predict(embedding)
        bic=float(cluster_model.bic(embedding))
        failure_labels=np.full(len(paths),-1,dtype=np.int64);failure_bic=None
        metric_embedding=embedding;metric_labels=labels
    else:
        cluster_model=SpectralClustering(
            n_clusters=a.clusters,affinity='nearest_neighbors',n_neighbors=20,
            assign_labels='cluster_qr',random_state=a.seed,
        )
        labels=cluster_model.fit_predict(embedding)
        bic=None
        failure_labels=np.full(len(paths),-1,dtype=np.int64);failure_bic=None
        metric_embedding=embedding;metric_labels=labels
    truth=modes.dot(1<<np.arange(modes.shape[1]))
    occupancy=np.bincount(labels[labels>=0],minlength=a.clusters)
    cluster_success=[float(success[labels==z].mean()) if np.any(labels==z) else 0.0 for z in range(a.clusters)]
    cluster_success_count=[int(success[labels==z].sum()) for z in range(a.clusters)]
    cluster_true_modes=[int(len(np.unique(truth[(labels==z)&success]))) for z in range(a.clusters)]
    metrics={
        'method':'offline_trajectory_bmd_proxy',
        'full_paper_ppo_steering':False,
        'uses_original_demonstrations':False,
        'uses_ground_truth_modes_for_training':False,
        'ground_truth_modes_used_for_posthoc_evaluation':True,
        'feature_kind':a.feature_kind,'pca_whiten':bool(a.pca_whiten),'clusterer':a.clusterer,
        'hierarchical_success':a.hierarchical_success,
        'uses_binary_success_feedback_for_partition':a.hierarchical_success,
        'failure_clusters':a.failure_clusters if a.hierarchical_success else 0,
        'clusters':a.clusters,'seed':a.seed,'episodes':len(paths),
        'successful_episodes':int(success.sum()),
        'silhouette':float(silhouette_score(metric_embedding,metric_labels)),
        'gmm_bic':bic,
        'failure_gmm_bic':failure_bic,
        'nmi_all':float(normalized_mutual_info_score(truth,labels)),
        'ari_all':float(adjusted_rand_score(truth,labels)),
        'nmi_success':float(normalized_mutual_info_score(truth[success],labels[success])),
        'ari_success':float(adjusted_rand_score(truth[success],labels[success])),
        'purity_success':purity_score(labels[success],truth[success]),
        'occupied_latents':int((occupancy>0).sum()),
        'min_latent_size':int(occupancy.min()),
        'max_latent_size':int(occupancy.max()),
        'latent_occupancy':occupancy.tolist(),
        'latent_success_rate':cluster_success,
        'latent_success_count':cluster_success_count,
        'latent_true_mode_coverage':cluster_true_modes,
    }
    sample_episode_ids=data['episode_ids'].numpy()
    episode_ids=np.unique(sample_episode_ids)
    if len(episode_ids)!=len(paths):
        raise ValueError(f'{len(paths)} paths but {len(episode_ids)} unique episode ids')
    episode_to_index={int(episode_id):index for index,episode_id in enumerate(episode_ids)}
    sample_labels=np.asarray([labels[episode_to_index[int(episode_id)]] for episode_id in sample_episode_ids])
    np.savez_compressed(a.output_dir/'discovery.npz',trajectory_features=features,embedding=embedding,
                        episode_latents=labels,sample_latents=sample_labels,successes=success,
                        episode_failure_latents=failure_labels,ground_truth_modes=modes)
    (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
