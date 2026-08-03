import argparse,json
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--min-size',type=int,default=5)
    p.add_argument('--min-success',type=int,default=1)
    p.add_argument('--min-stability-nmi',type=float,default=.6)
    a=p.parse_args()
    rows=[]
    for seed42_dir in sorted(a.root.glob('*_seed42')):
        seed43_dir=Path(str(seed42_dir).replace('_seed42','_seed43'))
        if not seed43_dir.exists():
            continue
        m42=json.loads((seed42_dir/'metrics.json').read_text())
        m43=json.loads((seed43_dir/'metrics.json').read_text())
        z42=np.load(seed42_dir/'discovery.npz')['episode_latents']
        z43=np.load(seed43_dir/'discovery.npz')['episode_latents']
        stability=float(normalized_mutual_info_score(z42,z43))
        min_success42=min(m42['latent_success_count'])
        min_success43=min(m43['latent_success_count'])
        passed=(
            m42['occupied_latents']==m42['clusters']
            and m43['occupied_latents']==m43['clusters']
            and min(m42['min_latent_size'],m43['min_latent_size'])>=a.min_size
            and min(min_success42,min_success43)>=a.min_success
            and stability>=a.min_stability_nmi
        )
        rows.append({
            'name':seed42_dir.name.removesuffix('_seed42'),
            'seed42_directory':str(seed42_dir),
            'seed43_directory':str(seed43_dir),
            'feature_kind':m42['feature_kind'],
            'pca_whiten':m42['pca_whiten'],
            'clusterer':m42['clusterer'],
            'silhouette_mean':(m42['silhouette']+m43['silhouette'])/2,
            'cross_seed_nmi':stability,
            'min_latent_size':min(m42['min_latent_size'],m43['min_latent_size']),
            'min_successful_per_latent':min(min_success42,min_success43),
            'passed':passed,
        })
    eligible=[row for row in rows if row['passed']]
    eligible.sort(key=lambda row:(row['cross_seed_nmi'],row['silhouette_mean']),reverse=True)
    result={
        'selection_uses_ground_truth_modes':False,
        'gates':{'min_size':a.min_size,'min_success':a.min_success,'min_stability_nmi':a.min_stability_nmi},
        'best':eligible[0] if eligible else None,
        'rows':rows,
    }
    (a.root/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
