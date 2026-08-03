import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--min-latent-size',type=int,default=5)
p.add_argument('--min-successful-per-latent',type=int,default=1)
a=p.parse_args()
rows=[]
for path in a.root.glob('*seed42/metrics.json'):
    metric=json.loads(path.read_text())
    metric['directory']=str(path.parent)
    success_counts=metric.get('latent_success_count')
    if success_counts is None:
        success_counts=[
            int(round(size*rate))
            for size,rate in zip(metric['latent_occupancy'],metric['latent_success_rate'])
        ]
    metric['selection_min_successful_per_latent']=min(success_counts)
    if (metric['occupied_latents']==metric['clusters']
            and metric['min_latent_size']>=a.min_latent_size
            and min(success_counts)>=a.min_successful_per_latent):
        rows.append(metric)
if not rows:
    failure={
        'selection_protocol':'unsupervised_silhouette_then_negative_bic',
        'ground_truth_metrics_used_for_selection':False,
        'min_latent_size':a.min_latent_size,
        'min_successful_per_latent':a.min_successful_per_latent,
        'status':'no_candidate_passed',
        'best':None,
        'alternative':None,
        'candidates':[],
    }
    (a.root/'selection.json').write_text(json.dumps(failure,indent=2))
    raise SystemExit('no discovery candidate passed unsupervised occupancy/success gate')
# Selection is strictly unsupervised: ground-truth NMI/ARI/purity are report-only.
# Silhouette is primary; lower GMM BIC breaks ties without using environment modes.
rows.sort(key=lambda x:(x['silhouette'],-x['gmm_bic'] if x['gmm_bic'] is not None else float('-inf')),reverse=True)
result={
    'selection_protocol':'unsupervised_silhouette_then_negative_bic',
    'ground_truth_metrics_used_for_selection':False,
    'min_latent_size':a.min_latent_size,
    'min_successful_per_latent':a.min_successful_per_latent,
    'best':rows[0],
    'alternative':rows[1] if len(rows)>1 else rows[0],
    'candidates':rows,
}
(a.root/'selection.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
