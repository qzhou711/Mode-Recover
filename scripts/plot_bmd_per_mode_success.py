import argparse,csv,json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS={
    'Baseline':[
        'baseline_e100/standard480_seed42',
        'baseline_trainseed43/epoch_0100/standard480',
        'baseline_trainseed44/epoch_0100/standard480',
    ],
    'True-mode oracle':[
        'oracle_e100/standard480_seed42',
        'oracle_trainseed43/epoch_0100/standard480',
        'oracle_trainseed44/epoch_0100/standard480',
    ],
    'K=16 offline proxy':[
        'bmd_k16_e250/standard480_seed42',
        'bmd_k16_trainseed43/epoch_0250/standard480',
        'bmd_k16_trainseed44/epoch_0250/standard480',
    ],
}


def load(path):
    data=np.load(path/'trajectories.npz',allow_pickle=True)
    modes=data['modes']
    return data['successes'].astype(bool),modes.dot(1<<np.arange(modes.shape[1]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    teacher_success,teacher_mode=load(a.teacher)
    modes=np.unique(teacher_mode[teacher_success])
    results={};rows=[]
    for method,relative_paths in METHODS.items():
        success_rates=[];retention_rates=[]
        for seed_index,relative in enumerate(relative_paths):
            student_success,student_mode=load(a.root/relative)
            if len(student_success)!=len(teacher_success):
                raise ValueError(f'length mismatch for {relative}')
            sr=[];ret=[]
            for mode in modes:
                condition=teacher_success & (teacher_mode==mode)
                sr.append(float(student_success[condition].mean()))
                ret.append(float((student_success[condition] & (student_mode[condition]==mode)).mean()))
                rows.append({
                    'method':method,'train_seed':[42,43,44][seed_index],
                    'teacher_mode_code':int(mode),'teacher_support':int(condition.sum()),
                    'student_success_rate':sr[-1],'exact_mode_retention_rate':ret[-1],
                })
            success_rates.append(sr);retention_rates.append(ret)
        success_rates=np.asarray(success_rates);retention_rates=np.asarray(retention_rates)
        results[method]={
            'success_mean':success_rates.mean(0).tolist(),
            'success_std':success_rates.std(0,ddof=1).tolist(),
            'retention_mean':retention_rates.mean(0).tolist(),
            'retention_std':retention_rates.std(0,ddof=1).tolist(),
        }
    with (a.output_dir/'per_mode_rates.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={'conditioning':'teacher-success mode under paired episode/noise seed',
             'teacher_mode_codes':modes.tolist(),'methods':results}
    (a.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2))
    x=np.arange(len(modes));fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    for method,values in results.items():
        axes[0].errorbar(x,values['success_mean'],yerr=values['success_std'],marker='o',ms=3,capsize=2,label=method)
        axes[1].errorbar(x,values['retention_mean'],yerr=values['retention_std'],marker='o',ms=3,capsize=2,label=method)
    axes[0].set(ylabel='Student success rate',title='Per-mode success conditioned on paired Teacher mode',ylim=(-.05,1.05))
    axes[1].set(ylabel='Exact mode retention',xlabel='Teacher mode code',ylim=(-.05,1.05))
    axes[1].set_xticks(x,modes,rotation=60);axes[0].legend(ncol=3)
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(a.output_dir/'per_mode_success_and_retention.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    main()
