#!/usr/bin/env python3
"""Summarize only the six completed v2.2 development runs; never open tests."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mdsop_runtime_v2_2 import RUNTIME_PATH,load_runtime_config
from mdsop_training_v2_2 import experiment_output_paths,atomic_write_json,atomic_write_text


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=RUNTIME_PATH)
    args=parser.parse_args();cfg=load_runtime_config(args.config)
    fig,axes=plt.subplots(3,2,figsize=(14,12));summary=[]
    for ax,(exp,spec) in zip(axes.flat,cfg['experiments'].items()):
        paths=experiment_output_paths(cfg,exp)
        progress=json.loads(paths['progress'].read_text(encoding='utf-8'))
        if progress['status'] not in ['early_stopped','max_epochs_complete']:raise ValueError('Run not complete: '+exp)
        with paths['history'].open(encoding='utf-8',newline='') as f:history=list(csv.DictReader(f,delimiter='\t'))
        epochs=[int(r['virtual_epoch']) for r in history]
        ax.plot(epochs,[float(r['train_loss']) for r in history],label='Train SmoothL1')
        ax.plot(epochs,[float(r['selection_metric']) for r in history],label='Validation macro RMSE')
        ax.set(title=exp,xlabel='Virtual epoch',ylabel='Loss / RMSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
        summary.append({'experiment_id':exp,'status':progress['status'],'epochs':len(history),
            'best_epoch':progress['best_epoch'],'best_selection_metric':progress['best_selection_metric'],
            'best_checkpoint':str(paths['best']),'final_train_loss':float(history[-1]['train_loss'])})
    root=Path(cfg['result_root']);fig.tight_layout()
    fig.savefig(root/'six_experiment_training_curves_v2_2_20261107.png',dpi=170)
    fig.savefig(root/'six_experiment_training_curves_v2_2_20261107.svg');plt.close(fig)
    atomic_write_json(root/'step5_development_summary_v2_2_20261107.json',summary)
    atomic_write_text(root/'step5_development_summary_v2_2_20261107.tsv',
        '\t'.join(summary[0])+'\n'+'\n'.join('\t'.join(str(r[k]) for k in summary[0]) for r in summary)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
