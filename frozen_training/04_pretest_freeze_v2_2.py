#!/usr/bin/env python3
"""Later step6/7 only: freeze six completed checkpoints without test scoring."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import torch
from mdsop_runtime_v2_2 import load_runtime_config, RUNTIME_PATH, PROJECT, RUN_TAG, sha256_file, validate_resume_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=RUNTIME_PATH)
    parser.add_argument('--experimental10',action='store_true',help='Step7 gate; requires existing step6 freeze')
    args=parser.parse_args();cfg=load_runtime_config(args.config);config_sha=sha256_file(args.config)
    step=7 if args.experimental10 else 6
    root=PROJECT/'data'/RUN_TAG/'audits'/('step7_preexperimental10' if step==7 else 'step6_pretest')
    path=root/'frozen_models_v2_2.json'
    if path.exists():raise RuntimeError('Freeze already exists; do not silently overwrite')
    if step==7:
        from mdsop_runtime_v2_2 import require_frozen_models
        require_frozen_models(PROJECT/'data'/RUN_TAG/'audits/step6_pretest/frozen_models_v2_2.json','md_test')
    experiments={}
    for exp,spec in cfg['experiments'].items():
        best=Path(cfg['model_root'])/exp/cfg['artifacts']['best_checkpoint_name']
        progress=Path(cfg['result_root'])/exp/cfg['artifacts']['progress_name']
        state=json.loads(progress.read_text(encoding='utf-8'))
        if state['status'] not in ['early_stopped','max_epochs_complete']:raise ValueError('Training incomplete: '+exp)
        saved=torch.load(best,map_location='cpu',weights_only=False)
        validate_resume_checkpoint(saved,cfg,config_sha,exp)
        if saved['epoch']!=saved['best_epoch']:raise ValueError('Not a best checkpoint')
        experiments[exp]={'checkpoint':str(best),'checkpoint_sha256':sha256_file(best),
                          'window_size':spec['window_size'],'best_epoch':saved['best_epoch']}
    result={'status':'PASS','step':step,'run_tag':RUN_TAG,'created_at':dt.datetime.now().astimezone().isoformat(),
        'runtime_config_sha256':config_sha,'seeds':[20261007],'ensemble':False,
        'allowed_endpoints':['experimental10'] if step==7 else ['md_test','soppe_test','md_test_full100ns_reference'],
        'experiments':experiments,'test_scores_computed_by_this_script':False}
    root.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(str(path))


if __name__=='__main__':main()
