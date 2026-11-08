#!/usr/bin/env python3
"""Explicit endpoint evaluator. Never retrains or selects a checkpoint from tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mdsop_runtime_v2_2 import RUNTIME_PATH,require_frozen_models,runtime_identity,sha256_file
from mdsop_evaluation_v2_2 import (EvaluationAccessPolicy,EvaluationProteinDataset,STEP6_ENDPOINTS,
    make_evaluation_loader,load_frozen_checkpoint,record_metrics,summarize_predictions,
    atomic_write_json,atomic_write_tsv,atomic_write_gzip_tsv)
from mdsop_experimental10_v2_2 import Experimental10AccessPolicy,Experimental10Dataset

EXP_CHAINS={'1pd7':'A','1wrs':'R','1wrt':'R','1z9b':'A','2jwt':'A','2l6b':'A','2luo':'A','2m3o':'W','2xdi':'A','4aai':'A'}


def predict_records(model,loader,device,identity):
    """Shared by the guarded CLI and synthetic-only acceptance tests."""
    model.eval();per_record=[];residues=[];pp=[];yy=[]
    with torch.no_grad():
        for batch in loader:
            pred=model(batch['x'].to(device),batch['lengths'],batch['residue_mask'].to(device)).cpu().numpy()
            for i,rid in enumerate(batch['record_ids']):
                n=int(batch['lengths'][i]);p=pred[i,:n];y=batch['y'][i,:n].numpy();q=batch['qc_mask'][i,:n].numpy()
                metadata={**identity,'record_id':rid,'protein_id':batch['protein_ids'][i],
                    'sequence_sha256':batch['sequence_sha256s'][i],'joint_homology_group':batch['joint_homology_groups'][i]}
                per_record.append(record_metrics(prediction=p,target=y,qc_mask=q,metadata=metadata));pp.append(p[q]);yy.append(y[q])
                for j in range(n):residues.append({**metadata,'residue_index_1based':j+1,
                    'amino_acid':batch['sequences'][i][j],'prediction':float(p[j]),'target':float(y[j]),'qc_eligible':bool(q[j])})
    return per_record,residues,summarize_predictions(per_record,pp,yy)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=RUNTIME_PATH)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--endpoint',required=True,choices=list(STEP6_ENDPOINTS)+['experimental10'])
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();cfg,freeze=require_frozen_models(args.freeze,args.endpoint)
    if args.config.resolve()!=RUNTIME_PATH:raise ValueError('Only the accepted runtime config is allowed')
    import yaml
    binding=yaml.safe_load(Path(cfg['runtime_binding']['data_binding_path']).read_text(encoding='utf-8'))
    white_path=Path(binding['evaluator_whitelist']);white=json.loads(white_path.read_text(encoding='utf-8'))
    entry=white['endpoints'][args.endpoint];eval_root=Path(cfg['data_access']['evaluation_root'])
    if args.endpoint=='experimental10':
        dataset=Experimental10Dataset(Experimental10AccessPolicy(white_path,binding['evaluator_whitelist_sha256'],eval_root,args.freeze),
            manifest_path=Path(entry['record_manifest']),manifest_sha256=entry['record_manifest_sha256'],
            expected_ids=tuple(EXP_CHAINS),expected_chains=EXP_CHAINS)
    else:
        dataset=EvaluationProteinDataset(EvaluationAccessPolicy(white_path,binding['evaluator_whitelist_sha256'],eval_root,STEP6_ENDPOINTS,args.freeze),
            endpoint=args.endpoint,manifest_path=Path(entry['record_manifest']),manifest_sha256=entry['record_manifest_sha256'],
            expected_source='soppe_legacy' if args.endpoint=='soppe_test' else 'md_ired')
    loader=make_evaluation_loader(dataset,batch_size=10 if args.endpoint=='experimental10' else 8,num_workers=0)
    device=torch.device(args.device)
    for exp,rec in freeze['experiments'].items():
        model,meta=load_frozen_checkpoint(experiment_id=exp,checkpoint_path=Path(rec['checkpoint']),
            checkpoint_sha256=rec['checkpoint_sha256'],expected_window_size=int(rec['window_size']),
            expected_training_config_sha256=freeze['runtime_config_sha256'],device=device)
        identity={**runtime_identity(cfg),'experiment_id':exp,'endpoint':args.endpoint,
            'target_label_version':entry['label_version'],'runtime_config_sha256':sha256_file(RUNTIME_PATH),
            'checkpoint_file_sha256':rec['checkpoint_sha256'],
            'evaluation_manifest_sha256':entry['record_manifest_sha256'],
            'evaluator_whitelist_sha256':binding['evaluator_whitelist_sha256']}
        per_record,residues,summary=predict_records(model,loader,device,identity)
        summary.update({**identity,
            'checkpoint':meta,'prediction_transform':'none'})
        result_endpoint='md_test_5tauc' if args.endpoint=='md_test' else args.endpoint
        root=Path(cfg['project_root'])/'results'/cfg['run_tag']/result_endpoint/exp
        if (root/'summary.json').exists():raise RuntimeError('Evaluation output already exists; do not overwrite frozen results')
        root.mkdir(parents=True,exist_ok=True)
        atomic_write_json(root/'summary.json',summary)
        atomic_write_tsv(root/'per_protein_metrics.tsv',per_record,list(per_record[0]))
        atomic_write_gzip_tsv(root/'per_residue_predictions.tsv.gz',residues,list(residues[0]))
        print(exp,args.endpoint,summary['protein_record_macro'])


if __name__=='__main__':main()
