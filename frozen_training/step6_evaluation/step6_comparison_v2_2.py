"""Frozen paired rescoring: no fitting, prediction transforms, or model selection."""
from __future__ import annotations
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

CODE=Path('/data/jiangbin/SOPPCL/MD-SOP/code/v2_2_20261107')
sys.path.insert(0,str(CODE))
from mdsop_evaluation_v2_2 import (METRIC_NAMES,record_metrics,summarize_predictions,evaluation_input_digest,
    atomic_write_json,atomic_write_tsv,atomic_write_gzip_tsv)


def read_tsv(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))


def prediction_digest(predictions):
    h=hashlib.sha256()
    for rid,p in sorted(predictions.items()):
        h.update(rid.encode());h.update(np.asarray(p,dtype='<f4').tobytes())
    return h.hexdigest()


def parse_predictions(path,records,experiment_id,endpoint,expected_version=None,expected_checkpoint=None):
    grouped=defaultdict(list)
    for row in read_tsv(path):
        assert row['experiment_id']==experiment_id and row['endpoint']==endpoint,'Prediction model/endpoint mismatch'
        if expected_version is not None:
            assert row['model_version']==expected_version and row['run_tag']=='v2_2_20261107'
            assert row['checkpoint_file_sha256']==expected_checkpoint
        grouped[row['record_id']].append(row)
    assert set(grouped)=={r['record_id'] for r in records},'Prediction record set mismatch'
    result={}
    for record in records:
        rows=sorted(grouped[record['record_id']],key=lambda r:int(r['residue_index_1based']))
        assert [int(r['residue_index_1based']) for r in rows]==list(range(1,len(record['target'])+1)),'Missing/duplicate positions'
        assert all(r['sequence_sha256']==record['sequence_sha256'] for r in rows),'Prediction sequence hash mismatch'
        assert ''.join(r['amino_acid'] for r in rows)==record['sequence'],'Prediction residue mapping mismatch'
        mask=np.asarray([r['qc_eligible']=='True' for r in rows]);assert np.array_equal(mask,record['qc_mask']),'Prediction mask changed'
        truth=np.asarray([float(r['target']) if r['target'] else 0. for r in rows],dtype=np.float32)
        assert np.array_equal(truth[mask],record['target'][mask]),'Stored prediction targets differ from release'
        p=np.asarray([float(r['prediction']) for r in rows],dtype=np.float64)
        assert np.isfinite(p).all();result[record['record_id']]=p
    return result


def score_predictions(predictions,records,identity):
    per_record=[];pp=[];yy=[]
    for r in records:
        p=predictions[r['record_id']];q=r['qc_mask'];y=r['target']
        meta={**identity,**{k:r[k] for k in ['record_id','protein_id','sequence_sha256','joint_homology_group']}}
        per_record.append(record_metrics(prediction=p,target=y,qc_mask=q,metadata=meta));pp.append(p[q]);yy.append(y[q])
    summary=summarize_predictions(per_record,pp,yy)
    summary.update({**identity,'prediction_sha256':prediction_digest(predictions),'evaluation_input_sha256':evaluation_input_digest(records),
        'prediction_transform':'none'})
    return per_record,summary


def write_scored_endpoint(root,predictions,records,identity):
    root=Path(root);assert not root.exists(),'Do not overwrite endpoint '+str(root)
    per_record,summary=score_predictions(predictions,records,identity)
    def residues():
        for r in records:
            for j,p in enumerate(predictions[r['record_id']]):
                valid=bool(r['qc_mask'][j])
                yield {**identity,'record_id':r['record_id'],'protein_id':r['protein_id'],
                    'sequence_sha256':r['sequence_sha256'],'joint_homology_group':r['joint_homology_group'],
                    'residue_index_1based':j+1,'amino_acid':r['sequence'][j],'prediction':float(p),
                    'target':float(r['target'][j]) if valid else None,'qc_eligible':valid}
    root.mkdir(parents=True)
    atomic_write_json(root/'summary.json',summary)
    atomic_write_tsv(root/'per_protein_metrics.tsv',per_record,list(per_record[0]))
    first=next(residues());atomic_write_gzip_tsv(root/'per_residue_predictions.tsv.gz',residues(),list(first))
    return per_record,summary


def holm_adjust(values):
    adjusted=np.empty(len(values));running=0.
    for rank,index in enumerate(np.argsort(np.asarray(values))):
        running=max(running,min(1.,(len(values)-rank)*float(values[index])));adjusted[index]=running
    return adjusted.tolist()


def sign_flip_p(values,replicates,seed,exact_max_groups=16):
    values=np.asarray(values,dtype=np.float64);observed=abs(float(values.mean()))
    if len(values)<=exact_max_groups:
        signs=np.asarray(list(itertools.product([-1.,1.],repeat=len(values))))
        exceed=int(np.count_nonzero(np.abs((signs*values).mean(1))>=observed-1e-15))
        return exceed/len(signs),'exact',len(signs)
    rng=np.random.default_rng(seed);signs=rng.integers(0,2,size=(replicates,len(values)),dtype=np.int8)*2-1
    exceed=int(np.count_nonzero(np.abs((signs*values).mean(1))>=observed-1e-15))
    return (exceed+1)/(replicates+1),'monte_carlo_plus_one',replicates


def grouped_pair_statistics(rows,metric,replicates,seed):
    grouped=defaultdict(list);valid=[]
    for r in rows:
        a=float(r['old_'+metric]);b=float(r['new_'+metric])
        if math.isfinite(a) and math.isfinite(b):
            d=b-a;valid.append(d);grouped[r['joint_homology_group']].append(d)
    if not valid:return {'valid_record_pairs':0,'valid_homology_groups':0,'record_macro_delta':None,'group_balanced_delta':None}
    arrays=[np.asarray(v) for _,v in sorted(grouped.items())]
    totals=np.asarray([v.sum() for v in arrays]);counts=np.asarray([len(v) for v in arrays]);means=totals/counts
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(arrays),size=(replicates,len(arrays)))
    record_estimates=totals[indices].sum(1)/counts[indices].sum(1)
    group_estimates=means[indices].mean(1)
    rl,rh=np.quantile(record_estimates,[.025,.975]);gl,gh=np.quantile(group_estimates,[.025,.975])
    return {'valid_record_pairs':len(valid),'valid_homology_groups':len(arrays),'record_macro_delta':float(np.mean(valid)),
        'group_balanced_delta':float(means.mean()),'record_macro_group_bootstrap95_low':float(rl),
        'record_macro_group_bootstrap95_high':float(rh),'group_balanced_bootstrap95_low':float(gl),
        'group_balanced_bootstrap95_high':float(gh),'group_values':means}


def paired_records(old,new,experiment,endpoint):
    old={r['record_id']:r for r in old};new={r['record_id']:r for r in new};assert set(old)==set(new)
    output=[]
    for rid in sorted(old):
        a=old[rid];b=new[rid]
        assert a['sequence_sha256']==b['sequence_sha256'] and a['joint_homology_group']==b['joint_homology_group']
        assert int(a['eligible_residues'])==int(b['eligible_residues'])
        row={'experiment_id':experiment,'endpoint':endpoint,'record_id':rid,'joint_homology_group':a['joint_homology_group'],
            'eligible_residues':a['eligible_residues']}
        for name in [*METRIC_NAMES,'abs_bias']:
            left=abs(float(a['bias'])) if name=='abs_bias' else float(a[name]);right=abs(float(b['bias'])) if name=='abs_bias' else float(b[name])
            row['old_'+name]=left;row['new_'+name]=right;row['new_minus_old_'+name]=right-left
        output.append(row)
    return output


def assert_score_compatible(per_record,summary,reference_root,tolerance=1e-8):
    reference={r['record_id']:r for r in read_tsv(Path(reference_root)/'per_protein_metrics.tsv')}
    maximum=0.
    for row in per_record:
        old=reference[row['record_id']]
        assert int(old['eligible_residues'])==row['eligible_residues'] and old['joint_homology_group']==row['joint_homology_group']
        for metric in METRIC_NAMES:
            a=float(old[metric]) if old[metric] else float('nan');b=float(row[metric])
            if math.isnan(a):assert math.isnan(b);continue
            error=abs(a-b);maximum=max(maximum,error);assert error<tolerance,(reference_root,row['record_id'],metric,error)
    old_summary=json.loads((Path(reference_root)/'summary.json').read_text())
    old_summary=old_summary.get('metrics',old_summary)
    for aggregation in ['protein_record_macro','pooled_residue','unique_sequence_macro','homology_group_macro']:
        if aggregation not in old_summary:continue
        for metric in METRIC_NAMES:
            a=old_summary[aggregation][metric];b=summary[aggregation][metric]
            if a is None:assert not math.isfinite(b);continue
            error=abs(float(a)-float(b));maximum=max(maximum,error);assert error<tolerance,(aggregation,metric,error)
    return maximum
