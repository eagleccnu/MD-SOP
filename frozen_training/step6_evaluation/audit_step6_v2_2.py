"""Independent mathematical/file audit of all 36 version/target score cells."""
from __future__ import annotations
from collections import defaultdict
import csv,datetime as dt,gzip,hashlib,json,math,os,pwd
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr

P=Path('/data/jiangbin/SOPPCL/MD-SOP');T='v2_2_20261107';CODE=P/'code'/T
sys.path.insert(0,str(CODE))
from mdsop_runtime_v2_2 import require_frozen_models,sha256_file
A=P/'data'/T/'audits/step6_pretest';RESULT=P/'results'/T;CMP=RESULT/'step6_comparison'


def read(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))


def number(x):return float(x) if x not in ['',None] else float('nan')
def nm(values):
    values=np.asarray(values,dtype=float);return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float('nan')
def metrics(p,y):
    e=p-y;valid=len(p)>1 and np.ptp(p)>0 and np.ptp(y)>0
    return {'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e))),'bias':float(e.mean()),
        'pcc':float(np.corrcoef(p,y)[0,1]) if valid else float('nan'),
        'spearman':float(spearmanr(p,y).statistic) if valid else float('nan'),
        'out_of_range_fraction':float(np.mean((p<0)|(p>1)))}


def main():
    assert pwd.getpwuid(os.geteuid()).pw_name=='amax';os.umask(0o002)
    cfg,freeze=require_frozen_models(A/'frozen_models_v2_2.json','md_test')
    frozen=json.loads((A/'step6_inputs_and_code_freeze_v2_2.json').read_text())
    policy=json.loads(Path(frozen['policy_path']).read_text())
    complete=json.loads((CMP/'step6_comparison_completion_v2_2_20261107.json').read_text());assert complete['status']=='PASS'
    for r in frozen['files']:assert sha256_file(r['path'])==r['sha256'],r['path']
    for r in frozen['training_assets']:
        q=Path(r['path']);s=q.stat();assert sha256_file(q)==r['sha256'] and s.st_mtime_ns==r['mtime_ns'] and s.st_size==r['bytes']
    def near(a,b):
        nonlocal maximum
        a=number(a);b=number(b)
        if math.isnan(a):assert math.isnan(b);return
        difference=abs(a-b);maximum=max(maximum,difference);assert difference<1e-8,(a,b,difference)
    maximum=0.;cells=0;record_scores=0;full_predictions={};authorities={};total_positions=0
    for endpoint,spec in policy['endpoints'].items():
        require_frozen_models(A/'frozen_models_v2_2.json',endpoint)
        manifest=P/'data'/T/'evaluation'/endpoint/'manifest.tsv'
        rows=read(manifest);assert len(rows)==spec['records'];lookup={}
        for r in rows:
            label=Path(r['label_path']);assert label.resolve().is_relative_to(P/'data'/T/'evaluation'/endpoint/'labels')
            assert r['role_v2_2']=='test' and r['train_access']=='False' and r['eval_access']=='True'
            assert sha256_file(label)==r['label_file_sha256']
            with np.load(label,allow_pickle=False) as data:
                y=np.array(data['y']);q=np.array(data['qc_mask']);seq=bytes(data['sequence_ascii']).decode('ascii')
                assert not data['train_access'].any() and data['eval_access'].all() and not data['loss_mask'].any()
            assert y.shape==q.shape==(int(r['sequence_length']),)
            assert hashlib.sha256(seq.encode()).hexdigest()==r['sequence_sha256']
            lookup[r['record_id']]={'y':y,'q':q,'seq':seq,'row':r}
        assert sum(len(v['y']) for v in lookup.values())==spec['complete_residues']
        assert sum(int(v['q'].sum()) for v in lookup.values())==spec['eligible_residues']
        authorities[endpoint]=lookup
    summary_rows=read(CMP/'all_version_target_metrics_v2_2_20261107.tsv')
    assert len(summary_rows)==36
    stat_lookup={}
    for entry in summary_rows:
        exp=entry['experiment_id'];version=entry['model_version'];target=entry['target_endpoint'];base=authorities[target]
        if version=='v2.1':root=RESULT/'v2_1_fixed_prediction_rescoring'/target/exp
        else:root=RESULT/('md_test_5tauc' if target=='md_test' else target)/exp
        per={r['record_id']:r for r in read(root/'per_protein_metrics.tsv')};summary=json.loads((root/'summary.json').read_text())
        grouped=defaultdict(list)
        for row in read(root/'per_residue_predictions.tsv.gz'):grouped[row['record_id']].append(row)
        assert set(grouped)==set(base)==set(per)
        direct=[];pp=[];yy=[];all_arrays={}
        for rid,authority in base.items():
            rows=sorted(grouped[rid],key=lambda r:int(r['residue_index_1based']))
            assert [int(r['residue_index_1based']) for r in rows]==list(range(1,len(authority['y'])+1))
            assert ''.join(r['amino_acid'] for r in rows)==authority['seq']
            assert all(r['sequence_sha256']==authority['row']['sequence_sha256'] for r in rows)
            q=np.asarray([r['qc_eligible']=='True' for r in rows]);assert np.array_equal(q,authority['q'])
            y=np.asarray([number(r['target']) if r['target'] else 0. for r in rows],dtype=np.float32)
            assert np.array_equal(y[q],authority['y'][q])
            p=np.asarray([float(r['prediction']) for r in rows]);assert np.isfinite(p).all()
            values=metrics(p[q],authority['y'][q].astype(np.float64))
            for name,value in values.items():near(value,per[rid][name])
            direct.append({'record_id':rid,'sequence_sha256':authority['row']['sequence_sha256'],
                'joint_homology_group':authority['row']['joint_homology_group'],**values})
            pp.append(p[q]);yy.append(authority['y'][q].astype(np.float64));all_arrays[rid]=p
            total_positions+=len(p);record_scores+=1
        direct_macro={m:nm([r[m] for r in direct]) for m in values}
        direct_pooled=metrics(np.concatenate(pp),np.concatenate(yy))
        for prefix,aggregation,current in [('macro','protein_record_macro',direct_macro),('pooled','pooled_residue',direct_pooled)]:
            for name,value in current.items():near(value,entry[prefix+'_'+name]);near(value,summary.get('metrics',summary)[aggregation][name])
        for field,prefix,key in [('sequence_sha256','unique_sequence_macro','unique_sequence_macro'),('joint_homology_group','homology_group_macro','homology_group_macro')]:
            groups=defaultdict(list)
            for r in direct:groups[r[field]].append(r)
            for metric in values:
                aggregate=nm([nm([r[metric] for r in rs]) for rs in groups.values()])
                near(aggregate,entry[prefix+'_'+metric]);near(aggregate,summary.get('metrics',summary)[key][metric])
        digest=hashlib.sha256()
        for rid,p in sorted(all_arrays.items()):digest.update(rid.encode());digest.update(np.asarray(p,dtype='<f4').tobytes())
        assert digest.hexdigest()==entry['prediction_sha256']
        old_key=(exp,version,'md' if target.startswith('md_test') else 's')
        if old_key in full_predictions:
            for rid,p in all_arrays.items():assert np.array_equal(p,full_predictions[old_key][rid]),'Predictions changed with target'
        else:full_predictions[old_key]=all_arrays
        stat_lookup[exp,version,target]={r['record_id']:r for r in direct};cells+=1
    pairs=read(CMP/'paired_per_protein_new_minus_old_v2_2_20261107.tsv')
    assert len(pairs)==6*(136+273+136)
    for row in pairs:
        a=stat_lookup[row['experiment_id'],'v2.1',row['endpoint']][row['record_id']]
        b=stat_lookup[row['experiment_id'],'v2.2',row['endpoint']][row['record_id']]
        for metric in [*values,'abs_bias']:
            x=abs(a['bias']) if metric=='abs_bias' else a[metric];y=abs(b['bias']) if metric=='abs_bias' else b[metric]
            near(x,row['old_'+metric]);near(y,row['new_'+metric]);near(y-x,row['new_minus_old_'+metric])
    statistics=read(CMP/'paired_group_statistics_v2_2_20261107.tsv');assert len(statistics)==126
    for row in statistics:
        metric=row['metric'];grouped=defaultdict(list);differences=[]
        for pair in pairs:
            if pair['experiment_id']==row['experiment_id'] and pair['endpoint']==row['target_endpoint']:
                d=number(pair['new_minus_old_'+metric])
                if math.isfinite(d):differences.append(d);grouped[pair['joint_homology_group']].append(d)
        assert len(differences)==int(row['valid_record_pairs']) and len(grouped)==int(row['valid_homology_groups'])
        near(np.mean(differences),row['record_macro_delta']);near(np.mean([np.mean(v) for v in grouped.values()]),row['group_balanced_delta'])
        assert float(row['record_macro_group_bootstrap95_low'])<=float(row['record_macro_group_bootstrap95_high'])
    rmse=[r for r in statistics if r['metric']=='rmse'];assert len(rmse)==18
    previous=0.
    for rank,r in enumerate(sorted(rmse,key=lambda r:float(r['group_sign_flip_p']))):
        expected=max(previous,min(1.,(18-rank)*float(r['group_sign_flip_p'])));near(expected,r['holm_p_family_18_rmse_version_pairs']);previous=expected
    old_count=0
    for r in read(P/'data'/T/'manifests/v2_1_reference_registry.tsv'):
        q=Path(r['remote_path']);s=q.stat();assert sha256_file(q)==r['sha256'] and s.st_size==int(r['bytes']) and s.st_mtime_ns==int(r['mtime_ns']);old_count+=1
    count3=0
    for r in read(P/'data'/T/'audits/step3_data_preparation/step3_artifact_sha256.tsv'):
        q=P/r['relative_path'];assert sha256_file(q)==r['sha256'] and q.stat().st_size==int(r['bytes']);count3+=1
    assert not (P/'data'/T/'audits/step7_preexperimental10').exists()
    assert not [q for q in (RESULT/'experimental10').rglob('*') if q.is_file()]
    owner=pwd.getpwnam('amax')
    for root in [A,CODE/'step6_evaluation',RESULT/'step6_comparison',RESULT/'v2_1_fixed_prediction_rescoring',
                 RESULT/'md_test_5tauc',RESULT/'soppe_test',RESULT/'md_test_full100ns_reference']:
        for q in root.rglob('*'):
            s=q.stat();assert s.st_uid==owner.pw_uid and s.st_gid==owner.pw_gid
    result={'status':'PASS','step':6,'run_tag':T,'created_at':dt.datetime.now().astimezone().isoformat(),
        'version_target_score_cells_verified':cells,'per_record_scores_independently_recomputed':record_scores,
        'full_length_prediction_positions_checked':total_positions,'paired_protein_rows_verified':len(pairs),
        'paired_metric_summaries_checked':len(statistics),'maximum_absolute_metric_difference':maximum,
        'same_predictions_under_both_MD_targets':True,'all_original_assets_unchanged':old_count,
        'all_step3_assets_unchanged':count3,'all_training_assets_unchanged':len(frozen['training_assets']),
        'pretest_files_unchanged':len(frozen['files']),'Experimental10_opened':False,'new_training_performed':False,
        'model_freeze_sha256':sha256_file(A/'frozen_models_v2_2.json'),
        'step6_inputs_and_code_freeze_sha256':sha256_file(A/'step6_inputs_and_code_freeze_v2_2.json')}
    output=A/'step6_independent_audit_v2_2_20261107.json'
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
