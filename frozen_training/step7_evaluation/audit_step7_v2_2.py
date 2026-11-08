"""Independent Experimental10 scoring/statistics/provenance audit."""
from __future__ import annotations
from collections import defaultdict
import csv,datetime as dt,gzip,hashlib,itertools,json,math,os,pwd
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr

P=Path('/data/jiangbin/SOPPCL/MD-SOP');T='v2_2_20261107';CODE=P/'code'/T
sys.path.insert(0,str(CODE))
from mdsop_runtime_v2_2 import require_frozen_models,sha256_file
A=P/'data'/T/'audits/step7_preexperimental10';CMP=P/'results'/T/'experimental10/comparison'


def read(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))


def num(v):return float(v) if v not in ['',None] else float('nan')
def mean(values):
    values=np.asarray(values,dtype=float);valid=np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else float('nan')
def direct_metrics(p,y):
    error=p-y;defined=len(p)>1 and np.ptp(p)>0 and np.ptp(y)>0
    return {'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(np.abs(error))),
        'bias':float(error.mean()),'pcc':float(np.corrcoef(p,y)[0,1]) if defined else np.nan,
        'spearman':float(spearmanr(p,y).statistic) if defined else np.nan,
        'out_of_range_fraction':float(np.mean((p<0)|(p>1)))}


def main():
    assert pwd.getpwuid(os.geteuid()).pw_name=='amax';os.umask(0o002)
    cfg,freeze=require_frozen_models(A/'frozen_models_v2_2.json','experimental10')
    frozen=json.loads((A/'step7_inputs_and_code_freeze_v2_2.json').read_text());policy=json.loads(Path(frozen['policy_path']).read_text())
    assert json.loads((CMP/'step7_comparison_completion_v2_2_20261107.json').read_text())['status']=='PASS'
    for item in frozen['files']:assert sha256_file(item['path'])==item['sha256'],item['path']
    for item in frozen['preserved_assets']:
        q=Path(item['path']);s=q.stat();assert sha256_file(q)==item['sha256'] and s.st_size==item['bytes'] and s.st_mtime_ns==item['mtime_ns'],str(q)
    maximum=0.;max_ci_error=0.
    def near(a,b):
        nonlocal maximum
        a=num(a);b=num(b)
        if math.isnan(a):assert math.isnan(b);return
        delta=abs(a-b);maximum=max(maximum,delta);assert delta<1e-8,(a,b,delta)
    registry={r['record_id']:r for r in read(Path(policy['mask_registry_path']))}
    authority={}
    for row in read(P/'data'/T/'evaluation/experimental10/manifest.tsv'):
        rid=row['record_id'];path=Path(row['label_path'])
        assert path.resolve().is_relative_to(P/'data'/T/'evaluation/experimental10/labels')
        assert row['role_v2_2']=='reference_experimental10' and row['partition_v2_2']=='protected'
        assert row['source']=='historical_test' and sha256_file(path)==row['label_file_sha256']
        with np.load(path,allow_pickle=False) as data:
            y=np.array(data['y']);q=np.array(data['qc_mask']);sequence=bytes(data['sequence_ascii']).decode('ascii')
            assert not data['train_access'].any() and data['eval_access'].all() and not data['loss_mask'].any()
        ref=registry[rid];assert len(y)==int(ref['sequence_length']) and int(q.sum())==int(ref['valid_labels'])
        assert hashlib.sha256(sequence.encode()).hexdigest()==ref['sequence_sha256']==row['sequence_sha256']
        assert hashlib.sha256(q.astype(np.uint8).tobytes()).hexdigest()==ref['mask_uint8_sha256']
        authority[rid]={'y':y,'q':q,'sequence':sequence,'protein_id':row['protein_id'],'chain_id':ref['authority_chain_id'],
            'group':row['joint_homology_group'],'sequence_sha256':row['sequence_sha256']}
    assert len(authority)==10 and sum(len(r['y']) for r in authority.values())==982 and sum(int(r['q'].sum()) for r in authority.values())==709
    assert len({r['group'] for r in authority.values()})==8
    summaries=read(CMP/'all_15_models_summary.tsv');assert len(summaries)==15
    protein={(r['model_id'],r['record_id']):r for r in read(CMP/'all_models_per_protein_metrics.tsv')};assert len(protein)==150
    grouped=defaultdict(lambda:defaultdict(list))
    for row in read(CMP/'all_15_models_per_residue_predictions.tsv.gz'):grouped[row['model_id']][row['record_id']].append(row)
    assert set(grouped)=={r['model_id'] for r in summaries}
    original_rows=read(Path(policy['old_comparison_root'])/'all_models_per_residue_predictions.tsv.gz')
    original=defaultdict(lambda:defaultdict(list))
    for row in original_rows:original[row['model_id']][row['record_id']].append(row)
    recalculated={};positions=0
    for summary in summaries:
        model=summary['model_id'];assert set(grouped[model])==set(authority)
        direct=[];pp=[];yy=[];digest=hashlib.sha256()
        for rid in sorted(authority):
            ref=authority[rid];rows=sorted(grouped[model][rid],key=lambda r:int(r['residue_index_1based']))
            assert [int(r['residue_index_1based']) for r in rows]==list(range(1,len(ref['y'])+1))
            assert ''.join(r['amino_acid'] for r in rows)==ref['sequence']
            assert all(r['chain_id']==ref['chain_id'] and r['joint_homology_group']==ref['group'] and r['sequence_sha256']==ref['sequence_sha256'] for r in rows)
            q=np.asarray([r['qc_eligible']=='True' for r in rows]);assert np.array_equal(q,ref['q'])
            for row,keep,target in zip(rows,q,ref['y']):
                if keep:assert np.float32(float(row['target']))==target
                else:assert row['target']=='','Missing experimental label was filled'
            p=np.asarray([float(r['prediction']) for r in rows]);assert np.isfinite(p).all()
            if model.startswith('v2_2__'):
                exp=model.replace('v2_2__','',1)
                source=read(P/'results'/T/'experimental10'/exp/'per_residue_predictions.tsv.gz')
                source=[r for r in source if r['record_id']==rid]
            else:source=original[model.replace('v2_1__','',1)][rid]
            source=sorted(source,key=lambda r:int(r['residue_index_1based']))
            assert np.array_equal(p,np.asarray([float(r['prediction']) for r in source])),'Frozen raw predictions changed'
            current=direct_metrics(p[q],ref['y'][q].astype(np.float64));stored=protein[model,rid]
            for metric,value in current.items():near(value,stored[metric])
            direct.append({'rid':rid,'group':ref['group'],**current});pp.append(p[q]);yy.append(ref['y'][q].astype(np.float64))
            digest.update(rid.encode());digest.update(np.asarray(p,dtype='<f4').tobytes());positions+=len(p)
        assert digest.hexdigest()==summary['prediction_sha256']
        for metric in current:
            near(mean([r[metric] for r in direct]),summary['macro_'+metric])
            near(direct_metrics(np.concatenate(pp),np.concatenate(yy))[metric],summary['pooled_'+metric])
            groups=defaultdict(list)
            for r in direct:groups[r['group']].append(r[metric])
            near(mean([mean(v) for v in groups.values()]),summary['homology_group_macro_'+metric])
        recalculated[model]={r['rid']:r for r in direct}
    assert positions==14730
    paired=read(CMP/'paired_per_protein_differences.tsv');statistics=read(CMP/'paired_group_statistics.tsv')
    assert len(paired)==240 and len(statistics)==168
    pairs_by_id=defaultdict(list)
    for row in paired:
        a=recalculated[row['reference_model_id']][row['record_id']];b=recalculated[row['new_model_id']][row['record_id']]
        for metric in [*current,'abs_bias']:
            left=abs(a['bias']) if metric=='abs_bias' else a[metric];right=abs(b['bias']) if metric=='abs_bias' else b[metric]
            near(left,row['old_'+metric]);near(right,row['new_'+metric]);near(right-left,row['new_minus_old_'+metric])
        pairs_by_id[row['comparison_id']].append(row)
    ordered_specs=[('v2_1__'+e,'v2_2__'+e) for e in cfg['experiments']]
    ordered_specs += [(h,'v2_2__'+e) for e in cfg['experiments'] for h in policy['historical_models']]
    names=[*current,'abs_bias'];p_values=[]
    for row in statistics:
        metric=row['metric'];groups=defaultdict(list)
        for pair in pairs_by_id[row['comparison_id']]:
            d=num(pair['new_minus_old_'+metric])
            if math.isfinite(d):groups[pair['joint_homology_group']].append(d)
        sums=np.array([sum(v) for k,v in sorted(groups.items())]);counts=np.array([len(v) for k,v in sorted(groups.items())]);means=sums/counts
        near(sums.sum()/counts.sum(),row['record_macro_delta']);near(means.mean(),row['group_balanced_delta'])
        i=ordered_specs.index((row['reference_model_id'],row['new_model_id']));mi=names.index(metric)
        rng=np.random.default_rng(policy['statistics']['bootstrap_seed']+i*10+mi)
        idx=rng.integers(0,len(groups),size=(policy['statistics']['bootstrap_replicates'],len(groups)))
        weighted=sums[idx].sum(1)/counts[idx].sum(1);balanced=means[idx].mean(1)
        ci=[*np.percentile(weighted,[2.5,97.5]),*np.percentile(balanced,[2.5,97.5])]
        fields=['record_macro_group_bootstrap95_low','record_macro_group_bootstrap95_high','group_balanced_bootstrap95_low','group_balanced_bootstrap95_high']
        for field,value in zip(fields,ci):
            error=abs(value-num(row[field]));max_ci_error=max(max_ci_error,float(error));assert error<1e-8
        if metric=='rmse':
            assert len(groups)==8 and row['sign_flip_method']=='exact' and int(row['sign_flip_draws'])==256
            signs=np.asarray(list(itertools.product([-1.,1.],repeat=8)));p=np.mean(np.abs((signs*means).mean(1))>=abs(means.mean())-1e-15)
            near(p,row['group_sign_flip_p']);p_values.append((row,float(p)))
    assert len(p_values)==24;running=0.
    for rank,(row,value) in enumerate(sorted(p_values,key=lambda r:r[1])):
        running=max(running,min(1.,(24-rank)*value));near(running,row['holm_p_family_24_RMSE_comparisons'])
    count=0
    for item in read(P/'data'/T/'manifests/v2_1_reference_registry.tsv'):
        q=Path(item['remote_path']);s=q.stat();assert sha256_file(q)==item['sha256'] and s.st_size==int(item['bytes']) and s.st_mtime_ns==int(item['mtime_ns']);count+=1
    count3=0
    for item in read(P/'data'/T/'audits/step3_data_preparation/step3_artifact_sha256.tsv'):
        q=P/item['relative_path'];assert sha256_file(q)==item['sha256'] and q.stat().st_size==int(item['bytes']);count3+=1
    curves=CMP/'figures/protein_curves'
    for r in authority.values():
        for extension in ['png','svg']:assert (curves/(r['protein_id']+'.'+extension)).stat().st_size>1000
    for extension in ['png','svg']:
        for name in ['all_15_models_overview','historical_reference_curves']:assert (CMP/'figures'/(name+'.'+extension)).stat().st_size>1000
    assert (CMP/'experimental10_comparison_v2_2_20261107.html').exists()
    owner=pwd.getpwnam('amax')
    for root in [A,CODE/'step7_evaluation',P/'results'/T/'experimental10']:
        for q in root.rglob('*'):
            s=q.stat();assert s.st_uid==owner.pw_uid and s.st_gid==owner.pw_gid
    result={'status':'PASS','step':7,'run_tag':T,'created_at':dt.datetime.now().astimezone().isoformat(),
        'models_verified':15,'protein_scores_independently_recomputed':150,'full_prediction_positions_verified':positions,
        'valid_positions_per_model':709,'reference_homology_groups':8,'paired_rows_verified':len(paired),'statistics_verified':len(statistics),
        'CI_bounds_replicated':4*len(statistics),'exact_sign_flip_and_Holm_tests_replicated':24,
        'maximum_metric_difference':maximum,'maximum_CI_bound_difference':max_ci_error,
        'original_assets_unchanged':count,'step3_assets_unchanged':count3,'preserved_pre_step7_assets_unchanged':len(frozen['preserved_assets']),
        'frozen_inputs_and_code_unchanged':len(frozen['files']),'no_training_or_model_selection':True,
        'all_10_protein_curve_PNG_SVG_pairs_present':True,'experimental_missing_labels_preserved':True,
        'model_freeze_sha256':sha256_file(A/'frozen_models_v2_2.json'),'companion_freeze_sha256':sha256_file(A/'step7_inputs_and_code_freeze_v2_2.json')}
    with (A/'step7_independent_audit_v2_2_20261107.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
