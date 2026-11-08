"""Step7 only: frozen experimental comparisons, no training or model selection."""
from __future__ import annotations
import datetime as dt,hashlib,json,os
from pathlib import Path
import numpy as np
import yaml
from step7_experimental_comparison_v2_2 import *
from mdsop_runtime_v2_2 import require_frozen_models,runtime_identity,sha256_file,RUNTIME_PATH
from mdsop_experimental10_v2_2 import Experimental10AccessPolicy,Experimental10Dataset

A=P/'data'/T/'audits/step7_preexperimental10';FREEZE=A/'frozen_models_v2_2.json'
RESULT=P/'results'/T/'experimental10';CMP=RESULT/'comparison'


def load_records(cfg,policy):
    binding=yaml.safe_load(Path(cfg['runtime_binding']['data_binding_path']).read_text())
    whitelist=Path(binding['evaluator_whitelist']);entry=json.loads(whitelist.read_text())['endpoints']['experimental10']
    dataset=Experimental10Dataset(Experimental10AccessPolicy(whitelist,binding['evaluator_whitelist_sha256'],
        Path(cfg['data_access']['evaluation_root']),FREEZE),manifest_path=Path(entry['record_manifest']),
        manifest_sha256=entry['record_manifest_sha256'],expected_ids=tuple(policy['protein_ids']),expected_chains=policy['chains'])
    registry={r['record_id']:r for r in read_tsv(Path(policy['mask_registry_path']))};records=[]
    for item in dataset:
        record={**{k:item[k] for k in ['record_id','protein_id','pdb_id','chain_id','sequence_sha256','joint_homology_group','sequence']},
            'target':item['y'].numpy(),'qc_mask':item['qc_mask'].numpy()}
        reference=registry[record['record_id']]
        assert record['chain_id']==reference['authority_chain_id'] and record['sequence_sha256']==reference['sequence_sha256']
        assert len(record['target'])==int(reference['sequence_length']) and int(record['qc_mask'].sum())==int(reference['valid_labels'])
        digest=hashlib.sha256(record['qc_mask'].astype(np.uint8).tobytes()).hexdigest();assert digest==reference['mask_uint8_sha256']
        positions=','.join(str(v+1) for v in np.flatnonzero(record['qc_mask']));assert positions==reference['valid_sequence_positions_1based']
        records.append(record)
    assert len(records)==10 and sum(len(r['target']) for r in records)==982 and sum(int(r['qc_mask'].sum()) for r in records)==709
    assert len({r['joint_homology_group'] for r in records})==8
    return records,entry,binding


def main():
    os.umask(0o002);cfg,freeze=require_frozen_models(FREEZE,'experimental10')
    frozen=json.loads((A/'step7_inputs_and_code_freeze_v2_2.json').read_text());assert frozen['status']=='FROZEN_BEFORE_EXPERIMENTAL10_OPENING'
    for item in frozen['files']:assert sha256_file(item['path'])==item['sha256'],item['path']
    policy=json.loads(Path(frozen['policy_path']).read_text());assert not CMP.exists(),'Do not overwrite comparison'
    records,entry,binding=load_records(cfg,policy);experiments=list(cfg['experiments']);historical=policy['historical_models']
    old_root=Path(policy['old_comparison_root']);legacy=load_legacy_predictions(old_root/'all_models_per_residue_predictions.tsv.gz',records,experiments+historical)
    raw_check=verify_original_historical_source(Path(policy['original_historical_source']),records,legacy,policy['historical_raw_columns'])
    old_per=read_tsv(old_root/'all_models_per_protein_metrics.tsv');old_summary=read_tsv(old_root/'all_models_summary.tsv')
    identity=runtime_identity(cfg);predictions={};scores={};compat=[];controls=[];rows=[];all_protein=[]
    CMP.mkdir(parents=True)
    for exp in experiments:
        root=RESULT/exp
        pred=parse_predictions(root/'per_residue_predictions.tsv.gz',records,exp,'experimental10','v2.2',freeze['experiments'][exp]['checkpoint_sha256'])
        model='v2_2__'+exp;predictions[model]=pred
        meta={**identity,'model_id':model,'experiment_id':exp,'target_label_version':entry['label_version'],
            'checkpoint_file_sha256':freeze['experiments'][exp]['checkpoint_sha256'],'source_prediction_file_sha256':sha256_file(root/'per_residue_predictions.tsv.gz'),
            'step7_freeze_sha256':sha256_file(A/'step7_inputs_and_code_freeze_v2_2.json'),'model_family':'MD-SOP-v2.2'}
        per,summary=score_predictions(pred,records,meta)
        error=assert_score_compatible(per,summary,root);compat.append({'model_id':model,'maximum_metric_difference':error,'kind':'new_canonical_roundtrip'})
        scores[model]=(per,summary);rows.append(flat_summary(model,'v2.2',exp,summary));all_protein.extend(per)
    for original in experiments+historical:
        is_old=original in experiments;model='v2_1__'+original if is_old else original;version='v2.1' if is_old else 'historical'
        pred=legacy[original];predictions[model]=pred
        meta={'model_version':version,'model_id':model,'experiment_id':original,'run_tag':T,'model_family':'MD-SOP-v2.1' if is_old else 'historical_frozen_pipeline',
            'target_label_version':entry['label_version'],'source_prediction_file_sha256':sha256_file(old_root/'all_models_per_residue_predictions.tsv.gz'),
            'step7_freeze_sha256':sha256_file(A/'step7_inputs_and_code_freeze_v2_2.json'),
            'historical_model_registry_sha256':frozen['historical_model_registry_sha256']}
        per,summary=write_scored_endpoint(CMP/'reference_scores'/model,pred,records,meta)
        error=assert_legacy_scores(per,summary,original,old_per,old_summary)
        compat.append({'model_id':model,'maximum_metric_difference':error,'kind':'frozen_legacy_scores_reproduced'})
        scores[model]=(per,summary);rows.append(flat_summary(model,version,original,summary));all_protein.extend(per)
    for exp in experiments:
        if exp.startswith('B_'):
            left=predictions['v2_1__'+exp];right=predictions['v2_2__'+exp]
            difference=max(float(np.max(np.abs(left[rid]-right[rid]))) for rid in left)
            assert difference<policy['control_prediction_tolerance']
            controls.append({'experiment_id':exp,'maximum_absolute_prediction_difference':difference,
                'canonical_float32_digest_equal':prediction_digest(left)==prediction_digest(right),'status':'PASS'})
    pairs,stats=paired_comparisons(scores,experiments,historical,policy['statistics'])
    assert len(rows)==15 and len(all_protein)==150 and len(pairs)==240 and len(stats)==168
    atomic_write_tsv(CMP/'all_15_models_summary.tsv',rows,list(rows[0]));atomic_write_json(CMP/'all_15_models_summary.json',rows)
    atomic_write_tsv(CMP/'primary_six_v2_2_experimental_scores.tsv',[r for r in rows if r['model_version']=='v2.2'],list(rows[0]))
    atomic_write_tsv(CMP/'all_models_per_protein_metrics.tsv',all_protein,list(dict.fromkeys(k for r in all_protein for k in r)))
    freeze_digest=sha256_file(A/'step7_inputs_and_code_freeze_v2_2.json')
    def residue_rows():
        for model in scores:
            summary=scores[model][1]
            for record in records:
                for j,value in enumerate(predictions[model][record['record_id']]):
                    keep=bool(record['qc_mask'][j])
                    yield {'model_id':model,'model_version':summary['model_version'],'record_id':record['record_id'],
                        'protein_id':record['protein_id'],'chain_id':record['chain_id'],'joint_homology_group':record['joint_homology_group'],
                        'sequence_sha256':record['sequence_sha256'],'residue_index_1based':j+1,'amino_acid':record['sequence'][j],
                        'qc_eligible':keep,'target':float(record['target'][j]) if keep else None,'prediction':float(value),
                        'source_prediction_file_sha256':summary['source_prediction_file_sha256'],
                        'step7_freeze_sha256':freeze_digest}
    first=next(residue_rows());atomic_write_gzip_tsv(CMP/'all_15_models_per_residue_predictions.tsv.gz',residue_rows(),list(first))
    atomic_write_tsv(CMP/'paired_per_protein_differences.tsv',pairs,list(pairs[0]))
    atomic_write_tsv(CMP/'paired_group_statistics.tsv',stats,list(dict.fromkeys(k for r in stats for k in r)))
    atomic_write_json(CMP/'prediction_identity_and_compatibility.json',{'digests':{m:prediction_digest(p) for m,p in predictions.items()},
        'legacy_compatibility':compat,'historical_raw_source_check':raw_check,'SOPPE_only_control':controls})
    plot_results(CMP,records,predictions,scores,experiments,historical);render_html(CMP,rows,records)
    result={'status':'PASS','step':7,'run_tag':T,'created_at':dt.datetime.now().astimezone().isoformat(),
        'models':15,'new_models':6,'old_version_models':6,'historical_reference_models':3,
        'record_scores':150,'full_prediction_positions':15*982,'scored_positions_across_all_models':15*709,
        'reference_homology_groups':8,'pairwise_comparisons':24,'per_protein_paired_rows':len(pairs),'paired_metric_summaries':len(stats),
        'maximum_score_compatibility_difference':max(r['maximum_metric_difference'] for r in compat),'SOPPE_only_control':controls,
        'new_training_or_model_selection':False,'prediction_transform':'none','historical_three_endpoint_summary_created':False}
    atomic_write_json(CMP/'step7_comparison_completion_v2_2_20261107.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
