"""Rescore one immutable prediction per version against fixed target releases."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from step6_comparison_v2_2 import (METRIC_NAMES,read_tsv,parse_predictions,prediction_digest,score_predictions,write_scored_endpoint,
    paired_records,grouped_pair_statistics,sign_flip_p,holm_adjust,assert_score_compatible,
    atomic_write_json,atomic_write_tsv)
from mdsop_runtime_v2_2 import PROJECT as P,RUN_TAG as T,RUNTIME_PATH,require_frozen_models,runtime_identity,sha256_file
from mdsop_evaluation_v2_2 import EvaluationAccessPolicy,EvaluationProteinDataset,STEP6_ENDPOINTS

A=P/'data'/T/'audits/step6_pretest'
FREEZE=A/'frozen_models_v2_2.json'
COMPARISON=P/'results'/T/'step6_comparison'


def load_endpoints(cfg,freeze,policy):
    binding=yaml.safe_load(Path(cfg['runtime_binding']['data_binding_path']).read_text())
    white_path=Path(binding['evaluator_whitelist']);white=json.loads(white_path.read_text())
    ep={};identity={}
    for name in STEP6_ENDPOINTS:
        spec=white['endpoints'][name]
        dataset=EvaluationProteinDataset(EvaluationAccessPolicy(white_path,binding['evaluator_whitelist_sha256'],
            Path(cfg['data_access']['evaluation_root']),STEP6_ENDPOINTS,FREEZE),endpoint=name,
            manifest_path=Path(spec['record_manifest']),manifest_sha256=spec['record_manifest_sha256'],
            expected_source='soppe_legacy' if name=='soppe_test' else 'md_ired')
        records=[]
        for item in dataset:
            records.append({**{k:item[k] for k in ['record_id','protein_id','sequence_sha256','joint_homology_group','sequence']},
                'target':item['y'].numpy(),'qc_mask':item['qc_mask'].numpy()})
        expected=policy['endpoints'][name]
        assert len(records)==expected['records']
        assert sum(len(r['target']) for r in records)==expected['complete_residues']
        assert sum(int(r['qc_mask'].sum()) for r in records)==expected['eligible_residues']
        assert len({r['sequence_sha256'] for r in records})==expected['unique_sequences']
        ep[name]=records
        identity[name]={'evaluation_manifest_sha256':spec['record_manifest_sha256'],
            'evaluator_whitelist_sha256':binding['evaluator_whitelist_sha256'],'target_label_version':spec['label_version']}
    main={r['record_id']:r for r in ep['md_test']};aux={r['record_id']:r for r in ep['md_test_full100ns_reference']}
    assert set(main)==set(aux)
    for rid,a in main.items():
        b=aux[rid]
        assert a['sequence']==b['sequence'] and a['sequence_sha256']==b['sequence_sha256']
        assert a['joint_homology_group']==b['joint_homology_group'] and np.array_equal(a['qc_mask'],b['qc_mask'])
    return ep,identity


def flat_summary(version,exp,endpoint,summary):
    result={'model_version':version,'experiment_id':exp,'target_endpoint':endpoint,'record_count':summary['record_count'],
        'unique_sequence_count':summary['unique_sequence_count'],'homology_group_count':summary['joint_homology_group_count'],
        'complete_residue_count':summary['complete_residue_count'],'eligible_residue_count':summary['eligible_residue_count'],
        'prediction_sha256':summary['prediction_sha256'],'evaluation_input_sha256':summary['evaluation_input_sha256']}
    for prefix,key in [('macro','protein_record_macro'),('pooled','pooled_residue'),('unique_sequence_macro','unique_sequence_macro'),('homology_group_macro','homology_group_macro')]:
        result.update({prefix+'_'+m:summary[key][m] for m in METRIC_NAMES})
    result.update({'macro_pcc_valid_records':summary['protein_record_macro']['pcc_valid_records'],
        'macro_spearman_valid_records':summary['protein_record_macro']['spearman_valid_records']})
    return result


def main():
    os.umask(0o002)
    cfg,freeze=require_frozen_models(FREEZE,'md_test')
    frozen=json.loads((A/'step6_inputs_and_code_freeze_v2_2.json').read_text());assert frozen['status']=='FROZEN_BEFORE_TEST_OPENING'
    for item in frozen['files']:assert sha256_file(item['path'])==item['sha256'],item['path']
    policy=json.loads(Path(frozen['policy_path']).read_text())
    assert not COMPARISON.exists(),'Do not overwrite a published comparison'
    ep,target_identity=load_endpoints(cfg,freeze,policy)
    main_identity=runtime_identity(cfg)
    rows=[];all_pairs=[];statistics=[];scores={};compat=[];control=[];digests=[]
    for exp in cfg['experiments']:
        scores[exp]={}
        for version in ['v2.1','v2.2']:
            scores[exp][version]={}
            for domain in ['md_test','soppe_test']:
                targets=['md_test','md_test_full100ns_reference'] if domain=='md_test' else ['soppe_test']
                source_records=ep['md_test_full100ns_reference'] if version=='v2.1' and domain=='md_test' else ep[domain]
                if version=='v2.1':source=P/'results/v2_1_20261007'/domain/exp
                else:source=P/'results'/T/('md_test_5tauc' if domain=='md_test' else domain)/exp
                predictions=parse_predictions(source/'per_residue_predictions.tsv.gz',source_records,exp,domain,
                    'v2.2' if version=='v2.2' else None,freeze['experiments'][exp]['checkpoint_sha256'] if version=='v2.2' else None)
                digest=prediction_digest(predictions)
                digests.append({'experiment_id':exp,'model_version':version,'domain':domain,'prediction_sha256':digest,
                    'source_prediction_path':str(source/'per_residue_predictions.tsv.gz'),'source_prediction_file_sha256':sha256_file(source/'per_residue_predictions.tsv.gz')})
                for target in targets:
                    current={'model_version':version,'run_tag':T,'experiment_id':exp,'endpoint':target,
                        'md_label_version':'md_ired_5tauc_windowmean_v2_2' if version=='v2.2' else 'md_ired_full100ns_three_replica_mean_v2_1',
                        'runtime_config_sha256':sha256_file(RUNTIME_PATH),'data_binding_sha256':main_identity['data_binding_sha256'],
                        'training_code_reference_sha256':main_identity['code_manifest_sha256'] if version=='v2.2' else frozen['v2_1_model_code_sha256'],
                        'training_code_reference_kind':'accepted_20_file_manifest' if version=='v2.2' else 'v2_1_model_code_file',
                        'step6_code_manifest_sha256':sha256_file(A/'step6_inputs_and_code_freeze_v2_2.json'),
                        'source_prediction_file_sha256':sha256_file(source/'per_residue_predictions.tsv.gz'),
                        'checkpoint_file_sha256':freeze['experiments'][exp]['checkpoint_sha256'] if version=='v2.2' else frozen['v2_1_best_checkpoints'][exp]['sha256'],
                        **target_identity[target]}
                    if version=='v2.1':destination=P/'results'/T/'v2_1_fixed_prediction_rescoring'/target/exp
                    elif target=='md_test_full100ns_reference':destination=P/'results'/T/target/exp
                    else:destination=None
                    if destination is None:
                        per,summary=score_predictions(predictions,ep[target],current)
                        observed=assert_score_compatible(per,summary,source)
                        compat.append({'model_version':version,'experiment_id':exp,'target':target,'maximum_metric_difference':observed,'kind':'v2_2_raw_output_roundtrip'})
                        # Canonical inference summaries retain float32 predictions;
                        # all comparisons use the common serialized-prediction scale.
                    else:per,summary=write_scored_endpoint(destination,predictions,ep[target],current)
                    assert summary['prediction_sha256']==digest
                    if version=='v2.1' and target==('md_test_full100ns_reference' if domain=='md_test' else domain):
                        observed=assert_score_compatible(per,summary,source)
                        compat.append({'model_version':version,'experiment_id':exp,'target':target,'maximum_metric_difference':observed,'kind':'old_published_baseline_reproduced'})
                    scores[exp][version][target]=(per,summary,predictions)
                    rows.append(flat_summary(version,exp,target,summary))
                print('SCORED',exp,version,domain,flush=True)
        for target in ['md_test','soppe_test','md_test_full100ns_reference']:
            old=scores[exp]['v2.1'][target];new=scores[exp]['v2.2'][target]
            assert old[1]['evaluation_input_sha256']==new[1]['evaluation_input_sha256']
            pairs=paired_records(old[0],new[0],exp,target);all_pairs.extend(pairs)
            pair_index=list(cfg['experiments']).index(exp)*3+['md_test','soppe_test','md_test_full100ns_reference'].index(target)
            for mi,metric in enumerate([*METRIC_NAMES,'abs_bias']):
                stat=grouped_pair_statistics(pairs,metric,policy['statistics']['bootstrap_replicates'],policy['statistics']['bootstrap_seed']+pair_index*10+mi)
                means=stat.pop('group_values',None)
                result={'experiment_id':exp,'target_endpoint':target,'metric':metric,**stat}
                if metric=='rmse':
                    p,method,n=sign_flip_p(means,policy['statistics']['sign_flip_replicates'],policy['statistics']['sign_flip_seed']+pair_index,
                        policy['statistics']['exact_sign_flip_max_groups'])
                    result.update({'group_sign_flip_p':p,'sign_flip_method':method,'sign_flip_draws':n})
                statistics.append(result)
        if exp.startswith('B_'):
            for target in ['md_test','soppe_test']:
                old=scores[exp]['v2.1'][target][2];new=scores[exp]['v2.2'][target][2]
                maximum=max(float(np.max(np.abs(old[rid]-new[rid]))) for rid in old)
                assert maximum<policy['control_prediction_tolerance'],(exp,target,maximum)
                control.append({'experiment_id':exp,'domain':target,'maximum_absolute_prediction_difference':maximum,
                    'canonical_float32_prediction_digest_equal':prediction_digest(old)==prediction_digest(new),'status':'PASS'})
    rmse=[r for r in statistics if r['metric']=='rmse'];assert len(rmse)==18
    for r,p in zip(rmse,holm_adjust([r['group_sign_flip_p'] for r in rmse])):r['holm_p_family_18_rmse_version_pairs']=p
    COMPARISON.mkdir(parents=True)
    atomic_write_tsv(COMPARISON/'all_version_target_metrics_v2_2_20261107.tsv',rows,list(rows[0]))
    primary=[r for r in rows if r['model_version']=='v2.2' and r['target_endpoint'] in ['md_test','soppe_test']]
    atomic_write_tsv(COMPARISON/'primary_12_test_scores_v2_2_20261107.tsv',primary,list(primary[0]));assert len(primary)==12
    matrix=[r for r in rows if r['target_endpoint'] in ['md_test','md_test_full100ns_reference']]
    atomic_write_tsv(COMPARISON/'md_2x2_all_six_routes_v2_2_20261107.tsv',matrix,list(matrix[0]));assert len(matrix)==24
    atomic_write_tsv(COMPARISON/'paired_per_protein_new_minus_old_v2_2_20261107.tsv',all_pairs,list(all_pairs[0]))
    fields=list(dict.fromkeys(k for r in statistics for k in r))
    atomic_write_tsv(COMPARISON/'paired_group_statistics_v2_2_20261107.tsv',statistics,fields)
    atomic_write_json(COMPARISON/'prediction_identity_and_compatibility_v2_2_20261107.json',{'prediction_digests':digests,'score_compatibility':compat,'SOPPE_only_control':control})
    plot_results(rows,statistics)
    result={'status':'PASS','step':6,'run_tag':T,'created_at':dt.datetime.now().astimezone().isoformat(),
        'primary_scores':12,'MD_2x2_cells':24,'all_version_target_scores':len(rows),'paired_protein_rows':len(all_pairs),
        'paired_metric_summaries':len(statistics),'RMSE_multiple_testing_family_size':18,'statistics':policy['statistics'],
        'prediction_transforms':False,'new_training_or_epoch_selection':False,'Experimental10_opened':False,
        'maximum_scoring_compatibility_difference':max(r['maximum_metric_difference'] for r in compat),'SOPPE_only_control':control}
    atomic_write_json(COMPARISON/'step6_comparison_completion_v2_2_20261107.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def plot_results(rows,statistics):
    experiments=list(dict.fromkeys(r['experiment_id'] for r in rows));targets=['md_test','soppe_test','md_test_full100ns_reference']
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,target in zip(axes,targets):
        lookup={(r['experiment_id'],r['model_version']):r for r in rows if r['target_endpoint']==target};x=np.arange(6)
        ax.bar(x-.18,[lookup[e,'v2.1']['macro_rmse'] for e in experiments],width=.36,label='v2.1 fixed')
        ax.bar(x+.18,[lookup[e,'v2.2']['macro_rmse'] for e in experiments],width=.36,label='v2.2 retrained')
        ax.set_xticks(x,experiments,rotation=65,ha='right');ax.set(title=target,ylabel='Protein-record macro RMSE');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(COMPARISON/('version_comparison_same_targets_v2_2_20261107.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,exp in zip(axes.flat,experiments):
        subset={(r['model_version'],r['target_endpoint']):r for r in rows if r['experiment_id']==exp}
        for version,marker in [('v2.1','o'),('v2.2','s')]:
            ax.plot([0,1],[subset[version,t]['macro_rmse'] for t in ['md_test_full100ns_reference','md_test']],marker=marker,label=version)
        ax.set_xticks([0,1],['Full100ns target','5tau_c target']);ax.set(title=exp,ylabel='Macro RMSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same predictions; target changes along each line. Compare versions within a column.');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(COMPARISON/('md_2x2_by_route_v2_2_20261107.'+ext),dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
