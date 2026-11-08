"""Experimental10 adapters and comparison; reuse the frozen step6 scorer/stats."""
from __future__ import annotations
from collections import defaultdict
import html,json,math
from pathlib import Path
import sys
import numpy as np

P=Path('/data/jiangbin/SOPPCL/MD-SOP');T='v2_2_20261107';CODE=P/'code'/T
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'step6_evaluation'))
from step6_comparison_v2_2 import (METRIC_NAMES,read_tsv,parse_predictions,prediction_digest,score_predictions,
    write_scored_endpoint,paired_records,grouped_pair_statistics,sign_flip_p,holm_adjust,
    atomic_write_json,atomic_write_tsv,atomic_write_gzip_tsv,assert_score_compatible)


def load_legacy_predictions(path,records,expected_models):
    lookup={r['record_id']:r for r in records};grouped=defaultdict(lambda:defaultdict(list))
    for row in read_tsv(path):grouped[row['model_id']][row['record_id']].append(row)
    assert set(grouped)==set(expected_models),'Historical model set changed'
    result={}
    for model,by_record in grouped.items():
        assert set(by_record)==set(lookup),'Historical protein set changed';pred={}
        for rid,record in lookup.items():
            rows=sorted(by_record[rid],key=lambda r:int(r['residue_index_1based']))
            assert [int(r['residue_index_1based']) for r in rows]==list(range(1,len(record['target'])+1)),'Historical residue axis changed'
            assert ''.join(r['amino_acid'] for r in rows)==record['sequence'],'Historical sequence differs'
            assert all(r['sequence_sha256']==record['sequence_sha256'] and r['chain_id']==record['chain_id']
                and r['joint_homology_group']==record['joint_homology_group'] for r in rows),'Historical identity differs'
            mask=np.asarray([r['qc_eligible']=='True' for r in rows]);assert np.array_equal(mask,record['qc_mask']),'Historical mask differs'
            truth=np.asarray([float(r['target']) if r['target'] else 0. for r in rows],dtype=np.float32)
            assert np.array_equal(truth[mask],record['target'][mask]),'Historical target differs'
            values=np.asarray([float(r['prediction']) for r in rows]);assert np.isfinite(values).all()
            pred[rid]=values
        result[model]=pred
    return result


def verify_original_historical_source(path,records,predictions,columns):
    grouped=defaultdict(list)
    for row in read_tsv(path):grouped[row['protein_id']].append(row)
    assert set(grouped)=={r['protein_id'] for r in records}
    for record in records:
        rows=sorted(grouped[record['protein_id']],key=lambda r:int(r['seq_position']))
        assert [int(r['seq_position']) for r in rows]==list(range(1,len(record['target'])+1))
        assert ''.join(r['aa'] for r in rows)==record['sequence']
        mask=np.asarray([r['label_available']=='1' for r in rows]);assert np.array_equal(mask,record['qc_mask'])
        old_y=np.asarray([float(r['experimental_s2']) if keep else 0. for r,keep in zip(rows,mask)])
        assert np.allclose(old_y[mask],record['target'][mask],atol=1e-7,rtol=0)
        for model,column in columns.items():
            raw=np.asarray([float(r[column]) for r in rows],dtype=np.float32)
            assert np.array_equal(raw,np.asarray(predictions[model][record['record_id']],dtype=np.float32)),'Historical raw predictions differ'
    return {'status':'PASS','full_length_positions':sum(len(r['target']) for r in records),'raw_columns':columns,
        'prediction_clipping_or_smoothing':False}


def assert_legacy_scores(per_record,summary,model,old_per_record,old_summary):
    old={r['record_id']:r for r in old_per_record if r['model_id']==model};maximum=0.
    for row in per_record:
        reference=old[row['record_id']]
        assert reference['sequence_sha256']==row['sequence_sha256'] and reference['joint_homology_group']==row['joint_homology_group']
        for metric in METRIC_NAMES:
            a=float(reference[metric]) if reference[metric] else np.nan;b=float(row[metric])
            if np.isnan(a):assert np.isnan(b);continue
            error=abs(a-b);maximum=max(maximum,error);assert error<1e-8,(model,metric,error)
    reference=next(r for r in old_summary if r['model_id']==model)
    for prefix,aggregation in [('macro','protein_record_macro'),('pooled','pooled_residue'),('homology_group_macro','homology_group_macro')]:
        for metric in METRIC_NAMES:
            a=float(reference[prefix+'_'+metric]) if reference[prefix+'_'+metric] else np.nan;b=summary[aggregation][metric]
            if np.isnan(a):assert np.isnan(b);continue
            error=abs(a-b);maximum=max(maximum,error);assert error<1e-8,(model,prefix,metric,error)
    return maximum


def flat_summary(model,version,experiment,summary):
    result={'model_id':model,'model_version':version,'experiment_id':experiment,
        **{k:summary[k] for k in ['record_count','unique_sequence_count','joint_homology_group_count','complete_residue_count','eligible_residue_count']},
        'prediction_sha256':summary['prediction_sha256'],'evaluation_input_sha256':summary['evaluation_input_sha256']}
    for prefix,key in [('macro','protein_record_macro'),('pooled','pooled_residue'),('unique_sequence_macro','unique_sequence_macro'),('homology_group_macro','homology_group_macro')]:
        result.update({prefix+'_'+m:summary[key][m] for m in METRIC_NAMES})
    result.update({m+'_valid_records':summary['protein_record_macro'][m+'_valid_records'] for m in ['pcc','spearman']})
    return result


def paired_comparisons(scores,experiments,historical,policy):
    specifications=[('v2_1__'+e,'v2_2__'+e,'version_pair',e) for e in experiments]
    specifications += [(h,'v2_2__'+e,'historical_reference',e) for e in experiments for h in historical]
    pairs=[];stats=[]
    for i,(left,right,kind,exp) in enumerate(specifications):
        records=paired_records(scores[left][0],scores[right][0],exp,'experimental10')
        comparison_id=right+'__minus__'+left
        for row in records:row.update({'comparison_id':comparison_id,'comparison_kind':kind,'reference_model_id':left,'new_model_id':right})
        pairs.extend(records)
        for mi,metric in enumerate([*METRIC_NAMES,'abs_bias']):
            result=grouped_pair_statistics(records,metric,policy['bootstrap_replicates'],policy['bootstrap_seed']+i*10+mi)
            group_values=result.pop('group_values',None)
            row={'comparison_id':comparison_id,'comparison_kind':kind,'experiment_id':exp,'reference_model_id':left,
                'new_model_id':right,'metric':metric,**result}
            if metric=='rmse':
                p,method,n=sign_flip_p(group_values,policy['sign_flip_replicates'],policy['sign_flip_seed']+i,policy['exact_sign_flip_max_groups'])
                row.update({'group_sign_flip_p':p,'sign_flip_method':method,'sign_flip_draws':n})
            stats.append(row)
    rmse=[r for r in stats if r['metric']=='rmse'];assert len(rmse)==24
    for row,value in zip(rmse,holm_adjust([r['group_sign_flip_p'] for r in rmse])):row['holm_p_family_24_RMSE_comparisons']=value
    return pairs,stats


def plot_results(root,records,predictions,scores,experiments,historical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=root/'figures';curves=figures/'protein_curves';curves.mkdir(parents=True)
    for r in records:
        x=np.arange(1,len(r['target'])+1);truth=np.where(r['qc_mask'],r['target'],np.nan)
        fig,axes=plt.subplots(3,2,figsize=(14,10),sharex=True,sharey=True)
        all_values=np.concatenate([predictions[m][r['record_id']] for m in predictions]+[r['target'][r['qc_mask']]])
        limits=(min(-.05,float(all_values.min())-.03),max(1.05,float(all_values.max())+.03))
        for ax,exp in zip(axes.flat,experiments):
            ax.plot(x,truth,'ko-',markersize=2,linewidth=.8,label='Experiment (gaps preserved)')
            ax.plot(x,predictions['v2_1__'+exp][r['record_id']],'--',color='#d58222',linewidth=1.15,label='v2.1 fixed')
            ax.plot(x,predictions['v2_2__'+exp][r['record_id']],color='#1a65a0',linewidth=1.15,label='v2.2 fixed')
            ax.set(title=exp,ylabel='S2 (raw prediction)',xlabel='Sequence position (1-based)',ylim=limits);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(r['protein_id']+' chain '+r['chain_id']+': '+str(len(x))+' residues, '+str(int(r['qc_mask'].sum()))+' experimental labels')
        fig.tight_layout()
        for ext in ['png','svg']:fig.savefig(curves/(r['protein_id']+'.'+ext),dpi=150)
        plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(20,7),sharey=True)
    colors=['#111111','#8a4e9b','#ee8a32']
    for ax,r in zip(axes.flat,records):
        x=np.arange(1,len(r['target'])+1);ax.plot(x,np.where(r['qc_mask'],r['target'],np.nan),'ko',markersize=2,label='Experiment')
        for model,color in zip(historical,colors):ax.plot(x,predictions[model][r['record_id']],linewidth=1,color=color,label=model.replace('historical_',''))
        ax.set(title=r['protein_id'],xlabel='Sequence position',ylabel='Raw S2');ax.grid(alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=9)
    fig.suptitle('Three historical frozen pipelines: reference context, not matched retraining',y=1.01);fig.tight_layout(rect=[0,0,1,.94])
    for ext in ['png','svg']:fig.savefig(figures/('historical_reference_curves.'+ext),dpi=150,bbox_inches='tight')
    plt.close(fig)
    models=list(scores);fig,axes=plt.subplots(1,3,figsize=(19,6))
    for ax,metric in zip(axes,['rmse','mae','pcc']):
        colors=['#1a65a0' if m.startswith('v2_2__') else '#d58222' if m.startswith('v2_1__') else '#888888' for m in models]
        ax.bar(np.arange(len(models)),[scores[m][1]['protein_record_macro'][metric] for m in models],color=colors)
        ax.set_xticks(np.arange(len(models)),[m.replace('historical_','hist_') for m in models],rotation=75,ha='right',fontsize=8)
        ax.set(title='Protein macro '+metric.upper(),ylabel=metric.upper());ax.grid(axis='y',alpha=.2)
    fig.suptitle('All 15 frozen models; same 10 proteins and same 709 scored positions');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(figures/('all_15_models_overview.'+ext),dpi=160)
    plt.close(fig)


def render_html(root,rows,records):
    headings=['model_id','macro_rmse','macro_mae','macro_pcc','macro_spearman','macro_bias']
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]) if k=='model_id' else format(float(r[k]),'.6f'))+'</td>' for k in headings)+'</tr>' for r in rows)
    gallery=''.join('<section><h3>'+r['protein_id']+' / '+r['chain_id']+'</h3><img src="figures/protein_curves/'+r['protein_id']+'.png" alt="'+r['protein_id']+'逐残基预测曲线"></section>' for r in records)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MD-SOP v2.2 实验10蛋白评价</title>
<style>body{max-width:1350px;margin:30px auto;padding:0 24px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#24313d}h1,h2,h3{color:#174e70}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cad6de;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto}.note{background:#eef5fa;border-left:4px solid #3d7da5;padding:12px}section{margin:35px 0}code{font-size:14px}footer{border-top:1px solid #ccd;padding-top:12px}</style>
<h1>MD-SOP v2.2：Experimental10固定模型评价</h1><p>六个v2.2模型、六个v2.1模型与三种历史方法，共15个模型；同一10蛋白、8个同源组、982个完整链位置、709个固定有效位置。所有预测原始、未裁剪、未平滑、未校准。</p>
<p class="note">这是历史复用基准，不是全新盲测；仅一个训练seed。历史方法训练池/划分不同，原始SOPPCL已知存在1条完全相同序列、5条联合同源组重叠，故作为背景参考，不作为完全匹配的受控比较。不得据测试成绩重新选择模型或mask。</p>
<h2>主要指标</h2><p>逐蛋白等权macro；bias=prediction−target。RMSE/MAE越低越好，PCC/Spearman越高越好；bias须看距零大小。相关性不能代替绝对误差。</p><div style="overflow-x:auto"><table><thead><tr>'''+''.join('<th>'+k+'</th>' for k in headings)+'''</tr></thead><tbody>'''+body+'''</tbody></table></div>
<h2>15模型总览</h2><img src="figures/all_15_models_overview.png" alt="15模型指标总览">
<h2>新旧版本逐蛋白曲线</h2><p>每图六面板对应六条固定路线；橙色虚线为v2.1，蓝色实线为v2.2，黑色为实验值。横轴是完整权威序列位置，不是PDB残基编号。实验缺失处保持空缺，不补零、不插值；预测值仍覆盖完整链。</p>'''+gallery+'''
<h2>历史方法曲线（仅背景参考）</h2><img src="figures/historical_reference_curves.png" alt="历史方法逐残基曲线">
<h2>可复核产物</h2><ul><li><a href="all_15_models_summary.tsv">15模型汇总</a></li><li><a href="all_models_per_protein_metrics.tsv">逐蛋白指标</a></li><li><a href="paired_per_protein_differences.tsv">逐蛋白配对差</a></li><li><a href="paired_group_statistics.tsv">分组配对统计</a></li></ul>
<p>配对差为新模型−参考模型，按8个联合同源组bootstrap 20000次；RMSE采用256种精确组符号翻转，24项预声明比较统一Holm校正。CI为未调整的描述性区间，记录等权与组等权估计量分别报告；参见本步骤MD说明报告中的具体解读。</p><footer>本页为步骤7专项评价及离线曲线浏览页，不是步骤8最终三端点研究总报告。</footer></html>'''
    (root/'experimental10_comparison_v2_2_20261107.html').write_text(page,encoding='utf-8')
