#!/usr/bin/env python3
"""Step4 acceptance only: short train-set smoke tests, never formal training."""
from __future__ import annotations
import ast
import copy
import csv
import datetime as dt
import gzip
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import types
import unittest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from mdsop_runtime_v2_2 import PROJECT,RUN_TAG,CONFIG_ROOT,MD_LABEL_VERSION,data_bound_recipe,runtime_identity,validate_resume_checkpoint,sha256_file
from mdsop_dataset_v2_2 import AccessControlledProteinDataset,ManifestAccessPolicy,collate_protein_records,AccessViolationError
from mdsop_loss_v2_2 import record_macro_smooth_l1,mixed_source_record_macro_smooth_l1
from mdsop_training_v2_2 import (make_access_policy,move_batch_to_device,CyclingShuffleStream,
    checkpoint_payload,atomic_torch_save,restore_rng_state,optimizer_to_device,save_training_curves)
from model_bilstm_window_v2_2 import load_model_checkpoint,state_dict_sha256,EXPECTED_PARAMETER_COUNTS
from mdsop_evaluation_v2_2 import (EvaluationAccessPolicy,EvaluationProteinDataset,EvaluationAccessError,record_metrics,METRIC_NAMES,STEP6_ENDPOINTS,
    make_evaluation_loader,atomic_write_json,atomic_write_tsv,atomic_write_gzip_tsv)
from mdsop_experimental10_v2_2 import Experimental10AccessPolicy,Experimental10Dataset

A=PROJECT/'data'/RUN_TAG/'audits/step4_model_runtime'
CODE=PROJECT/'code'/RUN_TAG
SMOKE=A/'smoke'


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')


def load_module(path,name):
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(Path(path).read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    return module


def optimizer(model,cfg):
    t=cfg['training']
    return torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'],betas=tuple(t['betas']),eps=t['eps'])


def scheduler(opt,cfg):
    s=cfg['scheduler']
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode=s['mode'],patience=s['patience'],factor=s['factor'],min_lr=s['min_learning_rate'])


def step(model,opt,batch,mixed=False):
    model.train();opt.zero_grad(set_to_none=True)
    prediction=model(batch['x'],batch['lengths'],batch['residue_mask'])
    loss=(mixed_source_record_macro_smooth_l1(prediction,batch['y'],batch['loss_mask'],batch['sources']).loss if mixed
          else record_macro_smooth_l1(prediction,batch['y'],batch['loss_mask']).loss)
    assert torch.isfinite(loss);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);assert torch.isfinite(norm)
    opt.step();return float(loss.detach().cpu()),float(norm.detach().cpu())


def main():
    assert not (CONFIG_ROOT/'model_binding_v2_2.yaml').exists(),'Accepted step4 run already exists'
    cfg=data_bound_recipe();policy=make_access_policy(cfg)
    A.mkdir(parents=True,exist_ok=True);SMOKE.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    checks=[];details={}
    def check(name,function):
        try:
            result=function();checks.append({'name':name,'status':'PASS'});details[name]=result
            print('PASS '+name,flush=True);return result
        except Exception:
            checks.append({'name':name,'status':'FAIL'});details[name]={'traceback':traceback.format_exc()}
            print('FAIL '+name+'\n'+traceback.format_exc(),flush=True)
            dump(A/'step4_working_results.json',{'status':'FAIL','checks':checks,'details':details})
            raise
    def units():
        cmd=[sys.executable,'-m','unittest','discover','-s',str(CODE/'tests'),'-p','test_*_v2_2.py','-v']
        done=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},timeout=180)
        (A/'unit_tests_v2_2.log').write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
        assert done.returncode==0,done.stdout+done.stderr
        return {'returncode':0,'output_tail':done.stderr[-1500:]}
    check('unit_tests',units)
    def math_equivalence():
        paths=[('model_bilstm_window_v2_1.py','model_bilstm_window_v2_2.py',
            ['MDSoPBiLSTMRegressor.__init__','MDSoPBiLSTMRegressor.encode','MDSoPBiLSTMRegressor.forward_from_hidden','MDSoPBiLSTMRegressor.forward']),
            ('mdsop_loss_v2_1.py','mdsop_loss_v2_2.py',['record_macro_smooth_l1','mixed_source_record_macro_smooth_l1']),
            ('mdsop_training_v2_1.py','mdsop_training_v2_2.py',['CyclingShuffleStream','make_training_loader','make_validation_loader','summarize_validation_records']),
            ('03_train_md_sop_v2_1.py','03_train_md_sop_v2_2.py',['build_epoch_batches','run_validation'])]
        def select(path,name):
            nodes=ast.parse(path.read_text(encoding='utf-8')).body
            for part in name.split('.'):
                node=next(n for n in nodes if getattr(n,'name',None)==part);nodes=getattr(node,'body',[])
            return ast.dump(node,include_attributes=False)
        names=[]
        for old,new,functions in paths:
            for name in functions:
                assert select(PROJECT/'code/v2_1_20261007'/old,name)==select(CODE/new,name),name
                names.append(name)
        def optimization_loop(path):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple)
                        and [getattr(x,'id','') for x in n.target.elts]==['epoch_step','batch_cpu'])
        old=optimization_loop(PROJECT/'code/v2_1_20261007/03_train_md_sop_v2_1.py');new=optimization_loop(CODE/'03_train_md_sop_v2_2.py')
        assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
        return {'unchanged_function_AST':names,'complete_inner_optimization_loop_AST_identical':True}
    check('frozen_math_and_optimizer_loop',math_equivalence)
    datasets={s:AccessControlledProteinDataset(policy,roles=['train'],sources=[s],verify_component_hashes=True) for s in ['md_ired','soppe_legacy']}
    def data_counts_and_versions():
        result={}
        for source,n,val in [('md_ired',1070,153),('soppe_legacy',2187,273)]:
            assert len(datasets[source])==n
            validation=AccessControlledProteinDataset(policy,roles=['validation'],sources=[source])
            assert len(validation)==val
            result[source]={'train':n,'validation':val}
        for source in datasets:
            datasets[source].rows=tuple(datasets[source].rows)
        return result
    check('real_development_counts',data_counts_and_versions)
    def access_rejections():
        count=0
        for roles in [['test'],['reference_experimental10']]:
            try:AccessControlledProteinDataset(policy,roles=roles)
            except AccessViolationError:count+=1
            else:raise AssertionError('Forbidden role was allowed')
        bad=ManifestAccessPolicy(PROJECT/'data'/RUN_TAG/'evaluation/md_test/manifest.tsv','0'*64,policy.development_label_root,policy.evaluation_root)
        try:AccessControlledProteinDataset(bad,roles=['train'])
        except AccessViolationError:count+=1
        else:raise AssertionError('Test manifest was allowed')
        binding=__import__('yaml').safe_load((CONFIG_ROOT/'data_binding_v2_2.yaml').read_text())
        white=Path(binding['evaluator_whitelist']);root=Path(cfg['data_access']['evaluation_root'])
        try:
            EvaluationProteinDataset(EvaluationAccessPolicy(white,binding['evaluator_whitelist_sha256'],root),endpoint='md_test',
                manifest_path=root/'md_test/manifest.tsv',manifest_sha256='0'*64,expected_source='md_ired')
        except (EvaluationAccessError,ValueError,FileNotFoundError):count+=1
        else:raise AssertionError('Production test opened without model freeze')
        try:
            Experimental10Dataset(Experimental10AccessPolicy(white,binding['evaluator_whitelist_sha256'],root),
                manifest_path=root/'experimental10/manifest.tsv',manifest_sha256='0'*64,expected_ids=(),expected_chains={})
        except (EvaluationAccessError,ValueError,FileNotFoundError):count+=1
        else:raise AssertionError('Experimental10 opened without freeze')
        # The actual trainer CLI must reject an unbound scientific recipe before output creation.
        done=subprocess.run([sys.executable,str(CODE/'03_train_md_sop_v2_2.py'),'--config',str(CONFIG_ROOT/'common_train_v2_2.yaml'),
            '--experiment-id','A_MD_w1','--device','cuda:0','--validate-only'],capture_output=True,text=True)
        assert done.returncode!=0 and 'resolved v2.2 runtime config' in done.stderr
        return {'denied_role_manifest_and_test_gates':count,'actual_trainer_rejects_unbound_recipe':True,'new_test_label_archives_opened':0}
    check('actual_loader_and_evaluator_access_rejections',access_rejections)
    def synthetic_evaluation_loaders():
        root=SMOKE/'synthetic_evaluation';root.mkdir(exist_ok=True)
        toy_x=root/'toy_esmc.npy';toy_h=root/'toy_hmm.npy'
        np.save(toy_x,np.zeros((3,1152),dtype=np.float32));np.save(toy_h,np.zeros((3,30),dtype=np.float32))
        import hashlib
        sequence='AAA';seqsha=hashlib.sha256(sequence.encode()).hexdigest()
        versions={'md_test':MD_LABEL_VERSION,'soppe_test':'unchanged_frozen_v2_1_SOPPE_targets',
                  'md_test_full100ns_reference':'md_ired_full100ns_three_replica_mean_v2_1','experimental10':'unchanged_709_position_historical_mask'}
        white={'endpoints':{}}
        specs={}
        for endpoint,version in versions.items():
            directory=root/endpoint;labels=directory/'labels';labels.mkdir(parents=True,exist_ok=True)
            label=labels/'synthetic.npz';qc=np.array([True,False,True])
            np.savez_compressed(label,y=np.array([.2,0.,.7],dtype=np.float32),qc_mask=qc,label_mask=qc,raw_label_present=qc,
                train_access=np.zeros(3,dtype=bool),eval_access=np.ones(3,dtype=bool),loss_mask=np.zeros(3,dtype=bool),
                length=np.array([3],dtype=np.int64),sequence_ascii=np.frombuffer(b'AAA',dtype=np.uint8))
            r=dict(datasets['md_ired'].rows[0]);r.update({'record_id':'SYNTHETIC__toy','protein_id':'synthetic','pdb_id':'synthetic','chain_id':'A',
                'dataset_endpoint':endpoint,'sequence_sha256':seqsha,'sequence_length':'3','joint_homology_group':'SYNTHETIC_ONLY',
                'role_v2_1':'test','role_v2_2':'test','partition_v2_1':'test','partition_v2_2':'test',
                'esmc_path':str(toy_x),'esmc_sha256':sha256_file(toy_x),'hmm30_path':str(toy_h),'hmm30_sha256':sha256_file(toy_h),
                'label_path':str(label),'label_file_sha256':sha256_file(label),'raw_label_count':'2','qc_label_count':'2','loss_label_count':'0',
                'train_access':'False','eval_access':'True','source':'soppe_legacy' if endpoint=='soppe_test' else 'md_ired',
                'allowed_label_access':'auxiliary_full100ns_evaluation_only' if endpoint=='md_test_full100ns_reference' else 'final_evaluation_only_after_checkpoint_freeze',
                'label_version':version})
            if endpoint=='experimental10':r.update({'source':'historical_test','role_v2_1':'reference_experimental10','role_v2_2':'reference_experimental10',
                'partition_v2_1':'protected','partition_v2_2':'protected','allowed_label_access':'final_experimental10_evaluation_only',
                'qc_rule':'frozen_historical_experimental10_mask'})
            manifest=directory/'manifest.tsv'
            with manifest.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(r),delimiter='\t');w.writeheader();w.writerow(r)
            white['endpoints'][endpoint]={'record_manifest':str(manifest),'record_manifest_sha256':sha256_file(manifest)}
            specs[endpoint]=(manifest,sha256_file(manifest),r['source'])
        whitelist=root/'final_evaluator_whitelist.json';dump(whitelist,white)
        evaluator=load_module(CODE/'04_evaluate_md_sop_v2_2.py','synthetic_only_evaluator')
        for endpoint,(manifest,ms,source) in specs.items():
            if endpoint=='experimental10':
                data=Experimental10Dataset(Experimental10AccessPolicy(whitelist,sha256_file(whitelist),root),
                    manifest_path=manifest,manifest_sha256=ms,expected_ids=('synthetic',),expected_chains={'synthetic':'A'})
            else:
                data=EvaluationProteinDataset(EvaluationAccessPolicy(whitelist,sha256_file(whitelist),root,STEP6_ENDPOINTS),
                    endpoint=endpoint,manifest_path=manifest,manifest_sha256=ms,expected_source=source)
            sample=data[0];assert sample['x'].shape==(3,1182) and int(sample['qc_mask'].sum())==2
            for window in [1,15]:
                model,_=load_model_checkpoint(Path(cfg['initialization']['window_'+str(window)]['checkpoint']))
                identity={**runtime_identity(cfg),'experiment_id':'SYNTHETIC_ONLY_w'+str(window),'endpoint':endpoint,
                    'target_label_version':versions[endpoint],'runtime_config_sha256':'synthetic_not_formal',
                    'checkpoint_file_sha256':'untrained_initialization','evaluation_manifest_sha256':ms}
                per_record,residues,summary=evaluator.predict_records(model,make_evaluation_loader(data,batch_size=1),torch.device('cpu'),identity)
                assert summary['record_count']==1 and summary['eligible_residue_count']==2 and len(residues)==3
                target=directory/('writer_w'+str(window));target.mkdir(exist_ok=True)
                atomic_write_json(target/'summary.json',{**identity,**summary})
                atomic_write_tsv(target/'per_protein_metrics.tsv',per_record,list(per_record[0]))
                atomic_write_gzip_tsv(target/'per_residue_predictions.tsv.gz',residues,list(residues[0]))
                with gzip.open(target/'per_residue_predictions.tsv.gz','rt',encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
                assert rows[0]['model_version']=='v2.2' and rows[0]['code_manifest_sha256']==identity['code_manifest_sha256']
        return {'synthetic_endpoints_verified':list(specs),'prediction_and_all_writers_verified_for_both_heads':True,'production_test_label_archives_opened':0}
    check('synthetic_positive_evaluation_loaders',synthetic_evaluation_loaders)
    short={}
    for source,dataset in datasets.items():
        indices=sorted(range(len(dataset.rows)),key=lambda i:(int(dataset.rows[i]['sequence_length']),dataset.rows[i]['record_id']))[:2]
        short[source]=[dataset[i] for i in indices]
    def architecture_and_gpu():
        old=load_module(PROJECT/'code/v2_1_20261007/model_bilstm_window_v2_1.py','frozen_v21_model_reference')
        out=[]
        for window in [1,15]:
            device=torch.device('cuda:'+str(0 if window==1 else 1));torch.cuda.set_device(device)
            model,payload=load_model_checkpoint(Path(cfg['initialization']['window_'+str(window)]['checkpoint']),expected_window_size=window)
            assert model.parameter_count()==EXPECTED_PARAMETER_COUNTS[window]
            original=old.MDSoPBiLSTMRegressor(window_size=window);original.load_state_dict(payload['model_state_dict'])
            model.to(device).eval();original.to(device).eval()
            torch.manual_seed(123);x=torch.randn(2,11,1182,device=device);lengths=torch.tensor([11,7]);mask=torch.arange(11,device=device)[None,:]<lengths.to(device)[:,None]
            with torch.no_grad():a=model(x,lengths,mask);b=original(x,lengths,mask)
            err=float((a-b).abs().max());assert err<=1e-7
            out.append({'window':window,'device':str(device),'parameters':model.parameter_count(),'maximum_v2_1_v2_2_prediction_difference':err,
                        'initial_state_sha256':state_dict_sha256(model.state_dict())})
        return out
    check('architecture_initialization_and_both_GPUs',architecture_and_gpu)
    def route_smoke():
        out=[]
        for exp,spec in cfg['experiments'].items():
            window=spec['window_size'];device=torch.device('cuda:'+str(0 if window==1 else 1));torch.cuda.set_device(device)
            model,_=load_model_checkpoint(Path(cfg['initialization']['window_'+str(window)]['checkpoint']),expected_window_size=window)
            model.to(device);opt=optimizer(model,cfg)
            items=[item for source in spec['train_sources'] for item in short[source]]
            batch=move_batch_to_device(collate_protein_records(items),device)
            before=state_dict_sha256(model.state_dict());loss,norm=step(model,opt,batch,len(spec['train_sources'])==2)
            assert state_dict_sha256(model.state_dict())!=before
            out.append({'experiment':exp,'smoke_optimizer_steps':1,'records':len(items),'loss':loss,'gradient_norm':norm,'formal_training':False})
        return out
    check('six_routes_single_step_smoke',route_smoke)
    def resume_test():
        results=[]
        for window in [1,15]:
            # CPU uses the ordinary Torch RNG for every dropout operation. This
            # verifies bitwise optimizer continuation without altering the formal
            # cuDNN backend or claiming its internal RNN dropout state is saved.
            exp='A_MD_w'+str(window);device=torch.device('cpu')
            initial=Path(cfg['initialization']['window_'+str(window)]['checkpoint'])
            model,_=load_model_checkpoint(initial);model.to(device);opt=optimizer(model,cfg);sch=scheduler(opt,cfg)
            batch=move_batch_to_device(collate_protein_records(short['md_ired']),device)
            torch.manual_seed(987);torch.cuda.manual_seed_all(987);step(model,opt,batch)
            stream=CyclingShuffleStream(1070,20261107);stream.next_indices(3)
            saved=checkpoint_payload(experiment_id=exp,config_sha256='a'*64,model=model,optimizer=opt,scheduler=sch,
                epoch=1,global_step=1,best_metric=.5,best_epoch=1,early_stop_reference=.5,early_stop_bad_epochs=0,
                streams={'md_ired':stream},history=[],exposures={},device=device,status='smoke_only',identity=runtime_identity(cfg))
            path=SMOKE/('resume_w'+str(window)+'_v2_2_20261107.pt');atomic_torch_save(path,saved)
            next_indices=stream.next_indices(7);step(model,opt,batch);expected=state_dict_sha256(model.state_dict())
            restored=torch.load(path,map_location='cpu',weights_only=False);validate_resume_checkpoint(restored,cfg,'a'*64,exp)
            second,_=load_model_checkpoint(initial);second.to(device);o2=optimizer(second,cfg);s2=scheduler(o2,cfg)
            second.load_state_dict(restored['model_state_dict']);o2.load_state_dict(restored['optimizer_state_dict']);optimizer_to_device(o2,device)
            s2.load_state_dict(restored['scheduler_state_dict']);restore_rng_state(restored['rng_state'],device)
            resumed=CyclingShuffleStream(1070,20261107);resumed.load_state_dict(restored['stream_states']['md_ired'])
            assert resumed.next_indices(7)==next_indices
            step(second,o2,batch);assert state_dict_sha256(second.state_dict())==expected
            rejects=0
            for field,value in [('format','md_sop_training_checkpoint_v2_1'),('run_tag','v2_1_20261007'),
                                ('md_label_version','full100ns'),('data_binding_sha256','0'*64),('config_sha256','f'*64)]:
                wrong=dict(restored);wrong[field]=value
                try:validate_resume_checkpoint(wrong,cfg,'a'*64,exp)
                except ValueError:rejects+=1
                else:raise AssertionError('Wrong resume identity allowed')
            gpu=torch.device('cuda:'+str(0 if window==1 else 1));torch.cuda.set_device(gpu)
            first_gpu,_=load_model_checkpoint(initial);first_gpu.to(gpu);gpu_opt=optimizer(first_gpu,cfg);gpu_sch=scheduler(gpu_opt,cfg)
            gpu_batch=move_batch_to_device(collate_protein_records(short['md_ired']),gpu);step(first_gpu,gpu_opt,gpu_batch)
            first_gpu.eval()
            with torch.no_grad():expected_prediction=first_gpu(gpu_batch['x'],gpu_batch['lengths'],gpu_batch['residue_mask']).clone()
            gpu_payload=checkpoint_payload(experiment_id=exp,config_sha256='a'*64,model=first_gpu,optimizer=gpu_opt,scheduler=gpu_sch,
                epoch=1,global_step=1,best_metric=.5,best_epoch=1,early_stop_reference=.5,early_stop_bad_epochs=0,
                streams={'md_ired':stream},history=[],exposures={},device=gpu,status='smoke_only',identity=runtime_identity(cfg))
            gpu_path=SMOKE/('resume_gpu_w'+str(window)+'_v2_2_20261107.pt');atomic_torch_save(gpu_path,gpu_payload)
            loaded=torch.load(gpu_path,map_location='cpu',weights_only=False);validate_resume_checkpoint(loaded,cfg,'a'*64,exp)
            other_gpu,_=load_model_checkpoint(initial);other_gpu.to(gpu);other_gpu.load_state_dict(loaded['model_state_dict'])
            other_opt=optimizer(other_gpu,cfg);other_opt.load_state_dict(loaded['optimizer_state_dict']);optimizer_to_device(other_opt,gpu)
            other_sch=scheduler(other_opt,cfg);other_sch.load_state_dict(loaded['scheduler_state_dict'])
            restore_rng_state(loaded['rng_state'],gpu)
            assert torch.equal(torch.cuda.get_rng_state(gpu),loaded['rng_state']['torch_cuda'])
            assert state_dict_sha256(other_gpu.state_dict())==state_dict_sha256(first_gpu.state_dict())
            assert other_opt.param_groups[0]['lr']==gpu_opt.param_groups[0]['lr']
            for k,v in gpu_opt.state_dict()['state'].items():
                for field,value in v.items():
                    observed=other_opt.state_dict()['state'][k][field]
                    assert torch.equal(value.cpu(),observed.cpu()) if torch.is_tensor(value) else value==observed
            other_gpu.eval()
            with torch.no_grad():observed_prediction=other_gpu(gpu_batch['x'],gpu_batch['lengths'],gpu_batch['residue_mask'])
            assert torch.equal(observed_prediction,expected_prediction)
            results.append({'window':window,'CPU_exact_next_optimizer_step_state_match':True,'queue_restoration_exact':True,'wrong_identity_rejections':rejects,
                            'GPU_model_optimizer_scheduler_rng_restored':True,'GPU_eval_prediction_bitwise_equal':True,
                            'GPU_stochastic_next_optimizer_step_bitwise_guaranteed':False,
                            'GPU_resume_limitation':'Original cuDNN LSTM recurrent-dropout internal state is not serialized by the v2.1 checkpoint API; backend retained for paired-comparison fidelity.',
                            'checkpoint_path':str(path),'GPU_checkpoint_path':str(gpu_path),'never_use_for_formal_resume':True})
        return results
    check('checkpoint_rng_optimizer_and_sampler_resume',resume_test)
    def tiny_overfit():
        result=[]
        for window in [1,15]:
            device=torch.device('cuda:'+str(0 if window==1 else 1));torch.cuda.set_device(device)
            model,_=load_model_checkpoint(Path(cfg['initialization']['window_'+str(window)]['checkpoint']));model.to(device)
            opt=optimizer(model,cfg);batch=move_batch_to_device(collate_protein_records(short['md_ired']),device)
            model.eval()
            with torch.no_grad():initial=float(record_macro_smooth_l1(model(batch['x'],batch['lengths'],batch['residue_mask']),batch['y'],batch['loss_mask']).loss)
            history=[]
            for i in range(48):history.append(step(model,opt,batch)[0])
            model.eval()
            with torch.no_grad():final=float(record_macro_smooth_l1(model(batch['x'],batch['lengths'],batch['residue_mask']),batch['y'],batch['loss_mask']).loss)
            assert final<initial*.5,(window,initial,final)
            fig,ax=plt.subplots(figsize=(7.5,4));ax.plot(np.arange(1,49),history);ax.set(xlabel='Smoke optimizer step (NOT a virtual epoch)',ylabel='Train SmoothL1',
                title='SMOKE ONLY: 2 training proteins, w'+str(window));ax.grid(alpha=.2);fig.tight_layout()
            for extension in ['png','svg']:fig.savefig(SMOKE/('overfit_loss_w'+str(window)+'_v2_2_20261107.'+extension),dpi=150)
            plt.close(fig)
            result.append({'window':window,'training_records':[r['record_id'] for r in short['md_ired']],
                'optimizer_steps':48,'initial_eval_loss_on_same_training_records':initial,'final_eval_loss_on_same_training_records':final,
                'formal_optimizer_hyperparameters_used':True,'smoke_weights_discarded_for_formal_initialization':True})
        return result
    check('tiny_training_set_fits_and_loss_curves',tiny_overfit)
    def old_scorer_compatibility():
        root=PROJECT/'results/v2_1_20261007';maximum=0.;nrecords=0;tables=0
        paths=[]
        for endpoint in ['md_test','soppe_test']:
            for exp in cfg['experiments']:
                paths.append((root/endpoint/exp/'per_residue_predictions.tsv.gz',root/endpoint/exp/'per_protein_metrics.tsv',None))
        paths.append((root/'experimental10/comparison/all_models_per_residue_predictions.tsv.gz',root/'experimental10/comparison/all_models_per_protein_metrics.tsv','model_id'))
        for predictions,metrics,modelcol in paths:
            def read(path):
                op=gzip.open if str(path).endswith('.gz') else open
                with op(path,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
            reference={((r[modelcol] if modelcol else ''),r['record_id']):r for r in read(metrics)}
            grouped={}
            for row in read(predictions):grouped.setdefault(((row[modelcol] if modelcol else ''),row['record_id']),[]).append(row)
            for key,records in grouped.items():
                ordered=sorted(records,key=lambda r:int(r['residue_index_1based']))
                prediction=np.array([float(r['prediction']) for r in ordered]);target=np.array([float(r['target']) if r['target'] else 0. for r in ordered])
                mask=np.array([r['qc_eligible']=='True' for r in ordered])
                observed=record_metrics(prediction=prediction,target=target,qc_mask=mask,metadata={})
                for field in METRIC_NAMES:
                    old=float(reference[key][field]) if reference[key][field] else np.nan;new=float(observed[field])
                    if np.isnan(old):assert np.isnan(new);continue
                    error=abs(old-new);maximum=max(maximum,error);assert error<1e-8,(predictions,key,field,error)
                nrecords+=1
            tables+=1
        # Raw outputs outside [0,1] remain raw; invalid targets are masked first.
        r=record_metrics(prediction=np.array([-0.2,.4,1.2]),target=np.array([.1,np.nan,.9]),qc_mask=np.array([True,False,True]),metadata={})
        assert r['out_of_range_fraction']==1.0
        return {'frozen_prediction_tables':tables,'frozen_per_record_scores_reproduced':nrecords,'maximum_metric_difference':maximum,
            'tolerance':1e-8,'v2_2_model_test_predictions_generated':False,'uses_only_already_published_v2_1_predictions_and_targets':True}
    check('frozen_scorer_compatibility',old_scorer_compatibility)
    for sh in ['run_matrix_v2_2.sh','start_tmux_v2_2.sh']:
        subprocess.run(['bash','-n',str(CODE/sh)],check=True)
    assert not list((PROJECT/'model'/RUN_TAG).rglob('*.pt'))
    assert not [p for p in (PROJECT/'results'/RUN_TAG).rglob('*') if p.is_file()]
    result={'status':'PASS','step':4,'run_tag':RUN_TAG,'created_at':dt.datetime.now().astimezone().isoformat(),
        'checks':checks,'details':details,'formal_training_started':False,'formal_model_directory_empty':True,
        'v2_2_model_test_scoring_performed':False,'smoke_output_directory':str(SMOKE),
        'untrained_original_initializations_required_for_step5':True}
    dump(A/'step4_model_runtime_acceptance_v2_2_20261107.json',result)
    print(json.dumps({'status':'PASS','checks':len(checks),'audit':str(A/'step4_model_runtime_acceptance_v2_2_20261107.json')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
