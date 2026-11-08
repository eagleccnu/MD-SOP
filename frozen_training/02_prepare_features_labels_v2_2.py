#!/usr/bin/env python3
"""Step3 only: new NPZ/indices/policies/data binding; no model/training/evaluation.

Reads immutable v2.1 feature components, preserves all NPZ arrays except MD y,
preserves record ordering, and binds the step2 window-label publication.
"""
from __future__ import annotations
import base64
import collections
import copy
import csv
import datetime as dt
import grp
import hashlib
import io
import json
import os
from pathlib import Path
import pwd
import stat
import zipfile

import numpy as np
import pandas as pd
import yaml

P=Path('/data/jiangbin/SOPPCL/MD-SOP');TAG='v2_2_20261107';OLD='v2_1_20261007'
C=P/'config'/TAG;D=P/'data'/TAG;OD=P/'data'/OLD;CODE=P/'code'/TAG
A=D/'audits/step3_data_preparation';M=D/'manifests'
RELEASE=D/'label_releases/md_ired_5tauc_windowmean_v2_2'
MD_VERSION='md_ired_5tauc_windowmean_v2_2';S_VERSION='unchanged_frozen_v2_1_SOPPE_targets'
EXP_VERSION='unchanged_709_position_historical_mask';FULL_VERSION='md_ired_full100ns_three_replica_mean_v2_1'
AUX='md_test_full100ns_reference'
EXPECTED={'md_train':(1070,928,157192,149299,149299),'md_validation':(153,133,21926,20753,0),
          'md_test':(136,123,19916,18939,0),'soppe_train':(2188,2187,260344,260191,260191),
          'soppe_validation':(273,270,32619,32619,0),'soppe_test':(273,269,32675,32673,0),
          'experimental10':(10,10,982,709,0)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()


def digest(data):return hashlib.sha256(data).hexdigest()
def js(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def ys(value):return yaml.safe_dump(value,allow_unicode=True,sort_keys=False,width=110).encode('utf-8')
def rows(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def ts(items):
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(items[0]),delimiter='\t',lineterminator='\n')
    w.writeheader();w.writerows(items);return out.getvalue().encode('utf-8')


def original_assets_unchanged(registry):
    for r in registry:
        p=Path(r['remote_path']);st=p.stat()
        assert sha(p)==r['sha256'] and st.st_size==int(r['bytes']) and st.st_mtime_ns==int(r['mtime_ns']),str(p)
        assert pwd.getpwuid(st.st_uid).pw_name==r['owner'] and grp.getgrgid(st.st_gid).gr_name==r['group']
        assert format(stat.S_IMODE(st.st_mode),'04o')==r['mode']


def label_destination(endpoint,record_id):
    if endpoint in ['md_train','md_validation','soppe_train','soppe_validation']:
        source='md' if endpoint.startswith('md_') else 'soppe'
        role='train' if endpoint.endswith('_train') else 'validation'
        return D/'development/labels'/source/role/(record_id+'.npz')
    return D/'evaluation'/endpoint/'labels'/(record_id+'.npz')


def record_manifest_path(endpoint):
    return D/'development/manifests'/(endpoint+'.tsv') if endpoint.endswith(('_train','_validation')) else D/'evaluation'/endpoint/'manifest.tsv'


def feature_index_path(endpoint):
    return D/'development/feature_indices'/(endpoint+'.tsv') if endpoint.endswith(('_train','_validation')) else D/'evaluation'/endpoint/'feature_index.tsv'


def build(payload):
    assert pwd.getpwuid(os.geteuid()).pw_name=='amax' and P.resolve()==P
    assert not A.exists() and not (C/'data_binding_v2_2.yaml').exists()
    assert not (D/'manifests/step3_dataset_index.tsv').exists()
    assert not list((D/'development').rglob('*.npz')) and not list((D/'evaluation').rglob('*.npz'))
    one=json.loads((D/'audits/step1_protocol/step1_protocol_acceptance_v2_2_20261107.json').read_text(encoding='utf-8'))
    two=json.loads((D/'audits/step2_label_release/label_replacement_audit_v2_2.json').read_text(encoding='utf-8'))
    assert one['status']==two['status']=='PASS'
    assert sha(C/'protocol_v2_2.yaml')==one['protocol_sha256']
    assert sha(C/'common_train_v2_2.yaml')==one['common_train_recipe_sha256']
    registry=rows(D/'manifests/v2_1_reference_registry.tsv');original_assets_unchanged(registry)
    assert sha(D/'manifests/v2_1_reference_registry.tsv')==one['reference_registry_sha256']
    release_manifest=RELEASE/'label_release_manifest_v2_2.json'
    assert sha(release_manifest)==two['release_manifest_sha256']
    release=json.loads(release_manifest.read_text(encoding='utf-8'))
    assert release['status']=='PASS' and not release['direct_trainer_access']
    for r in release['files']:assert sha(r['path'])==r['sha256']
    targets_path=RELEASE/'md_ired_5tauc_targets_v2_2.parquet'
    targets=pd.read_parquet(targets_path)
    assert targets.label_s2.dtype==np.float32 and len(targets)==188991
    target_groups={key:g.sort_values('sequence_position') for key,g in targets.groupby('record_id',sort=False)}
    assert len(target_groups)==1359
    joint_path=OD/'manifests/joint_split_manifest_v2_1.tsv'
    joint=rows(joint_path);joint_by_id={r['record_id']:r for r in joint}
    assert len(joint_by_id)==len(joint)
    protected=rows(D/'manifests/protected_reference_manifest.tsv')
    ref_by_id={r['record_id']:r for r in protected}
    source_index=rows(OD/'manifests/step3_dataset_index.tsv')
    assert set(r['dataset_endpoint'] for r in source_index)==set(EXPECTED)
    source_endpoints={r['dataset_endpoint']:r for r in source_index}
    source_rows={e:rows(r['record_manifest']) for e,r in source_endpoints.items()}
    feature_specs={};sequence_features={};all_primary_ids=set()
    # Verify all actual feature contents once per path; no feature extraction or copying.
    for endpoint, records in source_rows.items():
        spec=source_endpoints[endpoint]
        assert sha(spec['record_manifest'])==spec['record_manifest_sha256']
        assert sha(spec['feature_index'])==spec['feature_index_sha256']
        expected=EXPECTED[endpoint]
        assert len(records)==expected[0]
        assert len(set(r['sequence_sha256'] for r in records))==expected[1]
        for r in records:
            assert r['record_id'] not in all_primary_ids
            all_primary_ids.add(r['record_id'])
            auth=joint_by_id[r['record_id']]
            assert auth['sequence_sha256']==r['sequence_sha256'] and len(auth['sequence'])==int(r['sequence_length'])
            assert digest(auth['sequence'].encode('ascii'))==r['sequence_sha256']
            assert auth['role_v2_1']==r['role_v2_1'] and auth['partition_v2_1']==r['partition_v2_1']
            assert auth['joint_homology_group']==r['joint_homology_group']
            assert r['logical_feature_dimension']=='1182' and r['logical_feature_dtype']=='float32'
            pair=(r['esmc_sha256'],r['hmm30_sha256'])
            previous=sequence_features.setdefault(r['sequence_sha256'],pair)
            assert previous==pair,'Different canonical feature hashes for identical sequence'
            for kind,dimension in [('esmc',1152),('hmm30',30)]:
                path=r[kind+'_path'];spec=(r[kind+'_sha256'],int(r['sequence_length']),dimension)
                assert path not in feature_specs or feature_specs[path]==spec
                feature_specs[path]=spec
    feature_audit=[]
    for path,(expected,length,dimension) in sorted(feature_specs.items()):
        p=Path(path);st=p.stat();observed=sha(p)
        assert observed==expected,'Feature content changed: '+str(p)
        array=np.load(p,mmap_mode='r',allow_pickle=False)
        assert array.shape==(length,dimension) and array.dtype==np.float32
        assert np.isfinite(array).all()
        feature_audit.append({'path':path,'expected_sha256':expected,'observed_sha256':observed,'bytes':st.st_size,
                              'mtime_ns':st.st_mtime_ns,'rows':length,'columns':dimension,'dtype':'float32','all_finite':True})
        del array
    contents={};new_rows={};package_transition=[];old_lookup={};md_changed=0;unchanged_copies=0
    def add(path,data):
        assert path not in contents and not path.exists(),'Existing or duplicate output '+str(path)
        contents[path]=data
    for endpoint,records in source_rows.items():
        new_rows[endpoint]=[]
        add(feature_index_path(endpoint),Path(source_endpoints[endpoint]['feature_index']).read_bytes())
        for old in records:
            rid=old['record_id'];old_lookup[rid]=old
            source=Path(old['label_path']);source_bytes=source.read_bytes()
            assert digest(source_bytes)==old['label_file_sha256']
            with np.load(io.BytesIO(source_bytes),allow_pickle=False) as z:
                arrays={name:np.array(z[name],copy=True) for name in z.files}
            seq=arrays['sequence_ascii'].astype(np.uint8).tobytes().decode('ascii')
            assert seq==joint_by_id[rid]['sequence']
            length=int(old['sequence_length']);qc=arrays['qc_mask'];positions=np.flatnonzero(qc)
            assert arrays['y'].dtype==np.float32 and arrays['y'].shape==(length,)
            for name in ['label_mask','qc_mask','raw_label_present','train_access','eval_access','loss_mask']:
                assert arrays[name].shape==(length,) and arrays[name].dtype==np.bool_
            assert arrays['length'].shape==(1,) and int(arrays['length'][0])==length
            assert np.array_equal(arrays['loss_mask'],qc&arrays['train_access'])
            assert np.isfinite(arrays['y'][qc]).all()
            assert int(qc.sum())==int(old['qc_label_count']) and int(arrays['loss_mask'].sum())==int(old['loss_label_count'])
            row=dict(old);row.update({'role_v2_2':old['role_v2_1'],'partition_v2_2':old['partition_v2_1'],
                                     'plan_version':'v2.2','run_tag':TAG,'endpoint_kind':'primary',
                                     'source_label_path_v2_1':old['label_path'],'source_label_sha256_v2_1':old['label_file_sha256'],
                                     'feature_index_path':str(feature_index_path(endpoint)),
                                     'authority_chain_id':ref_by_id.get(rid,{}).get('chain_id',old['chain_id'])})
            is_md=endpoint.startswith('md_')
            if is_md:
                frame=target_groups[rid]
                assert frame.dataset_endpoint.eq(endpoint).all() and frame.role_v2_2.eq(old['role_v2_1']).all()
                assert np.array_equal(frame.sequence_position.to_numpy()-1,positions)
                assert np.array_equal(frame.old_label_s2.to_numpy(),arrays['y'][positions])
                assert frame.sequence_sha256.eq(old['sequence_sha256']).all()
                expected_y=arrays['y'].copy();expected_y[positions]=frame.label_s2.to_numpy(dtype=np.float32)
                arrays['y']=expected_y
                out=io.BytesIO();np.savez_compressed(out,**arrays);new_bytes=out.getvalue()
                version=MD_VERSION;md_changed+=int(digest(new_bytes)!=old['label_file_sha256'])
                row['upstream_label_path']=str(targets_path);row['upstream_label_sha256']=two['targets_parquet_sha256']
                row['label_release_manifest']=str(release_manifest);row['label_release_manifest_sha256']=two['release_manifest_sha256']
            else:
                new_bytes=source_bytes;version=EXP_VERSION if endpoint=='experimental10' else S_VERSION
                row['label_release_manifest']='';row['label_release_manifest_sha256']=''
                unchanged_copies+=1
            # Fresh package roundtrip: masks, sequence and all other arrays unchanged.
            with np.load(io.BytesIO(source_bytes),allow_pickle=False) as old_z, np.load(io.BytesIO(new_bytes),allow_pickle=False) as new_z:
                assert old_z.files==new_z.files
                for name in old_z.files:
                    if name!='y':assert np.array_equal(old_z[name],new_z[name])
                if is_md:
                    assert np.array_equal(new_z['y'][positions],target_groups[rid].label_s2.to_numpy(dtype=np.float32))
                    assert np.array_equal(new_z['y'][~qc],old_z['y'][~qc])
                else:assert source_bytes==new_bytes
            destination=label_destination(endpoint,rid);add(destination,new_bytes)
            row.update({'label_path':str(destination),'label_file_sha256':digest(new_bytes),'label_version':version})
            new_rows[endpoint].append(row)
            package_transition.append({'dataset_endpoint':endpoint,'endpoint_kind':'primary','record_id':rid,
                'old_path':old['label_path'],'old_sha256':old['label_file_sha256'],'new_path':str(destination),'new_sha256':digest(new_bytes),
                'target_policy':'replace_MD_effective_y_only' if is_md else 'byte_identical_copy',
                'all_non_y_arrays_unchanged':True,'qc_label_count':int(old['qc_label_count']),'loss_label_count':int(old['loss_label_count'])})
    # Seven primary endpoint manifests; source row order is intentionally retained.
    primary_index=[]
    for endpoint in sorted(new_rows):
        records=new_rows[endpoint]
        record_data=ts(records);rpath=record_manifest_path(endpoint);add(rpath,record_data)
        fpath=feature_index_path(endpoint);feature_data=contents[fpath]
        expected=EXPECTED[endpoint]
        actual=(len(records),len({r['sequence_sha256'] for r in records}),sum(int(r['sequence_length']) for r in records),
                sum(int(r['qc_label_count']) for r in records),sum(int(r['loss_label_count']) for r in records))
        assert actual==expected,(endpoint,actual,expected)
        primary_index.append({'dataset_endpoint':endpoint,'record_count':actual[0],'unique_sequence_count':actual[1],
            'full_sequence_residues':actual[2],'raw_label_count':sum(int(r['raw_label_count']) for r in records),
            'qc_label_count':actual[3],'loss_label_count':actual[4], 'record_manifest':str(rpath),'record_manifest_sha256':digest(record_data),
            'feature_index':str(fpath),'feature_index_sha256':digest(feature_data),
            'access_scope':'training_or_validation' if endpoint.endswith(('_train','_validation')) else 'final_evaluation_only',
            'endpoint_kind':'primary','label_version':records[0]['label_version']})
    # Auxiliary old MD-test: separate identity/index, never counted among seven primary endpoints.
    aux_features=rows(source_endpoints['md_test']['feature_index'])
    for r in aux_features:
        r['dataset_endpoint']=AUX;r['feature_key']=AUX+'::'+r['sequence_sha256']
    add(feature_index_path(AUX),ts(aux_features))
    aux_rows=[]
    for old in source_rows['md_test']:
        rid=old['record_id'];raw=Path(old['label_path']).read_bytes();assert digest(raw)==old['label_file_sha256']
        destination=label_destination(AUX,rid);add(destination,raw)
        r=dict(old);r.update({'role_v2_2':'test','partition_v2_2':'test','plan_version':'v2.2','run_tag':TAG,
            'dataset_endpoint':AUX,'feature_key':AUX+'::'+old['sequence_sha256'],'feature_index_path':str(feature_index_path(AUX)),
            'label_path':str(destination),'label_file_sha256':digest(raw),'label_version':FULL_VERSION,
            'endpoint_kind':'auxiliary_reference','allowed_label_access':'auxiliary_full100ns_evaluation_only',
            'source_label_path_v2_1':old['label_path'],'source_label_sha256_v2_1':old['label_file_sha256'],
            'label_release_manifest':'','label_release_manifest_sha256':'','authority_chain_id':old['chain_id']})
        aux_rows.append(r)
        package_transition.append({'dataset_endpoint':AUX,'endpoint_kind':'auxiliary_reference','record_id':rid,
            'old_path':old['label_path'],'old_sha256':old['label_file_sha256'],'new_path':str(destination),'new_sha256':digest(raw),
            'target_policy':'byte_identical_full100ns_reference','all_non_y_arrays_unchanged':True,
            'qc_label_count':int(old['qc_label_count']),'loss_label_count':0})
    add(record_manifest_path(AUX),ts(aux_rows))
    aux_index=[{'dataset_endpoint':AUX,'record_count':136,'unique_sequence_count':123,'full_sequence_residues':19916,
        'raw_label_count':18939,'qc_label_count':18939,'loss_label_count':0,'record_manifest':str(record_manifest_path(AUX)),
        'record_manifest_sha256':digest(contents[record_manifest_path(AUX)]),'feature_index':str(feature_index_path(AUX)),
        'feature_index_sha256':digest(contents[feature_index_path(AUX)]),'endpoint_kind':'auxiliary_reference','label_version':FULL_VERSION,
        'access_scope':'final_evaluation_only'}]
    add(M/'step3_dataset_index.tsv',ts(primary_index));add(M/'auxiliary_dataset_index_v2_2.tsv',ts(aux_index))
    global_features=(OD/'feature_index.tsv').read_bytes();add(D/'feature_index.tsv',global_features)
    # Keep all record ordering, including the zero-supervision SOPPCL__1zgg record.
    new_by_id={r['record_id']:r for items in new_rows.values() for r in items}
    old_trainer=rows(OD/'development/manifests/trainer_access_manifest.tsv')
    trainer=[new_by_id[r['record_id']] for r in old_trainer]
    assert len(trainer)==3684
    assert [r['record_id'] for r in trainer]==[r['record_id'] for r in old_trainer]
    assert all(r['role_v2_2'] in ['train','validation'] for r in trainer)
    assert sum(r['role_v2_2']=='train' and int(r['qc_label_count'])>0 for r in trainer)==3257
    trainer_path=D/'development/manifests/trainer_access_manifest.tsv';add(trainer_path,ts(trainer))
    all_primary=[r for endpoint in sorted(new_rows) for r in new_rows[endpoint]]
    add(M/'dataset_records_v2_2.tsv',ts(all_primary))
    split_copy=[]
    for r in joint:
        s=dict(r);s.update({'role_v2_2':r['role_v2_1'],'partition_v2_2':r['partition_v2_1'],
                           'endpoint_v2_2':r['endpoint_v2_1'],'derived_plan_version':'v2.2','derived_run_tag':TAG})
        split_copy.append(s)
    add(M/'joint_split_manifest_v2_2.tsv',ts(split_copy))
    active=[r for r in split_copy if r['role_v2_2'] in ['train','validation','test']]
    protected_groups={r['joint_homology_group'] for r in protected}
    assert not protected_groups.intersection(r['joint_homology_group'] for r in active)
    for key in ['joint_homology_group','sequence_sha256']:
        roles=collections.defaultdict(set)
        for r in active:roles[r[key]].add(r['role_v2_2'])
        assert all(len(v)==1 for v in roles.values())
    policy={'schema_version':'1.0','model_version':'v2.2','run_tag':TAG,
        'trainer_manifest':str(trainer_path),'trainer_manifest_sha256':digest(contents[trainer_path]),
        'development_label_root':str(D/'development/labels'),'evaluation_root':str(D/'evaluation'),
        'master_label_release_root':str(D/'label_releases'),'allowed_roles':['train','validation'],
        'allowed_sources':['md_ired','soppe_legacy'],'role_column':'role_v2_2','require_legacy_role_alias_equality':True,
        'label_version_by_source':{'md_ired':MD_VERSION,'soppe_legacy':S_VERSION},
        'fail_closed_before_any_label_archive_open':True,'test_labels_forbidden_to_trainer':True,
        'this_is_application_policy_not_os_permission_isolation':True}
    policy_path=D/'development/manifests/data_access_policy_v2_2.json';add(policy_path,js(policy))
    whitelist={'schema_version':2,'model_version':'v2.2','run_tag':TAG,'endpoints':{},
        'policy':'Only explicitly selected evaluator endpoints may be opened after all six checkpoints are frozen. Never pass this whitelist to trainer.',
        'protected_experimental_positions':709,'scoring_allowed_before_checkpoint_freeze':False}
    for entry in primary_index+aux_index:
        e=entry['dataset_endpoint']
        if e.endswith(('_train','_validation')):continue
        whitelist['endpoints'][e]={'record_manifest':entry['record_manifest'],'record_manifest_sha256':entry['record_manifest_sha256'],
            'feature_index':entry['feature_index'],'feature_index_sha256':entry['feature_index_sha256'],
            'record_count':entry['record_count'],'qc_label_count':entry['qc_label_count'],
            'endpoint_kind':entry['endpoint_kind'],'label_version':entry['label_version']}
    add(D/'evaluation/final_evaluator_whitelist.json',js(whitelist))
    guard={};exec(compile(payload['guard_script'],'access_manifest_guard_v2_2.py','exec'),guard)
    for row in trainer:guard['validate_training_row'](row,policy)
    cases=[]
    def forbidden(name,row):
        try:guard['validate_training_row'](row,policy)
        except guard['AccessViolationError']:cases.append({'case':name,'pass':True});return
        raise RuntimeError('Access test unexpectedly allowed: '+name)
    example=dict(next(r for r in trainer if r['source']=='md_ired' and r['role_v2_2']=='train'))
    for name,changes in [
        ('test_role',{'role_v2_2':'test'}),('reference_role',{'role_v2_2':'reference_experimental10'}),
        ('role_alias_mismatch',{'role_v2_1':'validation'}),('auxiliary_endpoint',{'dataset_endpoint':AUX}),
        ('evaluation_label_path',{'label_path':str(label_destination('md_test','FAKE'))}),
        ('master_release_path',{'label_path':str(RELEASE/'md_ired_5tauc_targets_v2_2.parquet')}),
        ('old_version_label_path',{'label_path':old_trainer[0]['label_path']}),
        ('another_record_label_path',{'label_path':str(label_destination('md_train','MD__WRONG_RECORD'))}),
        ('wrong_MD_target_version',{'label_version':FULL_VERSION}),
        ('fake_validation_loss',{'role_v2_2':'validation','role_v2_1':'validation','partition_v2_2':'validation','partition_v2_1':'validation',
                                 'dataset_endpoint':'md_validation','train_access':'False','eval_access':'True',
                                 'allowed_label_access':'early_stopping_and_checkpoint_selection','loss_label_count':'1',
                                 'label_path':str(label_destination('md_validation',example['record_id']))})]:
        mutated=dict(example);mutated.update(changes);forbidden(name,mutated)
    add(CODE/'access_manifest_guard_v2_2.py',payload['guard_script'].encode('utf-8'))
    add(A/'access_guard_row_tests_v2_2.json',js(cases))
    add(A/'feature_component_audit_v2_2.tsv',ts(feature_audit))
    add(A/'label_package_transition_v2_2.tsv',ts(package_transition))
    split_audit={'status':'PASS','source_path':str(joint_path),'source_sha256':sha(joint_path),
        'record_ids_sequences_groups_roles_and_order_preserved':True,'source_records':len(joint),'derived_records':len(split_copy),
        'zero_group_or_exact_sequence_overlap_across_active_partitions':True,'protected_reference_overlap_with_development':0,
        'trainer_record_order_identical':True,'primary_records':4103,'auxiliary_reference_records':136,
        'auxiliary_repeats_are_not_new_independent_proteins':True}
    add(A/'split_equivalence_audit_v2_2.json',js(split_audit))
    config={'step':3,'model_version':'v2.2','run_tag':TAG,'source_dataset_index':str(OD/'manifests/step3_dataset_index.tsv'),
        'source_joint_split':str(joint_path),'target_release_manifest':str(release_manifest),'target_release_manifest_sha256':two['release_manifest_sha256'],
        'features':'read_only_reuse_all_original_canonical_components','unchanged_targets':'byte_copy_SOPPE_experimental_and_auxiliary_reference',
        'MD_replacement':'new_y_at_exact_original_qc_positions_only','preserve_source_record_order':True,
        'expected_primary_datasets':{k:list(v) for k,v in EXPECTED.items()},'start_training':False}
    add(C/'step3_features_labels_v2_2.yaml',ys(config))
    summary={'status':'PASS','step':3,'run_tag':TAG,'model_version':'v2.2','created_at':dt.datetime.now().astimezone().isoformat(),
        'execution_user':'amax','primary_endpoints':7,'primary_records':4103,'auxiliary_reference_records':136,
        'total_new_npz_files':4239,'MD_window_target_packages':1359,'MD_package_hashes_changed':md_changed,
        'primary_unchanged_label_byte_copies':unchanged_copies,'auxiliary_full100ns_byte_copies':136,
        'development_manifest_records':3684,'supervised_train_records':3257,'validation_records':426,
        'zero_supervision_records':['SOPPCL__1zgg'],'counts_by_endpoint':primary_index,
        'auxiliary_dataset_index':aux_index,'unique_feature_component_paths':len(feature_audit),
        'feature_component_bytes_verified':sum(r['bytes'] for r in feature_audit),
        'feature_component_hash_shape_dtype_and_finiteness_verified':True,
        'feature_indices_byte_identical_to_v2_1':True,'identical_sequences_have_identical_canonical_features':True,
        'all_non_y_arrays_preserved':True,'MD_only_original_qc_y_replaced':True,'SOPPE_and_experimental_bytes_unchanged':True,
        'trainer_and_endpoint_order_preserved':True,'all_split_roles_preserved':True,'row_access_rejection_tests':len(cases),
        'protected_reference_count':36,'experimental10_mask_positions':709,
        'frozen_protocol_sha256':one['protocol_sha256'],'frozen_training_recipe_sha256':one['common_train_recipe_sha256'],
        'label_release_manifest_sha256':two['release_manifest_sha256'],'label_targets_parquet_sha256':two['targets_parquet_sha256'],
        'reference_registry_sha256':one['reference_registry_sha256'],'original_assets_to_recheck':len(registry),
        'trainer_manifest_path':str(trainer_path),'trainer_manifest_sha256':digest(contents[trainer_path]),
        'dataset_index_path':str(M/'step3_dataset_index.tsv'),'dataset_index_sha256':digest(contents[M/'step3_dataset_index.tsv']),
        'feature_index_path':str(D/'feature_index.tsv'),'feature_index_sha256':digest(contents[D/'feature_index.tsv']),
        'data_access_policy_path':str(policy_path),'data_access_policy_sha256':digest(contents[policy_path]),
        'evaluator_whitelist_path':str(D/'evaluation/final_evaluator_whitelist.json'),
        'evaluator_whitelist_sha256':digest(contents[D/'evaluation/final_evaluator_whitelist.json']),
        'master_label_table_excluded_from_trainer_access':True,'training_started':False,'test_prediction_scores_computed':False,
        'model_trainer_acceptance_status':'pending_step4','direct_training_allowed':False,
        'next_step':'step4_adapt_and_validate_models_data_loader_trainer_evaluators'}
    return contents,summary,registry,feature_audit,guard


def publish(payload):
    contents,summary,registry,feature_audit,guard=build(payload)
    if payload.get('preflight_only'):return {'summary':summary,'preflight_only':True}
    contents[CODE/'02_prepare_features_labels_v2_2.py']=payload['script'].encode('utf-8')
    os.umask(0o002);roots=[P/x/TAG for x in ['code','config','data','model','results','logs','tmp']]
    created=[]
    def safe_mkdir(path):
        if path.exists():assert path.is_dir() and path.resolve()==path;return
        assert any(r==path or r in path.parents for r in roots)
        safe_mkdir(path.parent);path.mkdir(mode=0o2775)
    def write(path,data):
        assert any(r in path.parents for r in roots) and not path.exists()
        safe_mkdir(path.parent)
        with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        created.append(path)
    # No writes occur before all source features/labels and in-memory outputs validate.
    for path,data in contents.items():write(path,data)
    policy_path=D/'development/manifests/data_access_policy_v2_2.json'
    actual_trainer=guard['load_trainer_rows'](policy_path)
    assert len(actual_trainer)==3684
    assert len(guard['load_trainer_rows'](policy_path,roles=['train'],drop_zero_supervision=True))==3257
    for name,kwargs in [('request_test_role',{'roles':['test']}),('request_reference_role',{'roles':['reference_experimental10']}),
                        ('open_test_manifest',{'manifest_path':record_manifest_path('md_test')}),
                        ('open_aux_reference_manifest',{'manifest_path':record_manifest_path(AUX)})]:
        try:guard['load_trainer_rows'](policy_path,**kwargs)
        except guard['AccessViolationError']:continue
        raise RuntimeError('Manifest-level guard unexpectedly allowed '+name)
    # Read every newly written package; no model or loss computation is run.
    for endpoint in list(EXPECTED)+[AUX]:
        for r in rows(record_manifest_path(endpoint)):
            path=Path(r['label_path']);assert sha(path)==r['label_file_sha256']
            old=Path(r['source_label_path_v2_1'])
            with np.load(path,allow_pickle=False) as z,np.load(old,allow_pickle=False) as o:
                for name in o.files:
                    if name!='y':assert np.array_equal(z[name],o[name])
                if endpoint.startswith('md_') and endpoint!=AUX:
                    expected=contents[path]
                    assert digest(expected)==sha(path)
                else:assert path.read_bytes()==old.read_bytes()
    original_assets_unchanged(registry)
    for item in feature_audit:
        st=Path(item['path']).stat()
        assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns']
    assert sha(C/'protocol_v2_2.yaml')==summary['frozen_protocol_sha256']
    assert sha(C/'common_train_v2_2.yaml')==summary['frozen_training_recipe_sha256']
    summary['all_original_assets_unchanged_after_step3']=True
    summary['frozen_scientific_recipes_unchanged']=True
    summary['manifest_access_rejection_tests']=4
    summary['all_new_packages_readback_verified']=True
    acceptance_path=A/'step3_data_acceptance_v2_2_20261107.json'
    write(acceptance_path,js(summary))
    binding={'schema_version':'1.0','model_version':'v2.2','run_tag':TAG,'status':'data_bound_pending_step4_acceptance',
        'training_allowed':False,'scientific_recipe':{'protocol_path':str(C/'protocol_v2_2.yaml'),'protocol_sha256':summary['frozen_protocol_sha256'],
            'common_train_path':str(C/'common_train_v2_2.yaml'),'common_train_sha256':summary['frozen_training_recipe_sha256']},
        'label_release_path':str(RELEASE/'label_release_manifest_v2_2.json'),'label_release_sha256':summary['label_release_manifest_sha256'],
        'label_targets_parquet_sha256':summary['label_targets_parquet_sha256'],
        'trainer_manifest':summary['trainer_manifest_path'],'trainer_manifest_sha256':summary['trainer_manifest_sha256'],
        'dataset_index':summary['dataset_index_path'],'dataset_index_sha256':summary['dataset_index_sha256'],
        'feature_index':summary['feature_index_path'],'feature_index_sha256':summary['feature_index_sha256'],
        'data_acceptance_path':str(acceptance_path),'data_acceptance_sha256':sha(acceptance_path),
        'data_access_policy':summary['data_access_policy_path'],'data_access_policy_sha256':summary['data_access_policy_sha256'],
        'evaluator_whitelist':summary['evaluator_whitelist_path'],'evaluator_whitelist_sha256':summary['evaluator_whitelist_sha256'],
        'model_acceptance_sha256':None,'model_acceptance_status':'pending_step4',
        'step4_resolution_policy':'Keep this data snapshot immutable. Add a separate model_binding_v2_2.yaml with actual acceptance hashes, then resolve runtime config from scientific recipe + data binding + model binding.',
        'resolved_training_config_path':str(C/'resolved_train_v2_2.yaml')}
    write(C/'data_binding_v2_2.yaml',ys(binding))
    report='# MD-SOP v2.2 步骤3：样本封装与访问隔离验收\n\n'
    report+='- 状态：**PASS**\n- WQQ时间：`'+summary['created_at']+'`\n- Run tag：`'+TAG+'`；创建用户：`amax`。\n\n'
    report+='## 七个主要端点\n\n| 端点 | 记录 | 唯一序列 | 完整残基 | QC标签 | loss标签 |\n|---|---:|---:|---:|---:|---:|\n'
    for e in summary['counts_by_endpoint']:
        report+='| '+e['dataset_endpoint']+' | '+str(e['record_count'])+' | '+str(e['unique_sequence_count'])+' | '+str(e['full_sequence_residues'])+' | '+str(e['qc_label_count'])+' | '+str(e['loss_label_count'])+' |\n'
    report+='\n主要端点共4103个新版本NPZ；另将136个旧MD-test全长轨迹包原字节复制到独立辅助目录，总计**4239个NPZ**。辅助包不算新增独立样本，不进入训练或主终点计数。\n\n'
    report+='## 标签、特征与顺序\n\n'
    report+='1359个MD包仅替换原有效位置的y为步骤2的float32窗口目标；所有其他数组及无效位置占位值不变。2744个S-OPPE/实验包与136个辅助包为原字节副本。'
    report+='3684行训练/验证访问清单与原版本行顺序相同；可监督训练记录3257条，验证426条。SOPPCL__1zgg保留但仍为零监督，训练采样必须排除。\n\n'
    report+='实际复核'+str(summary['unique_feature_component_paths'])+'个唯一特征文件、'+str(summary['feature_component_bytes_verified'])+'字节，哈希、[L,1152]/[L,30]形状、float32及有限值通过；'
    report+='ESM-C在前、HMM在后组成逻辑1182维输入。主全局和七端点特征索引为原字节副本；没有重提取特征、复制大矩阵或更改数据库。权威序列、角色、同源组与mask均保持原版本。\n\n'
    report+='## 访问隔离\n\n'
    report+='开发标签只写入development/labels；MD-test、S-test、Experimental10及full100ns辅助目标分别位于evaluation下。训练清单仅有train/validation，'
    report+='访问守卫验证精确清单路径/哈希、角色别名、标签版本、来源、label路径边界和loss计数，禁止evaluation、旧版本标签和label_releases主表。'
    report+=str(summary['row_access_rejection_tests'])+'个行级和4个清单级拒绝测试通过；守卫在打开标签前拒绝这些请求。受信任的制备程序按方案读取源标签以封装数据，不是模型评价。该守卫是数据访问基础设施，不是已完成的正式DataLoader/训练器；步骤4仍需集成并验证。'
    report+='同一amax用户可以手动访问来源，因此这里是物理分目录＋程序白名单，不宣称OS级权限沙箱。\n\n'
    report+='## 数据绑定与后续边界\n\n'
    report+='已生成data_binding_v2_2.yaml，绑定真实标签发布、训练清单、数据索引、特征索引和数据验收哈希；model_acceptance_sha256明确为null，training_allowed=false。'
    report+='两份步骤1冻结配方不变。步骤4新增模型验收绑定文件及resolved_train运行配置，不编辑冻结科学配方或本次数据快照。\n\n'
    report+='原4265个登记资产的哈希/元数据核对未变；完整数据包与两个发布数据表保留在WQQ。本地同步代码、清单、绑定、报告与哈希，不复制大特征或数千NPZ。'
    report+='本次没有加载回归模型、执行梯度更新或计算测试预测指标。下一步是步骤4适配并验收模型、DataLoader、训练与评价代码。\n'
    write(A/'step3_data_preparation_report_v2_2_20261107.md',report.encode('utf-8'))
    checks=[]
    for path in sorted(created):
        st=path.stat();assert st.st_uid==1000 and st.st_gid==1000
        checks.append({'relative_path':str(path.relative_to(P)),'bytes':st.st_size,'sha256':sha(path)})
    write(A/'step3_artifact_sha256.tsv',ts(checks))
    return collect()


def collect():
    summary=json.loads((A/'step3_data_acceptance_v2_2_20261107.json').read_text(encoding='utf-8'))
    manifest=rows(A/'step3_artifact_sha256.tsv');out=io.BytesIO();n=0
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for r in manifest:
            path=P/r['relative_path'];assert path.resolve()==path and TAG in path.parts and sha(path)==r['sha256']
            if path.suffix=='.npz':continue
            z.write(path,str(path.relative_to(P)));n+=1
        z.write(A/'step3_artifact_sha256.tsv',str((A/'step3_artifact_sha256.tsv').relative_to(P)))
    return {'summary':summary,'remote_artifact_count':len(manifest)+1,'local_metadata_file_count':n+1,
        'data_binding_sha256':sha(C/'data_binding_v2_2.yaml'),'new_npz_retained_on_wqq':True,
        'archive_base64':base64.b64encode(out.getvalue()).decode('ascii')}
