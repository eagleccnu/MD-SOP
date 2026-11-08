"""Immutable recipe + data + model acceptance gates, with no label loading."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import yaml

PROJECT=Path('/data/jiangbin/SOPPCL/MD-SOP')
RUN_TAG='v2_2_20261107'
CONFIG_ROOT=PROJECT/'config'/RUN_TAG
MD_LABEL_VERSION='md_ired_5tauc_windowmean_v2_2'
RUNTIME_PATH=CONFIG_ROOT/'resolved_train_v2_2.yaml'


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def read_yaml(path):return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def data_bound_recipe():
    """Acceptance-only intermediate; never sufficient to launch training."""
    binding_path=CONFIG_ROOT/'data_binding_v2_2.yaml'
    binding=read_yaml(binding_path)
    for field,hashfield in [('trainer_manifest','trainer_manifest_sha256'),('dataset_index','dataset_index_sha256'),
        ('feature_index','feature_index_sha256'),('data_acceptance_path','data_acceptance_sha256'),
        ('data_access_policy','data_access_policy_sha256'),('evaluator_whitelist','evaluator_whitelist_sha256'),
        ('label_release_path','label_release_sha256')]:
        if sha256_file(binding[field])!=binding[hashfield]:raise ValueError('Data binding hash mismatch: '+field)
    reference=binding['scientific_recipe']
    if sha256_file(reference['protocol_path'])!=reference['protocol_sha256']:raise ValueError('Frozen protocol changed')
    if sha256_file(reference['common_train_path'])!=reference['common_train_sha256']:raise ValueError('Frozen recipe changed')
    acceptance=json.loads(Path(binding['data_acceptance_path']).read_text(encoding='utf-8'))
    if acceptance['status']!='PASS' or acceptance['step']!=3:raise ValueError('Data acceptance is not PASS')
    cfg=copy.deepcopy(read_yaml(reference['common_train_path']))
    if cfg['run_tag']!=RUN_TAG or cfg['plan_version']!='v2.2':raise ValueError('Wrong version')
    cfg['data_access']['trainer_manifest_sha256']=binding['trainer_manifest_sha256']
    cfg['data_access']['future_manifest_binding_status']='bound_from_step3'
    cfg['runtime_binding']={'data_binding_path':str(binding_path),'data_binding_sha256':sha256_file(binding_path),
        'data_access_policy_path':binding['data_access_policy'],'data_access_policy_sha256':binding['data_access_policy_sha256'],
        'model_acceptance_sha256':None,'execution_kind':'acceptance_only'}
    return cfg


def load_runtime_config(path=RUNTIME_PATH):
    path=Path(path).resolve()
    if path!=RUNTIME_PATH:raise ValueError('Only the resolved v2.2 runtime config may launch formal work')
    cfg=read_yaml(path);expected=data_bound_recipe()
    for field,value in expected.items():
        if field in ['runtime_binding','dataset_binding','execution_gate']:continue
        if cfg.get(field)!=value:raise ValueError('Runtime differs from frozen scientific/data setting: '+field)
    rb=cfg['runtime_binding'];binding_path=CONFIG_ROOT/'model_binding_v2_2.yaml'
    if Path(rb['model_binding_path'])!=binding_path or sha256_file(binding_path)!=rb['model_binding_sha256']:
        raise ValueError('Model binding mismatch')
    model_binding=read_yaml(binding_path)
    if model_binding['status']!='PASS' or not model_binding['training_launch_ready']:
        raise ValueError('Model/trainer acceptance not complete')
    if rb['data_binding_sha256']!=expected['runtime_binding']['data_binding_sha256']:
        raise ValueError('Wrong data snapshot')
    if model_binding['data_binding_sha256']!=rb['data_binding_sha256']:raise ValueError('Model/data mismatch')
    if sha256_file(model_binding['model_acceptance_path'])!=model_binding['model_acceptance_sha256']:
        raise ValueError('Acceptance audit hash mismatch')
    audit=json.loads(Path(model_binding['model_acceptance_path']).read_text(encoding='utf-8'))
    if audit['status']!='PASS' or audit['step']!=4:raise ValueError('Acceptance status invalid')
    if rb['model_acceptance_sha256']!=model_binding['model_acceptance_sha256']:raise ValueError('Wrong acceptance identity')
    manifest_path=Path(model_binding['code_manifest_path'])
    if sha256_file(manifest_path)!=model_binding['code_manifest_sha256']:raise ValueError('Code manifest mismatch')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for item in manifest['files']:
        code_path=Path(item['path'])
        if not code_path.resolve().is_relative_to(PROJECT/'code'/RUN_TAG):raise ValueError('Code escaped version root')
        if sha256_file(code_path)!=item['sha256']:raise ValueError('Accepted code changed: '+str(code_path))
    if cfg['label_versions']['md_ired']!=MD_LABEL_VERSION:raise ValueError('Wrong MD target definition')
    if not cfg['execution_gate'].get('model_acceptance_verified'):raise ValueError('Execution gate is closed')
    return cfg


def runtime_identity(config):
    return {'model_version':'v2.2','run_tag':RUN_TAG,'md_label_version':MD_LABEL_VERSION,
        'data_binding_sha256':config['runtime_binding']['data_binding_sha256'],
        'model_acceptance_sha256':config['runtime_binding']['model_acceptance_sha256'],
        'trainer_manifest_sha256':config['data_access']['trainer_manifest_sha256'],
        'code_manifest_sha256':sha256_file(PROJECT/'data'/RUN_TAG/'audits/step4_model_runtime/code_manifest_v2_2.json')}


def validate_resume_checkpoint(saved,config,config_sha256,experiment_id):
    if saved.get('format')!='md_sop_training_checkpoint_v2_2':raise ValueError('Cross-version resume format rejected')
    if saved.get('experiment_id')!=experiment_id or saved.get('config_sha256')!=config_sha256:
        raise ValueError('Resume experiment/config mismatch')
    for field,value in runtime_identity(config).items():
        if saved.get(field)!=value:raise ValueError('Resume identity mismatch: '+field)


def require_frozen_models(path,endpoint):
    """No evaluation label is opened by this gate; require all six accepted bests."""
    cfg=load_runtime_config();path=Path(path).resolve()
    stage=7 if endpoint=='experimental10' else 6
    expected=PROJECT/'data'/RUN_TAG/'audits'/('step7_preexperimental10' if stage==7 else 'step6_pretest')/'frozen_models_v2_2.json'
    if path!=expected:raise ValueError('Wrong phase-specific model freeze path')
    registry=json.loads(path.read_text(encoding='utf-8'))
    if registry['status']!='PASS' or registry['step']!=stage or registry['run_tag']!=RUN_TAG:
        raise ValueError('Model freeze is not valid for this phase')
    if endpoint not in registry['allowed_endpoints']:raise ValueError('Endpoint not enabled by frozen decision')
    if registry['runtime_config_sha256']!=sha256_file(RUNTIME_PATH):raise ValueError('Runtime changed after model freeze')
    if set(registry['experiments'])!=set(cfg['experiments']):raise ValueError('All six models must be frozen together')
    for exp,rec in registry['experiments'].items():
        expected_best=Path(cfg['model_root'])/exp/cfg['artifacts']['best_checkpoint_name']
        if Path(rec['checkpoint'])!=expected_best or sha256_file(expected_best)!=rec['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed: '+exp)
    return cfg,registry
