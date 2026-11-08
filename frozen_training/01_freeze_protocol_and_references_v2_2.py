#!/usr/bin/env python3
"""Step 1 only: freeze recipes/references; never generate labels or train.

Run as amax through run_step1_from_local.py. Existing inputs are read-only.
Scientific settings are copied from the actual v2.1 training configuration.
Future dataset hashes stay unbound; steps 2--4 must provide separate binding.
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
import yaml

P = Path('/data/jiangbin/SOPPCL/MD-SOP')
OLD = 'v2_1_20261007'
TAG = 'v2_2_20261107'
DATE = '20261107'
VERSION = 'v2.2'
EXPERIMENTS = ['A_MD_w1', 'A_MD_w15', 'B_SOPPE_w1', 'B_SOPPE_w15', 'C_MIX_w1', 'C_MIX_w15']
C = P / 'config' / TAG
D = P / 'data' / TAG
A = D / 'audits/step1_protocol'
M = D / 'manifests'
CODE = P / 'code' / TAG
OD = P / 'data' / OLD
OC = P / 'config' / OLD
LABEL_VERSION = 'md_ired_5tauc_windowmean_v2_2'
EXPECTED = {
    'md_train': (1070, 928, 157192, 149299),
    'md_validation': (153, 133, 21926, 20753),
    'md_test': (136, 123, 19916, 18939),
    'soppe_train': (2188, 2187, 260344, 260191),
    'soppe_validation': (273, 270, 32619, 32619),
    'soppe_test': (273, 269, 32675, 32673),
    'experimental10': (10, 10, 982, 709),
}
EXP_CHAINS = {'1pd7': 'A', '1wrs': 'R', '1wrt': 'R', '1z9b': 'A', '2jwt': 'A',
              '2l6b': 'A', '2luo': 'A', '2m3o': 'W', '2xdi': 'A', '4aai': 'A'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_tsv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def encode_tsv(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8')


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def encode_yaml(value):
    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False, width=110).encode('utf-8')


def flatten(value, path=''):
    if isinstance(value, dict) and value:
        out = {}
        for k, v in value.items():
            out.update(flatten(v, path + '.' + str(k) if path else str(k)))
        return out
    return {path: value}


def config_diff(name, old, new):
    of, nf = flatten(old), flatten(new)
    allowed = {
        'protocol': {
            'plan_version': '版本元数据', 'protocol_id': '版本元数据', 'run_tag': '版本路径',
            'status': '冻结状态；数据发布待后续绑定', 'frozen_at': '本轮冻结时间',
            'project': '新计划路径与哈希，项目身份不变', 'paths': '新版本输出/派生清单路径',
            'authoritative_inputs': '增加只读MD窗口产物及旧版本追溯入口',
            'joint_split_policy': '复用已完成的v2.1划分，禁止重新分配，不改变实际角色',
            'model': '仅模型名称的版本号变化，架构不变',
            'training': '仅将原可选确认种子明确冻结为空；沿用已执行的单种子策略',
            'reference_protection': '仅26参考用途的版本文字，保护规则不变',
            'evaluation': '增加5τc主目标/full100ns辅助目标和同口径重评分规则',
            'step1_acceptance': '本轮验收与下一步骤标识',
            'md_label_policy': '本轮唯一科学变量：MD标签汇总定义及对应目标',
            'execution_seed_policy': '固定原实际使用的单种子、无ensemble策略',
            'frozen_reference_registry': '只读对照资产登记',
            'data_release_binding': '未来派生数据哈希独立绑定，不填造哈希',
            'change_control': '限制允许变化范围并保留历史基准使用说明',
        },
        'common_train': {
            'plan_version': '版本元数据', 'run_tag': '版本路径', 'protocol_id': '版本元数据',
            'code_root': '新版本代码路径', 'model_root': '新版本模型输出路径',
            'result_root': '新版本结果路径', 'log_root': '新版本日志路径', 'audit_root': '新版本审计路径',
            'data_access': '新数据路径及未来哈希绑定；记录数不变',
            'artifacts': '文件名增加v2.2与创建日期，不改变训练运算',
            'label_versions': '明确MD窗口标签与不变S标签身份',
            'initialization_policy': '只读复用原未训练初始化，不加载旧优化器/训练权重',
            'execution_gate': '未发布数据/未验收训练器前禁止直接启动',
            'dataset_binding': '独立绑定未来数据发布，保持本轮科学配方不变',
        },
    }[name]
    rows = []
    for key in sorted(set(of) | set(nf)):
        before = of.get(key, '__FIELD_NOT_PRESENT__')
        after = nf.get(key, '__FIELD_NOT_PRESENT__')
        if before == after:
            continue
        top = key.split('.')[0]
        if top not in allowed:
            raise RuntimeError('Unapproved configuration difference: ' + name + '.' + key)
        rows.append({'configuration': name, 'field': key,
                     'old': json.dumps(before, ensure_ascii=False, sort_keys=True),
                     'new': json.dumps(after, ensure_ascii=False, sort_keys=True),
                     'reason': allowed[top]})
    return rows


def build():
    if pwd.getpwuid(os.geteuid()).pw_name != 'amax':
        raise RuntimeError('Step1 must execute as amax')
    if P.resolve() != P:
        raise RuntimeError('Project root is not canonical')
    step0 = json.loads((D / 'audits/step0_workspace/workspace_acceptance_v2_2_20261107.json').read_text(encoding='utf-8'))
    assert step0['status'] == 'PASS' and step0['run_tag'] == TAG
    assert sha(C / 'MD-SOP v2.2 training plan.md') == step0['plan']['sha256']
    assert not (C / 'protocol_v2_2.yaml').exists(), 'Refusing to replace existing frozen protocol'
    assert not (C / 'common_train_v2_2.yaml').exists(), 'Refusing to replace existing frozen recipe'
    assert not A.exists(), 'Step1 audit directory already exists'
    registry = {}

    def register(path, category, scope='', expected='', note=''):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError('Missing reference asset: ' + str(path))
        key = str(path)
        if key in registry:
            row = registry[key]
            if expected and row['sha256'] != expected:
                raise RuntimeError('Recorded source hash mismatch: ' + key)
            return row
        observed = sha(path)
        if expected and observed != expected:
            raise RuntimeError('Source hash does not match frozen record: ' + key)
        st = path.stat()
        row = {'asset_type': category, 'scope': scope, 'remote_path': key, 'resolved_path': str(path.resolve()),
               'bytes': st.st_size, 'sha256': observed, 'expected_recorded_sha256': expected,
               'recorded_hash_verified': bool(expected), 'owner': pwd.getpwuid(st.st_uid).pw_name,
               'group': grp.getgrgid(st.st_gid).gr_name, 'mode': format(stat.S_IMODE(st.st_mode), '04o'),
               'mtime_ns': st.st_mtime_ns, 'access_for_v2_2': 'read_only', 'note': note}
        registry[key] = row
        return row

    # All v2.1 source code/configuration is frozen; no pyc/cache file is needed.
    for path in sorted((P / 'code' / OLD).rglob('*.py')):
        register(path, 'v2_1_code', note='Source only; no script executed')
    for path in sorted((P / 'config' / OLD).iterdir()):
        if path.is_file():
            register(path, 'v2_1_configuration')
    for path in sorted((OD / 'manifests').glob('*')):
        if path.is_file():
            register(path, 'v2_1_manifest')
    register(P / 'README.md', 'legacy_root_metadata')
    register(P / 'project_manifest.json', 'legacy_root_metadata')
    register(OD / 'feature_index.tsv', 'frozen_feature_index', note='Component hashes frozen in index; large feature arrays not rescanned at step1')
    register(OD / 'development/manifests/trainer_access_manifest.tsv', 'trainer_access_reference')
    old_protocol = yaml.safe_load((OC / 'protocol_v2_1.yaml').read_text(encoding='utf-8'))
    old_train = yaml.safe_load((OC / 'common_train_v2_1.yaml').read_text(encoding='utf-8'))
    decision_path = OD / 'audits/step6_pretest/seed_and_test_opening_decision.json'
    register(decision_path, 'v2_1_actual_seed_and_checkpoint_freeze')
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    assert decision['decision']['included_seeds'] == [20261007]
    assert decision['decision']['ensemble'] is False
    init_file = Path(old_train['initialization']['manifest'])
    register(init_file, 'paired_initialization_manifest')
    initialization = json.loads(init_file.read_text(encoding='utf-8'))
    assert initialization['seed'] == 20261007
    for window, key in [('w1', 'window_1'), ('w15', 'window_15')]:
        state = initialization['checkpoints'][window]
        assert state['backbone_state_sha256'] == old_train['initialization']['common_backbone_state_sha256']
        assert state['path'] == old_train['initialization'][key]['checkpoint']
        register(state['path'], 'untrained_initialization', window, old_train['initialization'][key]['file_sha256'])
    for exp in EXPERIMENTS:
        rec = decision['experiments'][exp]
        register(rec['checkpoint'], 'v2_1_best_checkpoint', exp, rec['checkpoint_sha256'], 'Reference only; forbidden as v2.2 training initialization')
        for name in ['history.tsv', 'run_manifest.json', 'best_validation_summary.json', 'run_config_snapshot.yaml']:
            register(P / 'results' / OLD / 'development' / exp / name, 'v2_1_development_reference', exp)
    # Hash prediction files only; do not parse predictions or compute any score.
    prediction_refs = []
    for endpoint in ['md_test', 'soppe_test']:
        for exp in EXPERIMENTS:
            for name in ['per_residue_predictions.tsv.gz', 'per_protein_metrics.tsv', 'summary.json']:
                path = P / 'results' / OLD / endpoint / exp / name
                category = 'frozen_predictions' if name.endswith('.gz') else 'frozen_existing_metrics'
                register(path, category, endpoint + '/' + exp, note='No re-scoring at step1')
                if name.endswith('.gz'):
                    prediction_refs.append({'endpoint': endpoint, 'experiments': exp, 'path': str(path), 'sha256': sha(path)})
    exp_predictions = P / 'results' / OLD / 'experimental10/comparison/all_models_per_residue_predictions.tsv.gz'
    row = register(exp_predictions, 'frozen_predictions', 'experimental10/all_models', note='Combined six v2.1 models plus historical references')
    prediction_refs.append({'endpoint': 'experimental10', 'experiments': ';'.join(EXPERIMENTS), 'path': str(exp_predictions), 'sha256': row['sha256']})
    for exp in EXPERIMENTS:
        for name in ['per_protein_metrics.tsv', 'summary.json']:
            register(P / 'results' / OLD / 'experimental10' / exp / name, 'frozen_existing_metrics', 'experimental10/' + exp)
    for name in ['final_results_summary.tsv', 'checkpoint_registry.tsv', 'environment_snapshot.json', 'MD-SOP-v2.1-final-report.md']:
        register(P / 'results' / OLD / 'final_report' / name, 'v2_1_final_archive')
    upstream = {
        '/data/jiangbin/iRED/results/ired_full_production_v1_20260731/aggregate/labels_by_residue.parquet': 'f81d5e226ecba8a34b615b6cc7846d951fee80724794017584f2edd99d7faba9',
        '/data/jiangbin/iRED/results/ired_full_production_v1_20260731/aggregate/residue_s2_all_runs.parquet': '6b33fbdb360763cd3b4f1459fb94af8e6136677ab4ca05accb8ef5394997912c',
        '/data/jiangbin/iRED/results/ired_full_production_v1_20260731/aggregate/run_qc.parquet': 'ac9628a7f4757e3505cad2f13101f39e537bd5efb589a9f25153927b0dccab2e',
        '/data/jiangbin/iRED/manifests/ired/ired_full_production_v1_20260731/selection.tsv': '9af548fa6bd143e18f0f81054513a1d84c6dd1eb74a7f43adc586d4a70473b4a',
    }
    for path, expected in upstream.items():
        register(path, 'immutable_iRED_source', expected=expected, note='Hash only; window labels are not generated in step1')
    for path in ['/data/jiangbin/iRED/config/ired_full_production_v1.yml', '/data/jiangbin/iRED/scripts/run_ired_production.py',
                 '/data/jiangbin/iRED/scripts/freeze_ired_full_manifest.py', '/data/jiangbin/iRED/src/iredmd/sampling.py']:
        register(path, 'iRED_label_definition_source')
    split_path = OD / 'manifests/joint_split_manifest_v2_1.tsv'
    split_sha = 'ef05bdaa7dc37399d4b4178f271588872b8df9edaeee3060d37d61874d5d9858'
    register(split_path, 'frozen_joint_split', expected=split_sha)
    split = read_tsv(split_path)
    reference_path = OD / 'manifests/protected_reference_manifest.tsv'
    references = read_tsv(reference_path)
    assert len(references) == 36
    assert collections.Counter(r['reference_class'] for r in references) == {'experimental10_historical': 10, 'experimental_nmr26_reference': 26}
    assert len({r['sequence_sha256'] for r in references}) == 35
    assert len({r['joint_homology_group'] for r in references}) == 30
    for r in references:
        assert digest(r['sequence'].encode('ascii')) == r['sequence_sha256']
        assert len(r['sequence']) == int(r['sequence_length'])
        for flag in ['protect_from_train', 'protect_from_validation', 'protect_from_md_test', 'protect_from_soppe_test']:
            assert r[flag] == 'True'
    active = [r for r in split if r['role_v2_1'] in ['train', 'validation', 'test']]
    protected_groups = {r['joint_homology_group'] for r in references}
    assert not protected_groups.intersection(r['joint_homology_group'] for r in active)
    for key in ['joint_homology_group', 'sequence_sha256']:
        roles = collections.defaultdict(set)
        for r in active:
            roles[r[key]].add(r['role_v2_1'])
        assert all(len(v) == 1 for v in roles.values())
    benchmark = Path(old_protocol['authoritative_inputs']['experimental10_benchmark_root'])
    for name in ['benchmark_manifest.tsv', 'protein_manifest.tsv', 'residue_alignment.tsv', 'provenance.json']:
        register(benchmark / name, 'historical_experimental10_reference')
    datasets = read_tsv(OD / 'manifests/step3_dataset_index.tsv')
    assert {r['dataset_endpoint'] for r in datasets} == set(EXPECTED)
    mask_rows = []
    labels = set()
    expected_counts = []
    ref10 = {r['protein_id']: r for r in references if r['reference_class'] == 'experimental10_historical'}
    assert set(ref10) == set(EXP_CHAINS)
    for endpoint in datasets:
        name = endpoint['dataset_endpoint']
        columns = ['record_count', 'unique_sequence_count', 'full_sequence_residues', 'qc_label_count']
        assert tuple(int(endpoint[c]) for c in columns) == EXPECTED[name]
        register(endpoint['record_manifest'], 'endpoint_manifest', name, endpoint['record_manifest_sha256'])
        register(endpoint['feature_index'], 'endpoint_feature_index', name, endpoint['feature_index_sha256'])
        rows = read_tsv(endpoint['record_manifest'])
        assert len(rows) == EXPECTED[name][0]
        expected_counts.append({'endpoint': name, 'records': EXPECTED[name][0], 'unique_sequences': EXPECTED[name][1],
                                'full_residues': EXPECTED[name][2], 'qc_label_count': EXPECTED[name][3],
                                'new_label_policy': LABEL_VERSION if name.startswith('md_') else 'unchanged_v2_1',
                                'source_manifest': endpoint['record_manifest'], 'source_manifest_sha256': endpoint['record_manifest_sha256']})
        for r in rows:
            assert r['label_path'] not in labels
            labels.add(r['label_path'])
            register(r['label_path'], 'v2_1_label_package', name, r['label_file_sha256'], 'Byte-level freeze; no replacement')
            assert r['logical_feature_dimension'] == '1182' and r['logical_feature_dtype'] == 'float32'
            if name == 'experimental10':
                reference = ref10[r['protein_id']]
                assert reference['chain_id'] == EXP_CHAINS[r['protein_id']]
                with np.load(r['label_path'], allow_pickle=False) as z:
                    mask = np.asarray(z['qc_mask'], dtype=bool)
                    sequence = np.asarray(z['sequence_ascii'], dtype=np.uint8).tobytes().decode('ascii')
                assert sequence == reference['sequence'] and digest(sequence.encode('ascii')) == r['sequence_sha256']
                assert len(mask) == int(r['sequence_length'])
                assert int(mask.sum()) == int(r['qc_label_count']) == int(reference['valid_experimental_labels'])
                mask_rows.append({'protein_id': r['protein_id'], 'record_id': r['record_id'],
                                  'authority_chain_id': reference['chain_id'], 'sequence_length': len(sequence),
                                  'sequence_sha256': r['sequence_sha256'], 'valid_labels': int(mask.sum()),
                                  'mask_uint8_sha256': digest(mask.astype(np.uint8).tobytes()),
                                  'valid_sequence_positions_1based': ','.join(str(i + 1) for i in np.flatnonzero(mask)),
                                  'source_label_path': r['label_path'], 'source_label_sha256': r['label_file_sha256']})
    assert len(labels) == 4103
    assert sum(r['valid_labels'] for r in mask_rows) == 709
    assert sum(r['sequence_length'] for r in mask_rows) == 982
    registry_rows = sorted(registry.values(), key=lambda r: (r['asset_type'], r['scope'], r['remote_path']))
    registry_bytes = encode_tsv(registry_rows)
    registry_sha = digest(registry_bytes)
    frozen_at = dt.datetime.now().astimezone().isoformat()
    protocol = copy.deepcopy(old_protocol)
    protocol.update({'protocol_id': 'md_sop_v2_2_20261107', 'plan_version': VERSION, 'run_tag': TAG,
                     'status': 'scientific_recipe_frozen_dataset_binding_pending', 'frozen_at': frozen_at})
    protocol['project'].update({'plan_path': str(C / 'MD-SOP v2.2 training plan.md'), 'plan_sha256': step0['plan']['sha256']})
    protocol['paths'] = {key: value.replace('/' + OLD, '/' + TAG) if isinstance(value, str) else value for key, value in protocol['paths'].items()}
    protocol['paths']['joint_split_manifest'] = str(M / 'joint_split_manifest_v2_2.tsv')
    protocol['reference_protection']['experimental_nmr26_use'] = 'reference_protection_only_in_v2_2'
    protocol['joint_split_policy'] = {
        'mode': 'reuse_frozen_v2_1_roles_without_reassignment', 'source_manifest': str(split_path),
        'source_manifest_sha256': split_sha, 'source_role_column': 'role_v2_1', 'derived_role_column': 'role_v2_2',
        'recluster': False, 'resplit': False, 'restore_quarantine_records': False,
        'use_label_values_for_assignment': False, 'require_record_sequence_group_role_mask_equivalence': True,
        'required_audits': ['zero_joint_group_overlap_across_train_validation_test',
                            'zero_exact_sequence_overlap_across_train_validation_test',
                            'zero_reference_group_overlap_with_development', 'consistent_role_within_each_joint_group']}
    protocol['model']['name'] = 'MD-SOP-BiLSTM-window-v2.2'
    protocol['training']['optional_confirmation_seeds'] = []
    protocol['execution_seed_policy'] = {'policy': 'single_seed', 'seeds': [20261007], 'ensemble': False,
                                         'extra_seed_search': False, 'reference_decision': str(decision_path)}
    protocol['md_label_policy'] = {
        'label_version': LABEL_VERSION, 'replacement_endpoints': ['md_train', 'md_validation', 'md_test'],
        'unchanged_endpoints': ['soppe_train', 'soppe_validation', 'soppe_test', 'experimental10'],
        'definition': 'mean_over_replicas_of_mean_over_complete_nonoverlapping_5tauc_windows',
        'replicas': ['R1', 'R2', 'R3'], 'replica_weighting': 'equal', 'within_replica_window_weighting': 'equal',
        'columns': ['window_s2_mean_r1', 'window_s2_mean_r2', 'window_s2_mean_r3'],
        'require_all_three_finite': True, 'skip_nan': False, 'accumulation_dtype': 'float64', 'stored_dtype': 'float32',
        'fallback_to_full100ns': False, 'first_window_only': False, 'treat_windows_as_independent_training_samples': False,
        'square_target_again': False, 'target_normalization': 'none', 'reuse_frozen_tau_c': True,
        'window_multiplier_tau_c': 5.0, 'window_quantum_ps': 100, 'window_boundary': 'half_open',
        'omit_incomplete_tail': True, 'final_100ns_endpoint_excluded': True,
        'preserve_existing_qc_masks_and_positions': True, 'new_qc_thresholds': False,
        'minimum_windows_per_replica': 1, 'new_protein_records': False,
        'expected_md_records': 1359, 'expected_valid_md_residue_targets': 188991,
        'tau_c_window_is_not_regression_head_window': True,
        'source_files': [{'path': k, 'sha256': v} for k, v in upstream.items()]}
    protocol['authoritative_inputs']['iRED_window_labels'] = next(iter(upstream))
    protocol['authoritative_inputs']['frozen_v2_1_joint_split'] = str(split_path)
    protocol['frozen_reference_registry'] = {'path': str(M / 'v2_1_reference_registry.tsv'), 'sha256': registry_sha,
                                            'asset_count': len(registry_rows), 'label_packages': len(labels),
                                            'prediction_tables': len(prediction_refs), 'large_feature_array_policy': 'freeze_existing_indices_and_recorded_hashes_not_full_array_rescan'}
    protocol['data_release_binding'] = {
        'status': 'pending_steps_2_3', 'required_before_training': True,
        'binding_path': str(C / 'data_binding_v2_2.yaml'),
        'resolved_training_config_path': str(C / 'resolved_train_v2_2.yaml'),
        'immutable_recipe_files': [str(C / 'protocol_v2_2.yaml'), str(C / 'common_train_v2_2.yaml')],
        'policy': 'Do not edit frozen scientific recipes; add a separately hashed data binding after data acceptance, then resolve a runtime config.'}
    protocol['evaluation']['md_test_primary_target'] = LABEL_VERSION
    protocol['evaluation']['md_test_auxiliary_target'] = 'md_ired_full100ns_three_replica_mean_v2_1'
    protocol['evaluation']['require_v2_1_v2_2_by_full100ns_5tauc_two_by_two'] = True
    protocol['evaluation']['same_target_and_mask_required_for_version_metric_differences'] = True
    protocol['evaluation']['v2_1_predictions_read_only_no_reselection'] = True
    protocol['evaluation']['scores_computed_at_step1'] = False
    protocol['evaluation']['historically_used_benchmarks_not_new_blind_test'] = True
    protocol['change_control'] = {
        'permitted_scientific_change': 'MD label definition including matching MD validation/test targets',
        'must_remain_equal': ['sample_ids', 'sequences', 'homology_groups', 'roles', 'qc_masks', 'features', 'model_architecture',
                              'initialization_parameters', 'random_streams', 'optimizer', 'loss', 'scheduler', 'early_stopping', 'maximum_update_budget'],
        'retrain_all_six_from_untrained_paired_initialization': True,
        'b_soppe_routes_are_negative_controls': True,
        'raw_test_predictions': {'clip': False, 'smooth': False, 'sigmoid': False, 'calibrate': False},
        'do_not_use_tests_to_change_protocol': True, 'historical_benchmark_reuse_disclosed': True}
    protocol['step1_acceptance']['next_step_only_after_pass'] = 'step2_publish_5tauc_labels_v2_2'
    train = copy.deepcopy(old_train)
    train.update({'plan_version': VERSION, 'run_tag': TAG, 'protocol_id': protocol['protocol_id']})
    for key in ['code_root', 'model_root', 'result_root', 'log_root', 'audit_root']:
        train[key] = train[key].replace('/' + OLD, '/' + TAG)
    for key in ['trainer_manifest', 'development_label_root', 'evaluation_root']:
        train['data_access'][key] = train['data_access'][key].replace('/' + OLD, '/' + TAG)
    train['data_access']['trainer_manifest_sha256'] = None
    train['data_access']['future_manifest_binding_status'] = 'pending_steps_2_3_not_a_frozen_data_hash'
    train['initialization_policy'] = {'access': 'read_only_original_untrained_states', 'use_trained_v2_1_weights': False,
                                      'load_old_optimizer_or_scheduler': False, 'must_match_original_parameter_state_hashes': True}
    train['label_versions'] = {'md_ired': LABEL_VERSION, 'soppe_legacy': 'unchanged_frozen_v2_1_SOPPE_targets',
                               'experimental10': 'unchanged_709_position_historical_mask'}
    train['dataset_binding'] = {'status': 'unbound', 'path': str(C / 'data_binding_v2_2.yaml'),
                                'reference_registry': str(M / 'v2_1_reference_registry.tsv'), 'reference_registry_sha256': registry_sha,
                                'required_fields': ['label_release_sha256', 'trainer_manifest_sha256', 'dataset_index_sha256',
                                                    'feature_index_sha256', 'data_acceptance_sha256', 'model_acceptance_sha256'],
                                'immutable_base_recipe': True}
    train['execution_gate'] = {'base_recipe_is_not_directly_runnable': True, 'require_data_binding': True,
                               'require_step3_data_acceptance': True, 'require_step4_model_trainer_acceptance': True,
                               'resolved_runtime_config': str(C / 'resolved_train_v2_2.yaml')}
    artifacts = train['artifacts']
    for key, old_name in list(artifacts.items()):
        path = Path(old_name)
        artifacts[key] = path.stem + '_v2_2_' + DATE + path.suffix
    # Literal invariants catch any unintended changes even when a whole subtree is diff-allowed.
    invariant_results = {}
    def check(name, condition):
        if not condition:
            raise RuntimeError('Scientific invariant failed: ' + name)
        invariant_results[name] = True
    for key in ['random_streams', 'training', 'scheduler', 'early_stopping', 'experiments', 'initialization']:
        check('common_train.' + key, train[key] == old_train[key])
    for key in ['expected_supervised_train_records', 'expected_validation_records', 'zero_supervision_records']:
        check('common_train.data_access.' + key, train['data_access'][key] == old_train['data_access'][key])
    for key in ['features', 'homology_definition', 'sample_definition', 'loss', 'experiments', 'paired_initialization', 'access_control']:
        check('protocol.' + key, protocol[key] == old_protocol[key])
    check('architecture_except_version_name', {k: v for k, v in protocol['model'].items() if k != 'name'} ==
          {k: v for k, v in old_protocol['model'].items() if k != 'name'})
    check('protocol.training_except_actual_single_seed_lock',
          {k: v for k, v in protocol['training'].items() if k != 'optional_confirmation_seeds'} ==
          {k: v for k, v in old_protocol['training'].items() if k != 'optional_confirmation_seeds'})
    check('all_six_experiments', list(train['experiments']) == EXPERIMENTS)
    check('single_seed_no_ensemble', protocol['execution_seed_policy']['seeds'] == [20261007] and not protocol['execution_seed_policy']['ensemble'])
    check('future_hash_not_fabricated', train['data_access']['trainer_manifest_sha256'] is None and train['dataset_binding']['status'] == 'unbound')
    check('all_output_roots_versioned', all('/' + TAG in train[k] for k in ['code_root', 'model_root', 'result_root', 'log_root', 'audit_root']))
    differences = config_diff('protocol', old_protocol, protocol) + config_diff('common_train', old_train, train)
    contents = {
        C / 'protocol_v2_2.yaml': encode_yaml(protocol), C / 'common_train_v2_2.yaml': encode_yaml(train),
        C / 'seed_policy_v2_2.json': encode_json(protocol['execution_seed_policy']),
        M / 'v2_1_reference_registry.tsv': registry_bytes,
        M / 'protected_reference_manifest.tsv': reference_path.read_bytes(),
        M / 'paired_initialization_reference_v2_2.json': init_file.read_bytes(),
        M / 'experimental10_mask_registry_v2_2.tsv': encode_tsv(sorted(mask_rows, key=lambda r: r['protein_id'])),
        M / 'frozen_dataset_counts_v2_2.tsv': encode_tsv(sorted(expected_counts, key=lambda r: r['endpoint'])),
        M / 'frozen_prediction_references_v2_2.tsv': encode_tsv(prediction_refs),
        A / 'source_configuration_snapshots/protocol_v2_1.yaml': (OC / 'protocol_v2_1.yaml').read_bytes(),
        A / 'source_configuration_snapshots/common_train_v2_1.yaml': (OC / 'common_train_v2_1.yaml').read_bytes(),
        A / 'config_diff_v2_1_vs_v2_2.tsv': encode_tsv(differences),
        A / 'config_diff_v2_1_vs_v2_2.json': encode_json(differences),
        A / 'scientific_invariants_v2_2.json': encode_json(invariant_results),
    }
    summary = {'status': 'PASS', 'model_version': VERSION, 'run_tag': TAG, 'step': 1, 'frozen_at': frozen_at,
               'execution_user': 'amax', 'registered_assets': len(registry_rows), 'registered_asset_types': dict(collections.Counter(r['asset_type'] for r in registry_rows)),
               'label_packages_hashed_and_matched_to_frozen_manifest': len(labels), 'prediction_tables_frozen_without_rescoring': len(prediction_refs),
               'frozen_references': 36, 'reference_unique_sequences': 35, 'reference_homology_groups': 30,
               'experimental10_count': len(mask_rows), 'experimental10_valid_positions': sum(r['valid_labels'] for r in mask_rows),
               'protected_reference_sequence_hashes_recalculated': True, 'zero_active_split_group_or_sequence_overlap': True,
               'scientific_invariant_count': len(invariant_results), 'configuration_differences_recorded': len(differences),
               'plan_sha256': step0['plan']['sha256'], 'protocol_sha256': digest(contents[C / 'protocol_v2_2.yaml']),
               'common_train_recipe_sha256': digest(contents[C / 'common_train_v2_2.yaml']), 'reference_registry_sha256': registry_sha,
               'dataset_binding_status': 'pending_steps_2_3', 'direct_training_allowed': False,
               'new_md_labels_generated': False, 'old_label_values_changed': False, 'training_started': False,
               'test_scores_computed': False, 'next_step': 'step2_publish_5tauc_labels_v2_2'}
    return contents, summary, registry_rows


def freeze(payload):
    contents, summary, registry = build()
    if payload.get('preflight_only'):
        return {'summary': summary, 'preflight_only': True}
    contents[CODE / '01_freeze_protocol_and_references_v2_2.py'] = payload['script'].encode('utf-8')
    # No v2.1 location is ever a write target.
    roots = [P / part / TAG for part in ['code', 'config', 'data', 'model', 'results', 'logs', 'tmp']]
    for path in contents:
        if not any(root in path.parents for root in roots) or path.exists():
            raise RuntimeError('Unsafe or already existing output: ' + str(path))
    os.umask(0o002)
    A.mkdir(mode=0o2775)
    (A / 'source_configuration_snapshots').mkdir(mode=0o2775)
    created = []
    def write(path, data):
        if not any(root in path.parents for root in roots):
            raise RuntimeError('Write escaped v2.2 version roots')
        with path.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        created.append(path)
    for path, content in contents.items():
        write(path, content)
    unchanged = []
    for row in registry:
        path = Path(row['remote_path']); st = path.stat()
        match = sha(path) == row['sha256'] and st.st_mtime_ns == row['mtime_ns'] and st.st_size == row['bytes']
        match = match and pwd.getpwuid(st.st_uid).pw_name == row['owner'] and grp.getgrgid(st.st_gid).gr_name == row['group']
        match = match and format(stat.S_IMODE(st.st_mode), '04o') == row['mode']
        if not match:
            raise RuntimeError('Source asset changed during freeze: ' + str(path))
        unchanged.append({'remote_path': str(path), 'before_sha256': row['sha256'], 'after_sha256': row['sha256'], 'metadata_unchanged': True})
    write(A / 'readonly_assets_before_after_v2_2.tsv', encode_tsv(unchanged))
    summary['all_registered_source_assets_unchanged_after_step1'] = True
    summary['new_output_owner_group'] = 'amax:amax'
    write(A / 'step1_protocol_acceptance_v2_2_20261107.json', encode_json(summary))
    report = '# MD-SOP v2.2 步骤1：协议与对照资产冻结验收\n\n'
    report += '- 验收结果：**PASS**\n- Run tag：`' + TAG + '`\n- WQQ冻结时间：`' + summary['frozen_at'] + '`\n'
    report += '- 创建账号：`amax`；没有修改v2.1、iRED原始产物或现有软件环境。\n\n'
    report += '## 正式冻结内容\n\n'
    report += '1. 六组A/B/C×w1/w15、原始配对初始化、全部随机流和v2.1实际优化/预算/选模设置。\n'
    report += '2. MD标签定义：先在每条重复内平均全部完整5τc窗口，再对R1/R2/R3等权平均；不是首窗口、轨迹拼接或窗口扩增样本。\n'
    report += '3. MD train/validation/test使用同一定义；S-OPPE、实验目标和全部mask/角色/特征不变。\n'
    report += '4. 单种子20261007，无ensemble；B组仍重训作为负对照；MD-test补旧/新模型×旧/新目标2×2比较。\n\n'
    report += '## 核验规模\n\n'
    report += '- 登记只读资产：**' + str(summary['registered_assets']) + '**个文件。\n'
    report += '- 原标签包：**4103**个，全部逐文件哈希与旧端点清单一致；没有改写数值。\n'
    report += '- 预测表：**13**个（MD/S各六组，实验10共用合并表），仅登记哈希，没有重新评分。\n'
    report += '- 保护参考：**36**条（26＋10），35条唯一序列、30个联合同源组；序列哈希重算通过。\n'
    report += '- Experimental10：**10**条完整序列、**982**残基、**709**个固定实验位置；登记逐蛋白mask哈希及位置。未读取实验目标数值计算指标。\n'
    report += '- 自动检查**' + str(summary['scientific_invariant_count']) + '**项科学配置不变量；逐字段记录**' + str(summary['configuration_differences_recorded']) + '**项差异及原因。\n'
    report += '- 所有登记来源文件在生成新文件后再次核对哈希及元数据，全部不变。大特征矩阵本次不全量重读，冻结其已有索引与记录的组件哈希，后续数据验收再绑定。\n\n'
    report += '## 数据尚未生成：不能直接训练\n\n'
    report += '`protocol_v2_2.yaml`和`common_train_v2_2.yaml`是已冻结的科学配方。新标签发布、trainer manifest及新数据哈希尚未生成，'
    report += '因此`trainer_manifest_sha256`明确为null，状态为unbound，不能复制旧哈希冒充新数据。\n\n'
    report += '步骤2/3通过后生成独立`data_binding_v2_2.yaml`；步骤4验收后生成`resolved_train_v2_2.yaml`用于正式训练。'
    report += '不得编辑已冻结配方来悄悄改变科学设置；本次没有伪造这些后续文件。\n\n'
    report += '## 文件入口\n\n'
    report += '- `config/' + TAG + '/protocol_v2_2.yaml`\n- `config/' + TAG + '/common_train_v2_2.yaml`\n'
    report += '- `data/' + TAG + '/manifests/v2_1_reference_registry.tsv`\n'
    report += '- `data/' + TAG + '/manifests/protected_reference_manifest.tsv`（原36参考清单的原字节副本，保留原schema/版本字段用于来源追溯）\n'
    report += '- `data/' + TAG + '/manifests/experimental10_mask_registry_v2_2.tsv`\n'
    report += '- 本目录的`config_diff_v2_1_vs_v2_2.tsv/json`、`scientific_invariants_v2_2.json`和SHA-256清单。\n\n'
    report += '## 当前边界\n\n步骤1完成。尚未生成5τc标签或NPZ、未重训、未进行新测试评分；下一步是步骤2发布窗口标签。'
    report += '历史基准已被使用，后续结果不能称为从未触碰的全新盲测。\n'
    write(A / 'step1_protocol_acceptance_v2_2_20261107.md', report.encode('utf-8'))
    manifest = []
    for path in sorted(created):
        st = path.stat()
        assert st.st_uid == 1000 and st.st_gid == 1000
        manifest.append({'relative_path': str(path.relative_to(P)), 'bytes': st.st_size, 'sha256': sha(path)})
    write(A / 'step1_artifact_sha256.tsv', encode_tsv(manifest))
    return collect()


def collect():
    summary = json.loads((A / 'step1_protocol_acceptance_v2_2_20261107.json').read_text(encoding='utf-8'))
    files = []
    for item in read_tsv(A / 'step1_artifact_sha256.tsv'):
        path = P / item['relative_path']
        assert P in path.parents and TAG in path.parts and path.resolve() == path
        assert sha(path) == item['sha256']
        files.append(path)
    files.append(A / 'step1_artifact_sha256.tsv')
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, str(path.relative_to(P)))
    return {'summary': summary, 'artifact_count': len(files), 'archive_base64': base64.b64encode(out.getvalue()).decode('ascii')}
