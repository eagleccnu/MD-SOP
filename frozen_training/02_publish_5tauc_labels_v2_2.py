#!/usr/bin/env python3
"""MD-SOP v2.2 step2: publish trusted-preparation-only 5tau_c label tables.

No original file is modified. No NPZ, training manifest, model, or score is
created. All outputs are under the existing v2_2_20261107 version roots.
"""
from __future__ import annotations
import base64
import collections
import csv
import datetime as dt
import gzip
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
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

P = Path('/data/jiangbin/SOPPCL/MD-SOP')
TAG = 'v2_2_20261107'
OLD = 'v2_1_20261007'
C = P / 'config' / TAG
D = P / 'data' / TAG
OD = P / 'data' / OLD
A = D / 'audits/step2_label_release'
LABEL_VERSION = 'md_ired_5tauc_windowmean_v2_2'
RELEASE = D / 'label_releases' / LABEL_VERSION
CODE = P / 'code' / TAG
EXPECTED = {'md_train': (1070, 149299), 'md_validation': (153, 20753), 'md_test': (136, 18939)}
AA = dict(zip(['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'],
              list('ARNDCQEGHILKMFPSTWYV')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def encode_tsv(items):
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(items[0]), delimiter='\t', lineterminator='\n')
    w.writeheader(); w.writerows(items)
    return out.getvalue().encode('utf-8')


def mask_sha(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def verify_inputs(registry):
    for r in registry:
        path = Path(r['remote_path']); st = path.stat()
        if sha(path) != r['sha256'] or st.st_size != int(r['bytes']) or st.st_mtime_ns != int(r['mtime_ns']):
            raise RuntimeError('Frozen original input changed: ' + str(path))
        if pwd.getpwuid(st.st_uid).pw_name != r['owner'] or grp.getgrgid(st.st_gid).gr_name != r['group'] or format(stat.S_IMODE(st.st_mode), '04o') != r['mode']:
            raise RuntimeError('Original ownership/mode changed: ' + str(path))


def build():
    assert pwd.getpwuid(os.geteuid()).pw_name == 'amax'
    assert P.resolve() == P
    assert not RELEASE.exists(), 'Refusing to overwrite an existing label release'
    assert not A.exists(), 'Step2 audit already exists'
    status1 = json.loads((D / 'audits/step1_protocol/step1_protocol_acceptance_v2_2_20261107.json').read_text(encoding='utf-8'))
    assert status1['status'] == 'PASS'
    assert sha(C / 'protocol_v2_2.yaml') == status1['protocol_sha256']
    assert sha(C / 'common_train_v2_2.yaml') == status1['common_train_recipe_sha256']
    assert sha(D / 'manifests/v2_1_reference_registry.tsv') == status1['reference_registry_sha256']
    protocol = yaml.safe_load((C / 'protocol_v2_2.yaml').read_text(encoding='utf-8'))
    policy = protocol['md_label_policy']
    assert policy['label_version'] == LABEL_VERSION
    assert policy['columns'] == ['window_s2_mean_r1','window_s2_mean_r2','window_s2_mean_r3']
    assert policy['require_all_three_finite'] and not policy['skip_nan'] and not policy['fallback_to_full100ns']
    registry = rows(D / 'manifests/v2_1_reference_registry.tsv')
    verify_inputs(registry)
    frozen_records = {r['remote_path']: r for r in registry}
    source_files = {Path(r['path']).name: Path(r['path']) for r in policy['source_files']}
    for item in policy['source_files']:
        assert sha(item['path']) == item['sha256']
    production_config_path = Path('/data/jiangbin/iRED/config/ired_full_production_v1.yml')
    assert sha(production_config_path) == frozen_records[str(production_config_path)]['sha256']
    production_config = json.loads(production_config_path.read_text(encoding='utf-8'))
    low, high = production_config['acceptance']['universal_s2_bounds']
    assert (low, high) == (-0.05, 1.05)
    labels = pd.read_parquet(source_files['labels_by_residue.parquet'])
    selection = pd.read_csv(source_files['selection.tsv'], sep='\t')
    runs = pd.read_parquet(source_files['run_qc.parquet'])
    raw = pd.read_parquet(source_files['residue_s2_all_runs.parquet'], columns=[
        'pdb_chain','replica','scope','window_index','residue_index','sequence_position','resname','nh_eligible','s2'])
    assert len(labels) == 188991 and labels.pdb_chain.nunique() == 1359
    assert len(selection) == 1359 and not selection.pdb_chain.duplicated().any()
    keys = ['pdb_chain','residue_index']
    assert not labels.duplicated(keys).any()
    assert not labels.duplicated(['pdb_chain','sequence_position']).any()
    assert not raw.duplicated(keys + ['replica','scope','window_index']).any()
    assert set(labels.pdb_chain) == set(selection.pdb_chain)
    source_values = labels[policy['columns']].to_numpy(dtype=np.float64)
    full_values = labels[['s2_r1','s2_r2','s2_r3']].to_numpy(dtype=np.float64)
    assert np.isfinite(source_values).all() and np.isfinite(full_values).all()
    assert ((source_values >= low) & (source_values <= high)).all()
    assert ((full_values >= low) & (full_values <= high)).all()
    full_mean = np.mean(full_values, axis=1, dtype=np.float64)
    assert np.max(np.abs(full_mean - labels.s2_replica_mean.to_numpy())) <= 1e-12
    new_mean = np.mean(source_values, axis=1, dtype=np.float64)
    eligible = raw.loc[raw.nh_eligible.astype(bool)].copy()
    windows = eligible.loc[eligible.scope.eq('window')].copy()
    assert np.isfinite(windows.s2).all()
    assert windows.s2.between(low, high).all()
    assert np.isfinite(eligible.loc[eligible.scope.eq('full'),'s2']).all()
    reconstructed = windows.groupby(keys + ['replica']).agg(
        mean=('s2','mean'), count=('s2','size'), unique_window_count=('window_index','nunique'),
        first_window=('window_index','min'), last_window=('window_index','max'),
        min_position=('sequence_position','min'), max_position=('sequence_position','max'),
        residue_names=('resname','nunique'), residue_name=('resname','first')).reset_index()
    assert (reconstructed['count'] == reconstructed.unique_window_count).all()
    assert reconstructed.first_window.eq(0).all()
    assert (reconstructed.last_window + 1 == reconstructed['count']).all()
    assert reconstructed.residue_names.eq(1).all()
    assert (reconstructed.min_position == reconstructed.max_position).all()
    assert len(reconstructed) == len(labels) * 3
    indexed_labels = labels.set_index(keys)
    rederived = {}
    for replica in ['R1','R2','R3']:
        r = reconstructed.loc[reconstructed.replica.eq(replica)].set_index(keys).reindex(indexed_labels.index)
        assert r['mean'].notna().all()
        name = replica.lower()
        err = float(np.max(np.abs(r['mean'].to_numpy() - indexed_labels['window_s2_mean_' + name].to_numpy())))
        assert err <= 1e-12
        assert np.array_equal(r['count'].to_numpy(), indexed_labels['window_count_' + name].to_numpy())
        assert np.array_equal(r.min_position.to_numpy(), indexed_labels.sequence_position.to_numpy())
        assert np.array_equal(r.residue_name.to_numpy(), indexed_labels.resname.to_numpy())
        rederived[replica] = {'residues': len(r), 'maximum_abs_error': err}
    raw_full = eligible.loc[eligible.scope.eq('full')].pivot(index=keys, columns='replica', values='s2').reindex(indexed_labels.index)
    assert not raw_full.isna().any().any()
    assert np.max(np.abs(raw_full[['R1','R2','R3']].to_numpy() - full_values)) <= 1e-12
    assert runs.scope.value_counts().to_dict() == {'window': 7701, 'full': 4077}
    assert (selection.window_quantized_ps > 0).all()
    assert (selection.window_quantized_ps % 100 == 0).all()
    assert np.array_equal((100000 // selection.window_quantized_ps).to_numpy(dtype=int), selection.expected_nonoverlap_windows.to_numpy(dtype=int))
    recomputed_lengths = np.asarray([int(round(float(tau) * 5 * 1000 / 100) * 100) for tau in selection.tau_window_ns])
    assert np.array_equal(recomputed_lengths, selection.window_quantized_ps.to_numpy(dtype=int))
    selection_index = selection.set_index('pdb_chain')
    for (protein, replica), g in runs.groupby(['pdb_chain','replica']):
        current = selection_index.loc[protein]
        windows_qc = g.loc[g.scope.eq('window')].sort_values('window_index')
        k = int(current.expected_nonoverlap_windows); w = int(current.window_quantized_ps)
        assert len(windows_qc) == k
        assert np.array_equal(windows_qc.window_index.to_numpy(), np.arange(k))
        assert np.array_equal(windows_qc.physical_start_ps.to_numpy(), np.arange(k) * w)
        assert np.array_equal(windows_qc.physical_stop_ps.to_numpy(), (np.arange(k) + 1) * w)
        assert windows_qc.frame_count.eq(w // 100).all()
        assert windows_qc.source_dt_ps.eq(100).all()
        f = g.loc[g.scope.eq('full')]
        assert len(f) == 1 and f.iloc[0].frame_count == 1001
        assert f.iloc[0].physical_start_ps == 0 and f.iloc[0].physical_stop_ps == 100000
    frozen_split_path = Path(protocol['joint_split_policy']['source_manifest'])
    assert sha(frozen_split_path) == protocol['joint_split_policy']['source_manifest_sha256']
    split = pd.read_csv(frozen_split_path, sep='\t', keep_default_na=False)
    split = split.loc[split.source.eq('md_ired') & split.role_v2_1.isin(['train','validation','test'])]
    assert len(split) == 1359 and not split.record_id.duplicated().any()
    dataset_index = pd.read_csv(OD / 'manifests/step3_dataset_index.tsv', sep='\t', keep_default_na=False)
    md_manifests = []
    for endpoint, (record_count, residue_count) in EXPECTED.items():
        index_row = dataset_index.loc[dataset_index.dataset_endpoint.eq(endpoint)].iloc[0]
        assert sha(index_row.record_manifest) == index_row.record_manifest_sha256
        manifest = pd.read_csv(index_row.record_manifest, sep='\t', keep_default_na=False)
        assert len(manifest) == record_count and manifest.dataset_endpoint.eq(endpoint).all()
        assert manifest.qc_label_count.sum() == residue_count
        md_manifests.append(manifest)
    manifest = pd.concat(md_manifests, ignore_index=True)
    assert set(manifest.record_id) == set(split.record_id)
    assert set(manifest.protein_id) == set(selection.pdb_chain)
    assert not manifest.protein_id.duplicated().any()
    split_by_id = split.set_index('record_id')
    labels = labels.copy()
    labels['label_s2_float64'] = new_mean
    labels['full100ns_s2_float64'] = full_mean
    label_groups = {k: g for k, g in labels.groupby('pdb_chain', sort=False)}
    source_hashes = {Path(s['path']).name: s['sha256'] for s in policy['source_files']}
    frames = []; integrity = []; role_transitions = []
    for record in manifest.itertuples():
        s = split_by_id.loc[record.record_id]
        selected = selection_index.loc[record.protein_id]
        assert record.sequence_sha256 == s.sequence_sha256 == selected.sequence_sha256
        assert int(record.sequence_length) == len(s.sequence) == int(selected.sequence_length)
        assert hashlib.sha256(s.sequence.encode('ascii')).hexdigest() == record.sequence_sha256
        assert record.role_v2_1 == s.role_v2_1
        assert record.joint_homology_group == s.joint_homology_group
        assert record.protein_id == s.protein_id
        label_path = Path(record.label_path)
        assert sha(label_path) == record.label_file_sha256
        current = label_groups[record.protein_id].sort_values('sequence_position').copy()
        positions = current.sequence_position.to_numpy(dtype=int) - 1
        assert np.all(positions >= 0) and np.all(positions < int(record.sequence_length))
        assert len(np.unique(positions)) == len(positions)
        with np.load(label_path, allow_pickle=False) as package:
            old_y = np.asarray(package['y'], dtype=np.float32).copy()
            seq = np.asarray(package['sequence_ascii'], dtype=np.uint8).tobytes().decode('ascii')
            masks = {k: np.asarray(package[k], dtype=bool).copy() for k in ['label_mask','qc_mask','raw_label_present','train_access','eval_access','loss_mask']}
            assert int(package['length'][0]) == len(seq) == len(old_y) == int(record.sequence_length)
        assert seq == s.sequence
        assert np.array_equal(np.flatnonzero(masks['qc_mask']), positions)
        assert np.array_equal(masks['label_mask'], masks['qc_mask'])
        assert np.array_equal(masks['loss_mask'], masks['qc_mask'] & masks['train_access'])
        assert masks['raw_label_present'][positions].all()
        expected_full32 = current.full100ns_s2_float64.to_numpy().astype(np.float32)
        assert np.array_equal(old_y[positions], expected_full32)
        assert all(AA[rn] == seq[pos] for rn, pos in zip(current.resname, positions))
        for replica in ['r1','r2','r3']:
            assert current['window_count_' + replica].eq(int(selected.expected_nonoverlap_windows)).all()
        assert np.allclose(current.tau_window_ns, selected.tau_window_ns, rtol=0, atol=1e-10)
        assert current.tau_window_source.eq(selected.tau_window_source).all()
        assert current.tau_tier.eq(selected.tau_tier).all()
        actual_role = record.role_v2_1
        assert record.dataset_endpoint == 'md_' + actual_role
        if actual_role == 'train':
            assert masks['train_access'].all() and not masks['eval_access'].any()
        else:
            assert not masks['train_access'].any() and masks['eval_access'].all() and not masks['loss_mask'].any()
        current = current.rename(columns={'dataset_role':'production_dataset_role','group_partition':'production_partition',
                                           'homology_group':'production_homology_group','chain_id':'trajectory_chain_id'})
        current['record_id'] = record.record_id
        current['protein_id'] = record.protein_id
        current['source'] = 'md_ired'
        current['dataset_endpoint'] = record.dataset_endpoint
        current['role_v2_1'] = actual_role
        current['role_v2_2'] = actual_role
        current['joint_homology_group'] = record.joint_homology_group
        current['sequence_sha256'] = record.sequence_sha256
        current['sequence_length'] = int(record.sequence_length)
        current['protein_chain_id'] = record.chain_id
        current['amino_acid'] = [seq[pos] for pos in positions]
        current['label_s2'] = current.label_s2_float64.to_numpy().astype(np.float32)
        current['old_label_s2'] = old_y[positions]
        current['new_minus_old_s2_float64'] = current.label_s2_float64 - current.full100ns_s2_float64
        current['new_minus_old_s2_float32_targets'] = current.label_s2.astype(np.float64) - current.old_label_s2.astype(np.float64)
        current['new_label_version'] = LABEL_VERSION
        current['old_label_version'] = 'md_ired_full100ns_three_replica_mean_v2_1'
        current['qc_eligible'] = True
        for mask_name in ['raw_label_present','train_access','eval_access','loss_mask']:
            current[mask_name] = masks[mask_name][positions]
        current['window_target_5tau_ns'] = float(selected.window_target_5tau_ns)
        current['window_quantized_ps'] = int(selected.window_quantized_ps)
        current['window_length_ns'] = int(selected.window_quantized_ps) / 1000
        current['expected_nonoverlap_windows'] = int(selected.expected_nonoverlap_windows)
        current['covered_time_ns_per_replica'] = int(selected.window_quantized_ps) * int(selected.expected_nonoverlap_windows) / 1000
        current['omitted_tail_ns_per_replica'] = 100 - current.covered_time_ns_per_replica
        current['source_label_path'] = record.label_path
        current['source_label_file_sha256'] = record.label_file_sha256
        current['source_labels_parquet_sha256'] = source_hashes['labels_by_residue.parquet']
        current['source_all_runs_parquet_sha256'] = source_hashes['residue_s2_all_runs.parquet']
        current['source_run_qc_parquet_sha256'] = source_hashes['run_qc.parquet']
        current['source_selection_sha256'] = source_hashes['selection.tsv']
        current['frozen_protocol_sha256'] = status1['protocol_sha256']
        frames.append(current)
        info = {'record_id': record.record_id, 'protein_id': record.protein_id, 'dataset_endpoint': record.dataset_endpoint,
                'role_v2_1': actual_role, 'role_v2_2': actual_role, 'joint_homology_group': record.joint_homology_group,
                'sequence_sha256': record.sequence_sha256, 'sequence_length': len(seq),
                'qc_label_count': len(positions), 'loss_label_count': int(masks['loss_mask'].sum()),
                'source_label_path': record.label_path, 'source_label_sha256': record.label_file_sha256,
                'tau_c_ns': float(selected.tau_window_ns), 'tau_c_source': selected.tau_window_source,
                'window_length_ns': int(selected.window_quantized_ps) / 1000,
                'complete_windows_per_replica': int(selected.expected_nonoverlap_windows),
                'original_full100ns_targets_match_float32': True, 'sequence_and_residue_identities_match': True,
                'mask_and_role_preserved': True}
        info.update({name + '_uint8_sha256': mask_sha(mask) for name, mask in masks.items()})
        integrity.append(info)
        if selected.dataset_role != record.dataset_endpoint:
            role_transitions.append({'record_id': record.record_id, 'production_role': selected.dataset_role,
                                     'frozen_md_sop_role': record.dataset_endpoint, 'reason': 'Preserve final MD-SOP v2.1 roles, not the older iRED production partition'})
    result = pd.concat(frames, ignore_index=True).sort_values(['dataset_endpoint','record_id','sequence_position']).reset_index(drop=True)
    assert len(result) == 188991 and result.record_id.nunique() == 1359
    assert result.sequence_sha256.nunique() == 1184
    assert not result.duplicated(['record_id','sequence_position']).any()
    assert result.label_s2.dtype == np.float32
    assert np.isfinite(result.label_s2).all() and result.label_s2.between(low, high).all()
    counts = []
    for endpoint, (n, nres) in EXPECTED.items():
        g = result.loc[result.dataset_endpoint.eq(endpoint)]
        assert len(g) == nres and g.record_id.nunique() == n
        delta = g.new_minus_old_s2_float64.to_numpy()
        counts.append({'endpoint': endpoint, 'records': n, 'valid_residue_targets': nres,
                       'unique_sequences': int(g.sequence_sha256.nunique()),
                       'training_loss_positions': int(g.loss_mask.sum()),
                       'old_full100ns_mean': float(g.full100ns_s2_float64.mean()), 'new_5tauc_mean': float(g.label_s2_float64.mean()),
                       'new_minus_old_mean': float(delta.mean()), 'mean_absolute_difference': float(np.abs(delta).mean())})
    assert result.loss_mask.sum() == 149299
    delta = result.new_minus_old_s2_float64.to_numpy()
    summary = {'status': 'PASS', 'step': 2, 'run_tag': TAG, 'label_version': LABEL_VERSION,
               'release_created_at': dt.datetime.now().astimezone().isoformat(), 'execution_user': 'amax',
               'records': 1359, 'unique_sequences': 1184, 'valid_residue_targets': 188991,
               'counts_by_endpoint': counts, 'raw_run_counts': {'full': 4077, 'window': 7701},
               'source_raw_residue_rows': len(raw), 'source_eligible_window_residue_rows': len(windows),
               'window_reconstruction': rederived, 'maximum_reconstruction_error': max(x['maximum_abs_error'] for x in rederived.values()),
               'old_full100ns_targets_match_all_original_npz_float32': True,
               'all_original_effective_positions_matched_exactly': True, 'all_authority_sequences_and_residue_identities_match': True,
               'all_masks_preserved': True, 'all_final_md_sop_roles_preserved': True,
               'production_vs_final_role_differences_preserved': len(role_transitions),
               'universal_s2_acceptance_bounds': [low, high],
               'new_target_float32_min': float(result.label_s2.min()), 'new_target_float32_max': float(result.label_s2.max()),
               'new_target_outside_0_1_count': int((~result.label_s2.between(0,1)).sum()),
               'paired_difference_float64': {'mean': float(delta.mean()), 'mae': float(np.abs(delta).mean()),
                                              'rmse': float(np.sqrt(np.mean(delta**2))),
                                              'abs_above_005_count': int((np.abs(delta) > .05).sum()),
                                              'abs_above_010_count': int((np.abs(delta) > .10).sum())},
               'unchanged_after_float32_conversion_count': int(np.sum(result.label_s2.to_numpy() == result.old_label_s2.to_numpy())),
               'label_storage_dtype': 'float32', 'aggregation_dtype': 'float64',
               'no_first_window_substitution': True, 'no_full100ns_fallback': True, 'no_clipping_or_normalization': True,
               'frozen_protocol_sha256': status1['protocol_sha256'], 'frozen_training_recipe_sha256': status1['common_train_recipe_sha256'],
               'reference_registry_sha256': status1['reference_registry_sha256'], 'registered_original_assets': len(registry),
               'input_sources': policy['source_files'], 'large_feature_arrays_rescanned': False,
               'source_npz_files_modified': False, 'new_npz_files_generated': False, 'training_started': False,
               'test_prediction_scores_computed': False, 'data_binding_written': False,
               'release_access': 'trusted_dataset_preparer_and_auditor_only_not_a_trainer_manifest',
               'next_step': 'step3_prepare_npz_manifests_and_access_whitelists'}
    return result, integrity, role_transitions, summary, registry


def publish(payload):
    result, integrity, role_transitions, summary, registry = build()
    if payload.get('preflight_only'):
        return {'summary': summary, 'preflight_only': True}
    os.umask(0o002)
    roots = [P / part / TAG for part in ['code','config','data','model','results','logs','tmp']]
    for parent in [D / 'label_releases']:
        if not parent.exists():
            parent.mkdir(mode=0o2775)
    RELEASE.mkdir(mode=0o2775)
    A.mkdir(mode=0o2775)
    created = []
    def write(path, content):
        assert any(root in path.parents for root in roots) and not path.exists()
        with path.open('xb') as f:
            f.write(content);f.flush();os.fsync(f.fileno())
        created.append(path)
    targets = RELEASE / 'md_ired_5tauc_targets_v2_2.parquet'
    metadata = {b'run_tag': TAG.encode(), b'md_label_version': LABEL_VERSION.encode(),
                b'aggregation': b'mean_over_replicas_of_mean_over_complete_nonoverlapping_5tauc_windows',
                b'access_policy': b'trusted_dataset_preparer_and_auditor_only_not_direct_training',
                b'protocol_sha256': summary['frozen_protocol_sha256'].encode()}
    table = pa.Table.from_pandas(result, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
    with targets.open('xb') as f:
        pq.write_table(table, f, compression='zstd', use_dictionary=True)
    created.append(targets)
    check = pq.read_table(targets, columns=['label_s2'])
    assert check.schema.field('label_s2').type == pa.float32()
    assert np.array_equal(check.column('label_s2').to_numpy(), result.label_s2.to_numpy())
    transition_cols = ['record_id','protein_id','dataset_endpoint','role_v2_1','role_v2_2','joint_homology_group',
        'sequence_sha256','sequence_position','residue_index','pdb_resid','resname','amino_acid',
        'old_label_s2','label_s2','full100ns_s2_float64','label_s2_float64','new_minus_old_s2_float64',
        'new_minus_old_s2_float32_targets','tau_window_ns','tau_window_source','tau_tier',
        'window_target_5tau_ns','window_quantized_ps','window_length_ns','window_count_r1','window_count_r2','window_count_r3',
        'covered_time_ns_per_replica','omitted_tail_ns_per_replica','new_label_version','old_label_version',
        'source_label_path','source_label_file_sha256','source_labels_parquet_sha256','source_all_runs_parquet_sha256',
        'source_run_qc_parquet_sha256','source_selection_sha256','frozen_protocol_sha256']
    transition = RELEASE / 'label_transition_v2_2.tsv.gz'
    with transition.open('xb') as f:
        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding='utf-8', newline='') as text:
                result[transition_cols].to_csv(text, sep='\t', index=False, float_format='%.17g', lineterminator='\n')
    created.append(transition)
    write(RELEASE / 'record_integrity_v2_2.tsv', encode_tsv(integrity))
    write(RELEASE / 'partition_counts_v2_2.tsv', encode_tsv(summary['counts_by_endpoint']))
    if role_transitions:
        write(RELEASE / 'production_vs_final_role_notes_v2_2.tsv', encode_tsv(role_transitions))
    write(CODE / '02_publish_5tauc_labels_v2_2.py', payload['script'].encode('utf-8'))
    verify_inputs(registry)
    assert sha(C / 'protocol_v2_2.yaml') == summary['frozen_protocol_sha256']
    assert sha(C / 'common_train_v2_2.yaml') == summary['frozen_training_recipe_sha256']
    summary['all_registered_original_assets_unchanged_after_step2'] = True
    summary['frozen_step1_recipes_unchanged'] = True
    files = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in created if RELEASE in p.parents]
    release_manifest = {
        'schema_version':'1.0', 'model_version':'v2.2', 'run_tag':TAG, 'label_version':LABEL_VERSION,
        'release_root':str(RELEASE), 'status':'PASS', 'records':1359, 'valid_residue_targets':188991,
        'target_column':'label_s2', 'target_dtype':'float32', 'diagnostic_target_column':'label_s2_float64',
        'old_target_column':'old_label_s2', 'index_keys':['record_id','sequence_position'],
        'source_scope':'existing_qc_eligible_NH_positions_only', 'role_column':'role_v2_2',
        'allowed_consumers':['trusted_step3_dataset_preparer','label_auditor'],
        'direct_trainer_access':False, 'contains_md_test_labels':True,
        'test_access_note':'Master table is only for trusted data preparation; step3 must create isolated development/evaluation packages. Do not pass master table to trainer.',
        'definition':'mean_over_replicas_of_mean_over_complete_nonoverlapping_5tauc_windows',
        'frozen_protocol_sha256':summary['frozen_protocol_sha256'],
        'reference_registry_sha256':summary['reference_registry_sha256'], 'files':files,
        'audit_json_path':str(A / 'label_replacement_audit_v2_2.json'),
        'counts_by_endpoint':summary['counts_by_endpoint']}
    manifest_path = RELEASE / 'label_release_manifest_v2_2.json'
    write(manifest_path, encode_json(release_manifest))
    summary['release_manifest_path'] = str(manifest_path)
    summary['release_manifest_sha256'] = sha(manifest_path)
    summary['targets_parquet_path'] = str(targets)
    summary['targets_parquet_sha256'] = sha(targets)
    summary['transition_path'] = str(transition)
    summary['transition_sha256'] = sha(transition)
    write(A / 'label_replacement_audit_v2_2.json', encode_json(summary))
    report = '# MD-SOP v2.2 步骤2：5τc窗口标签发布验收\n\n'
    report += '- 结果：**PASS**\n- Run tag：`' + TAG + '`\n- WQQ创建时间：`' + summary['release_created_at'] + '`\n'
    report += '- 标签版本：`' + LABEL_VERSION + '`；创建账号：`amax`。\n\n'
    report += '## 定义与完整性\n\n'
    report += '每条重复先平均全部完整5τc窗口，再对三条重复等权平均。使用冻结的τc、100 ps窗口量化、半开边界与尾段舍弃规则；'
    report += '不是首窗口、拼接轨迹或窗口扩增样本。float64计算平均，再保存float32训练目标，没有标准化、平方、裁剪或full100ns回退。\n\n'
    report += '| 端点 | 蛋白记录 | 有效残基 | 原目标均值 | 新窗口目标均值 | 平均新−旧 |\n|---|---:|---:|---:|---:|---:|\n'
    for row in summary['counts_by_endpoint']:
        report += '| ' + row['endpoint'] + ' | ' + str(row['records']) + ' | ' + str(row['valid_residue_targets']) + ' | ' + format(row['old_full100ns_mean'],'.6f') + ' | ' + format(row['new_5tauc_mean'],'.6f') + ' | ' + format(row['new_minus_old_mean'],'.6f') + ' |\n'
    report += '\n合计**1359条记录、1184条唯一序列、188991个原有效N–H位置**。与原1359个NPZ逐蛋白核对权威序列、残基身份、全部有效位置与旧目标float32值，完全一致；原mask和最终角色保留。\n\n'
    report += '## 独立重建与来源保护\n\n'
    report += '从逐窗口原始表重新聚合，每条重复对全部188991位置核对，最大误差为' + format(summary['maximum_reconstruction_error'],'.3g') + '，满足≤1e-12。'
    report += '同时核对7701次窗口/4077次完整轨迹记录、窗口索引/物理边界/帧数及每残基窗口数。'
    report += '物理容差沿用原[-0.05,1.05]，不因换标签变更阈值。新float32目标范围为[' + format(summary['new_target_float32_min'],'.6f') + ', ' + format(summary['new_target_float32_max'],'.6f') + ']。\n\n'
    report += 'iRED生产时角色与最终MD-SOP角色有' + str(len(role_transitions)) + '条记录不同，本次保留的是最终1070/153/136角色，没有按旧生产角色重分。'
    report += '轨迹chain字段可能为SYSTEM，不能与权威蛋白链ID直接等同；本次按蛋白ID、完整序列哈希、序列位置和残基身份配对，两个chain字段分别保留。\n\n'
    report += '冻结的4265个原资产在创建前后均核对未变，v2.2步骤1两份冻结配方也未修改。S-OPPE及实验包未生成新版本或改写。\n\n'
    delta = summary['paired_difference_float64']
    report += '## 标签差异摘要\n\n'
    report += '在float64汇总口径下，新−旧平均差为' + format(delta['mean'],'.6f') + '，MAE=' + format(delta['mae'],'.6f') + '，RMSE=' + format(delta['rmse'],'.6f') + '。'
    report += '|差|>0.05的残基为' + str(delta['abs_above_005_count']) + '，>0.10的为' + str(delta['abs_above_010_count']) + '。这些是标签差异，不是模型预测性能。\n\n'
    report += '## 文件与访问边界\n\n```text\n' + str(RELEASE) + '/\n'
    report += '  md_ired_5tauc_targets_v2_2.parquet\n  label_transition_v2_2.tsv.gz\n  record_integrity_v2_2.tsv\n  partition_counts_v2_2.tsv\n  production_vs_final_role_notes_v2_2.tsv\n  label_release_manifest_v2_2.json\n```\n\n'
    report += 'parquet包含全部MD角色，仅供受信任的步骤3数据制备程序与审计器使用，**不能直接交给训练器**。步骤3须建立开发/测试隔离的NPZ与访问白名单。'
    report += '旧有效位置与mask的逐记录哈希保存在record_integrity表；每个标签及差值有来源包/原始表/冻结协议哈希可追溯。\n\n'
    report += '本次没有生成NPZ、训练manifest、data_binding或resolved_train配置，没有加载模型、训练或计算测试预测分数。下一步是步骤3数据封装与访问隔离。\n'
    write(A / 'label_replacement_audit_v2_2.md', report.encode('utf-8'))
    checksums = [{'relative_path':str(p.relative_to(P)), 'bytes':p.stat().st_size, 'sha256':sha(p)} for p in sorted(created)]
    for path in created:
        assert path.stat().st_uid == 1000 and path.stat().st_gid == 1000
    write(A / 'step2_artifact_sha256.tsv', encode_tsv(checksums))
    return collect()


def collect():
    summary = json.loads((A / 'label_replacement_audit_v2_2.json').read_text(encoding='utf-8'))
    manifest = rows(A / 'step2_artifact_sha256.tsv')
    buffer = io.BytesIO()
    included = []
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for r in manifest:
            path = P / r['relative_path']
            assert path.resolve() == path and TAG in path.parts and sha(path) == r['sha256']
            # Full target and transition tables remain on WQQ; local archival is
            # code, reports, per-record metadata, hashes, and the release manifest.
            if path.name.endswith('.parquet') or path.name.endswith('.tsv.gz'):
                continue
            z.write(path,str(path.relative_to(P)));included.append(str(path))
        checks = A / 'step2_artifact_sha256.tsv'
        z.write(checks,str(checks.relative_to(P)))
    return {'summary':summary, 'remote_artifact_count':len(manifest)+1, 'local_archived_metadata_files':len(included)+1,
            'large_tables_retained_on_wqq':True, 'archive_base64':base64.b64encode(buffer.getvalue()).decode('ascii')}
