"""Fail-closed development manifest guard. No model or label archive is opened.

Step4 must use this guard (or enforce equivalent validated checks) before any
dataset label read. Feature inputs may be immutable external paths; labels may
only be inside the current version's development-label root.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re


class AccessViolationError(PermissionError):
    pass


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_training_row(row, policy):
    role = row.get('role_v2_2')
    source = row.get('source')
    if role not in policy['allowed_roles'] or source not in policy['allowed_sources']:
        raise AccessViolationError('Forbidden role or source')
    if row.get('role_v2_1') != role or row.get('partition_v2_2') != role or row.get('partition_v2_1') != role:
        raise AccessViolationError('Role aliases disagree')
    endpoint = ('md_' if source == 'md_ired' else 'soppe_') + role
    if row.get('dataset_endpoint') != endpoint or row.get('endpoint_kind') != 'primary':
        raise AccessViolationError('Endpoint is not an allowed development endpoint')
    path = Path(row.get('label_path', ''))
    if not path.is_absolute() or not inside(path, policy['development_label_root']):
        raise AccessViolationError('Label escaped current development-label root')
    if inside(path, policy['evaluation_root']) or inside(path, policy['master_label_release_root']):
        raise AccessViolationError('Evaluation/master label source forbidden')
    if any(s.lower() in ('evaluation', 'label_releases') for s in path.resolve().parts):
        raise AccessViolationError('Forbidden path component')
    source_dir = 'md' if source == 'md_ired' else 'soppe'
    expected_path = Path(policy['development_label_root']) / source_dir / role / (row['record_id'] + '.npz')
    if path.resolve() != expected_path.resolve():
        raise AccessViolationError('Label path does not match this record/source/role')
    if row.get('label_version') != policy['label_version_by_source'][source]:
        raise AccessViolationError('Wrong target version')
    expected_flags = ('True', 'False') if role == 'train' else ('False', 'True')
    if (row.get('train_access'), row.get('eval_access')) != expected_flags:
        raise AccessViolationError('Source access flags disagree with role')
    expected_use = 'training_loss' if role == 'train' else 'early_stopping_and_checkpoint_selection'
    if row.get('allowed_label_access') != expected_use:
        raise AccessViolationError('Wrong permitted label use')
    qc = int(row['qc_label_count']); loss = int(row['loss_label_count'])
    if qc < 0 or loss != (qc if role == 'train' else 0):
        raise AccessViolationError('Loss counts disagree with role')
    if row.get('logical_feature_dimension') != '1182' or row.get('logical_feature_dtype') != 'float32':
        raise AccessViolationError('Feature schema changed')
    for key in ['esmc_sha256', 'hmm30_sha256', 'label_file_sha256']:
        if not re.fullmatch('[0-9a-f]{64}', row.get(key, '')):
            raise AccessViolationError('Malformed component hash')


def load_trainer_rows(policy_path, manifest_path=None, roles=('train','validation'), sources=('md_ired','soppe_legacy'), drop_zero_supervision=False):
    policy = json.loads(Path(policy_path).read_text(encoding='utf-8'))
    if not set(roles) or not set(roles).issubset(policy['allowed_roles']):
        raise AccessViolationError('Forbidden requested role')
    if not set(sources) or not set(sources).issubset(policy['allowed_sources']):
        raise AccessViolationError('Forbidden requested source')
    manifest = Path(manifest_path or policy['trainer_manifest']).resolve()
    if manifest != Path(policy['trainer_manifest']).resolve():
        raise AccessViolationError('Only the exact frozen trainer manifest is allowed')
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != policy['trainer_manifest_sha256']:
        raise AccessViolationError('Trainer manifest checksum mismatch')
    with manifest.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    seen = set()
    for row in rows:
        validate_training_row(row, policy)
        if row['record_id'] in seen:
            raise AccessViolationError('Duplicate record ID in development manifest')
        seen.add(row['record_id'])
    return [r for r in rows if r['role_v2_2'] in roles and r['source'] in sources
            and (not drop_zero_supervision or int(r['qc_label_count']) > 0)]
