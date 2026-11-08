#!/usr/bin/env python3
"""Access-controlled MD-SOP v2.2 development dataset.

Only the frozen development trainer manifest is accepted.  Every manifest row is
validated before any label archive is opened, so a test/reference role or an
evaluation-label path fails closed.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from access_manifest_guard_v2_2 import validate_training_row, AccessViolationError as GuardAccessError


FORMAL_MANIFEST_SHA256 = "bde765b3ff9a7689255cae6fbb529c390636e2f8121cc38afb744fa656da234f"
FORMAL_MANIFEST_BASENAME = "trainer_access_manifest.tsv"
ALLOWED_ROLES = frozenset({"train", "validation"})
ALLOWED_SOURCES = frozenset({"md_ired", "soppe_legacy"})
EXPECTED_ENDPOINTS = {
    ("md_ired", "train"): "md_train",
    ("md_ired", "validation"): "md_validation",
    ("soppe_legacy", "train"): "soppe_train",
    ("soppe_legacy", "validation"): "soppe_validation",
}
REQUIRED_COLUMNS = frozenset(
    {
        "record_id",
        "source",
        "role_v2_1",
        "partition_v2_1",
        "role_v2_2",
        "partition_v2_2",
        "label_version",
        "endpoint_kind",
        "dataset_endpoint",
        "sequence_length",
        "esmc_path",
        "esmc_sha256",
        "hmm30_path",
        "hmm30_sha256",
        "logical_feature_dimension",
        "logical_feature_dtype",
        "label_path",
        "label_file_sha256",
        "qc_label_count",
        "loss_label_count",
        "train_access",
        "eval_access",
        "allowed_label_access",
    }
)
REQUIRED_LABEL_ARRAYS = frozenset(
    {
        "y",
        "label_mask",
        "qc_mask",
        "raw_label_present",
        "train_access",
        "eval_access",
        "loss_mask",
        "length",
        "sequence_ascii",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AccessViolationError(PermissionError):
    """Raised before protected or out-of-policy label data can be opened."""


class DatasetIntegrityError(RuntimeError):
    """Raised when a permitted development record violates its frozen schema."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: str, field: str, record_id: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise DatasetIntegrityError(
        f"{record_id}: {field} must be exactly True or False, got {value!r}"
    )


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class ManifestAccessPolicy:
    """Immutable allow-list for one development manifest and its label root."""

    manifest_path: Path
    manifest_sha256: str
    development_label_root: Path
    evaluation_root: Path
    expected_basename: str = FORMAL_MANIFEST_BASENAME

    def normalized(self) -> "ManifestAccessPolicy":
        return ManifestAccessPolicy(
            manifest_path=_resolved(Path(self.manifest_path)),
            manifest_sha256=self.manifest_sha256.lower(),
            development_label_root=_resolved(Path(self.development_label_root)),
            evaluation_root=_resolved(Path(self.evaluation_root)),
            expected_basename=self.expected_basename,
        )


class AccessControlledProteinDataset(Dataset):
    """Load ESM-C+HMM inputs and development labels under a fail-closed policy."""

    def __init__(
        self,
        policy: ManifestAccessPolicy,
        roles: Sequence[str],
        sources: Optional[Sequence[str]] = None,
        *,
        drop_zero_supervision: bool = True,
        verify_component_hashes: bool = False,
    ) -> None:
        super().__init__()
        self.policy = policy.normalized()
        self.roles = frozenset(roles)
        self.sources = frozenset(sources or ALLOWED_SOURCES)
        self.drop_zero_supervision = bool(drop_zero_supervision)
        self.verify_component_hashes = bool(verify_component_hashes)
        self._verified_paths: Set[Tuple[Path, str]] = set()

        if not self.roles or not self.roles.issubset(ALLOWED_ROLES):
            raise AccessViolationError(
                f"Requested roles must be a non-empty subset of {sorted(ALLOWED_ROLES)}"
            )
        if not self.sources or not self.sources.issubset(ALLOWED_SOURCES):
            raise AccessViolationError(
                f"Requested sources must be a non-empty subset of {sorted(ALLOWED_SOURCES)}"
            )
        self._validate_manifest_identity()
        all_rows = self._read_and_validate_all_rows()

        matching = [
            row
            for row in all_rows
            if row["role_v2_1"] in self.roles and row["source"] in self.sources
        ]
        self.excluded_zero_supervision_records = tuple(
            row["record_id"] for row in matching if int(row["qc_label_count"]) == 0
        )
        if self.drop_zero_supervision:
            matching = [row for row in matching if int(row["qc_label_count"]) > 0]
        self.rows = tuple(matching)
        if not self.rows:
            raise DatasetIntegrityError("No records remain after the requested development filters")

    def _validate_manifest_identity(self) -> None:
        path = self.policy.manifest_path
        if path.name != self.policy.expected_basename:
            raise AccessViolationError(
                f"Only {self.policy.expected_basename!r} is accepted, got {path.name!r}"
            )
        if not SHA256_RE.fullmatch(self.policy.manifest_sha256):
            raise DatasetIntegrityError("Expected manifest SHA-256 is malformed")
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != self.policy.manifest_sha256:
            raise AccessViolationError(
                f"Trainer manifest hash mismatch: expected {self.policy.manifest_sha256}, "
                f"observed {observed}"
            )

    def _read_and_validate_all_rows(self) -> List[Dict[str, str]]:
        with self.policy.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise DatasetIntegrityError(f"Trainer manifest lacks columns: {sorted(missing)}")
            rows = [dict(row) for row in reader]
        if not rows:
            raise DatasetIntegrityError("Trainer manifest is empty")

        seen: Set[str] = set()
        for row in rows:
            self._validate_row_before_label_open(row)
            record_id = row["record_id"]
            if record_id in seen:
                raise DatasetIntegrityError(f"Duplicate record_id: {record_id}")
            seen.add(record_id)
        return rows

    def _validate_row_before_label_open(self, row: Mapping[str, str]) -> None:
        record_id = row.get("record_id", "<missing-record-id>")
        role = row.get("role_v2_1", "")
        source = row.get("source", "")
        if role not in ALLOWED_ROLES:
            raise AccessViolationError(f"{record_id}: forbidden trainer role {role!r}")
        if source not in ALLOWED_SOURCES:
            raise AccessViolationError(f"{record_id}: forbidden source {source!r}")
        if row.get("partition_v2_1") != role:
            raise DatasetIntegrityError(
                f"{record_id}: partition {row.get('partition_v2_1')!r} != role {role!r}"
            )
        expected_endpoint = EXPECTED_ENDPOINTS[(source, role)]
        if row.get("dataset_endpoint") != expected_endpoint:
            raise AccessViolationError(
                f"{record_id}: endpoint {row.get('dataset_endpoint')!r} is not {expected_endpoint!r}"
            )

        train_access = _parse_bool(row.get("train_access", ""), "train_access", record_id)
        eval_access = _parse_bool(row.get("eval_access", ""), "eval_access", record_id)
        expected_access = (True, False) if role == "train" else (False, True)
        if (train_access, eval_access) != expected_access:
            raise AccessViolationError(
                f"{record_id}: access flags {(train_access, eval_access)} disagree with role {role}"
            )
        expected_label_access = (
            "training_loss"
            if role == "train"
            else "early_stopping_and_checkpoint_selection"
        )
        if row.get("allowed_label_access") != expected_label_access:
            raise AccessViolationError(
                f"{record_id}: allowed_label_access must be {expected_label_access!r}"
            )

        label_path = Path(row.get("label_path", ""))
        if not label_path.is_absolute():
            raise AccessViolationError(f"{record_id}: label path must be absolute")
        resolved_label = _resolved(label_path)
        if _is_within(resolved_label, self.policy.evaluation_root):
            raise AccessViolationError(f"{record_id}: evaluation labels are forbidden")
        if not _is_within(resolved_label, self.policy.development_label_root):
            raise AccessViolationError(
                f"{record_id}: label path escapes the development label allow-list"
            )
        if "evaluation" in {part.lower() for part in resolved_label.parts}:
            raise AccessViolationError(f"{record_id}: path contains forbidden evaluation component")

        try:
            length = int(row.get("sequence_length", ""))
            qc_count = int(row.get("qc_label_count", ""))
            loss_count = int(row.get("loss_label_count", ""))
            feature_dim = int(row.get("logical_feature_dimension", ""))
        except ValueError as exc:
            raise DatasetIntegrityError(f"{record_id}: malformed integer field") from exc
        if length <= 0 or qc_count < 0 or loss_count < 0:
            raise DatasetIntegrityError(f"{record_id}: invalid length/label count")
        if feature_dim != 1182 or row.get("logical_feature_dtype") != "float32":
            raise DatasetIntegrityError(f"{record_id}: logical feature schema is not [L,1182] float32")
        if role == "train" and loss_count != qc_count:
            raise DatasetIntegrityError(f"{record_id}: training loss count must equal QC count")
        if role == "validation" and loss_count != 0:
            raise AccessViolationError(f"{record_id}: validation labels must not be enabled for loss")
        for field in ("esmc_sha256", "hmm30_sha256", "label_file_sha256"):
            if not SHA256_RE.fullmatch(row.get(field, "")):
                raise DatasetIntegrityError(f"{record_id}: malformed {field}")
        guard_policy = {
            "allowed_roles": sorted(ALLOWED_ROLES), "allowed_sources": sorted(ALLOWED_SOURCES),
            "development_label_root": str(self.policy.development_label_root),
            "evaluation_root": str(self.policy.evaluation_root),
            "master_label_release_root": str(self.policy.development_label_root.parent.parent / "label_releases"),
            "label_version_by_source": {"md_ired": "md_ired_5tauc_windowmean_v2_2", "soppe_legacy": "unchanged_frozen_v2_1_SOPPE_targets"},
        }
        try:
            validate_training_row(row, guard_policy)
        except GuardAccessError as exc:
            raise AccessViolationError(str(exc)) from exc

    def __len__(self) -> int:
        return len(self.rows)

    def _verify_hash_once(self, path: Path, expected: str) -> None:
        key = (_resolved(path), expected)
        if key in self._verified_paths:
            return
        observed = sha256_file(path)
        if observed != expected:
            raise DatasetIntegrityError(
                f"Component hash mismatch for {path}: expected {expected}, observed {observed}"
            )
        self._verified_paths.add(key)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        record_id = row["record_id"]
        length = int(row["sequence_length"])
        esmc_path = Path(row["esmc_path"])
        hmm_path = Path(row["hmm30_path"])
        label_path = Path(row["label_path"])
        for path in (esmc_path, hmm_path, label_path):
            if not path.is_file():
                raise DatasetIntegrityError(f"{record_id}: component is missing: {path}")
        if self.verify_component_hashes:
            self._verify_hash_once(esmc_path, row["esmc_sha256"])
            self._verify_hash_once(hmm_path, row["hmm30_sha256"])
            self._verify_hash_once(label_path, row["label_file_sha256"])
        # Target identity is always checked, even when large feature hashing is
        # delegated to the immutable data-release audit for training throughput.
        self._verify_hash_once(label_path, row["label_file_sha256"])

        esmc = np.load(esmc_path, allow_pickle=False)
        hmm = np.load(hmm_path, allow_pickle=False)
        if esmc.shape != (length, 1152) or hmm.shape != (length, 30):
            raise DatasetIntegrityError(
                f"{record_id}: component shapes are {esmc.shape} and {hmm.shape}, expected "
                f"{(length, 1152)} and {(length, 30)}"
            )
        if not np.issubdtype(esmc.dtype, np.floating) or not np.issubdtype(hmm.dtype, np.floating):
            raise DatasetIntegrityError(f"{record_id}: feature components must be floating point")
        x_np = np.concatenate(
            (np.asarray(esmc, dtype=np.float32), np.asarray(hmm, dtype=np.float32)), axis=1
        )
        if not np.isfinite(x_np).all():
            raise DatasetIntegrityError(f"{record_id}: feature tensor contains non-finite values")

        with np.load(label_path, allow_pickle=False) as archive:
            missing = REQUIRED_LABEL_ARRAYS - set(archive.files)
            if missing:
                raise DatasetIntegrityError(f"{record_id}: label archive lacks {sorted(missing)}")
            arrays = {name: np.array(archive[name], copy=True) for name in REQUIRED_LABEL_ARRAYS}
        if arrays["length"].shape != (1,) or int(arrays["length"][0]) != length:
            raise DatasetIntegrityError(f"{record_id}: label archive length mismatch")
        one_dimensional = (
            "y",
            "label_mask",
            "qc_mask",
            "raw_label_present",
            "train_access",
            "eval_access",
            "loss_mask",
            "sequence_ascii",
        )
        if any(arrays[name].shape != (length,) for name in one_dimensional):
            raise DatasetIntegrityError(f"{record_id}: label arrays do not all have length {length}")
        bool_names = (
            "label_mask",
            "qc_mask",
            "raw_label_present",
            "train_access",
            "eval_access",
            "loss_mask",
        )
        if any(arrays[name].dtype != np.bool_ for name in bool_names):
            raise DatasetIntegrityError(f"{record_id}: mask arrays must have bool dtype")
        if arrays["y"].dtype != np.float32:
            raise DatasetIntegrityError(f"{record_id}: y must have float32 dtype")

        qc = arrays["qc_mask"]
        label_mask = arrays["label_mask"]
        train_access = arrays["train_access"]
        eval_access = arrays["eval_access"]
        stored_loss = arrays["loss_mask"]
        if np.any(qc & ~label_mask):
            raise DatasetIntegrityError(f"{record_id}: qc_mask is not a subset of label_mask")
        if int(qc.sum()) != int(row["qc_label_count"]):
            raise DatasetIntegrityError(f"{record_id}: QC label count disagrees with manifest")
        expected_train = row["role_v2_1"] == "train"
        if not np.all(train_access == expected_train) or not np.all(eval_access == (not expected_train)):
            raise AccessViolationError(f"{record_id}: per-residue access arrays disagree with role")
        computed_loss = qc & train_access
        computed_eval = qc & eval_access
        if not np.array_equal(stored_loss, computed_loss):
            raise DatasetIntegrityError(f"{record_id}: stored loss_mask violates qc AND train_access")
        if int(computed_loss.sum()) != int(row["loss_label_count"]):
            raise DatasetIntegrityError(f"{record_id}: loss label count disagrees with manifest")
        if not np.isfinite(arrays["y"][qc]).all():
            raise DatasetIntegrityError(f"{record_id}: a QC-eligible label is non-finite")

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x_np)),
            "y": torch.from_numpy(arrays["y"]),
            "label_mask": torch.from_numpy(label_mask),
            "qc_mask": torch.from_numpy(qc),
            "train_access": torch.from_numpy(train_access),
            "eval_access": torch.from_numpy(eval_access),
            "loss_mask": torch.from_numpy(computed_loss),
            "eval_mask": torch.from_numpy(computed_eval),
            "length": length,
            "record_id": record_id,
            "source": row["source"],
            "role": row["role_v2_1"],
            "dataset_endpoint": row["dataset_endpoint"],
            "sequence_sha256": row.get("sequence_sha256", ""),
        }


def collate_protein_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Dynamically pad complete chains; padding is distinct from unlabeled residues."""

    if not records:
        raise ValueError("Cannot collate an empty record list")
    lengths = torch.tensor([int(record["length"]) for record in records], dtype=torch.long)
    if torch.any(lengths <= 0):
        raise DatasetIntegrityError("All record lengths must be positive")
    batch_size = len(records)
    max_length = int(lengths.max().item())
    input_dim = int(records[0]["x"].shape[1])  # type: ignore[index,union-attr]
    x = torch.zeros((batch_size, max_length, input_dim), dtype=torch.float32)
    y = torch.zeros((batch_size, max_length), dtype=torch.float32)
    bool_fields = (
        "label_mask",
        "qc_mask",
        "train_access",
        "eval_access",
        "loss_mask",
        "eval_mask",
    )
    masks = {name: torch.zeros((batch_size, max_length), dtype=torch.bool) for name in bool_fields}
    residue_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    for row_index, record in enumerate(records):
        length = int(record["length"])
        record_x = record["x"]
        record_y = record["y"]
        if not isinstance(record_x, torch.Tensor) or record_x.shape != (length, input_dim):
            raise DatasetIntegrityError("Feature tensor shape changed before collation")
        if not isinstance(record_y, torch.Tensor) or record_y.shape != (length,):
            raise DatasetIntegrityError("Label tensor shape changed before collation")
        x[row_index, :length] = record_x
        y[row_index, :length] = record_y
        residue_mask[row_index, :length] = True
        for name in bool_fields:
            value = record[name]
            if not isinstance(value, torch.Tensor) or value.shape != (length,):
                raise DatasetIntegrityError(f"Mask {name} changed before collation")
            masks[name][row_index, :length] = value

    if torch.any(masks["loss_mask"] & ~residue_mask):
        raise DatasetIntegrityError("A training loss position falls in padding")
    if torch.any(masks["eval_mask"] & ~residue_mask):
        raise DatasetIntegrityError("An evaluation position falls in padding")

    return {
        "x": x,
        "y": y,
        **masks,
        "residue_mask": residue_mask,
        "padding_mask": ~residue_mask,
        "lengths": lengths,
        "record_ids": [str(record["record_id"]) for record in records],
        "sources": [str(record["source"]) for record in records],
        "roles": [str(record["role"]) for record in records],
        "dataset_endpoints": [str(record["dataset_endpoint"]) for record in records],
    }


def rows_by_source_role(rows: Iterable[Mapping[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = f"{row['source']}::{row['role_v2_1']}"
        counts[key] = counts.get(key, 0) + 1
    return counts
