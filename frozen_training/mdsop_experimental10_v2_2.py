#!/usr/bin/env python3
"""Fail-closed Experimental10 loader for MD-SOP v2.2 Step 7.

This module is intentionally separate from the Step-6 evaluator.  Step 6 keeps
its hard prohibition on Experimental10; this loader accepts exactly the one
frozen Experimental10 manifest after a Step-7 freeze has been verified by the
calling program.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Set, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from mdsop_evaluation_v2_2 import EvaluationAccessError, EvaluationIntegrityError, sha256_file


EXPERIMENTAL10_ENDPOINT = "experimental10"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "record_id", "source", "protein_id", "pdb_id", "chain_id",
        "role_v2_1", "partition_v2_1", "dataset_endpoint",
        "joint_homology_group", "sequence_sha256", "sequence_length",
        "esmc_path", "esmc_sha256", "hmm30_path", "hmm30_sha256",
        "logical_feature_dimension", "logical_feature_dtype", "label_path",
        "label_file_sha256", "raw_label_count", "qc_label_count",
        "loss_label_count", "train_access", "eval_access", "qc_rule",
        "allowed_label_access",
    }
)
REQUIRED_LABEL_ARRAYS = frozenset(
    {
        "y", "label_mask", "qc_mask", "raw_label_present", "train_access",
        "eval_access", "loss_mask", "length", "sequence_ascii",
    }
)


def resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_bool(value: str, field: str, record_id: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise EvaluationIntegrityError(
        f"{record_id}: {field} must be exactly True or False, got {value!r}"
    )


@dataclass(frozen=True)
class Experimental10AccessPolicy:
    whitelist_path: Path
    whitelist_sha256: str
    evaluation_root: Path
    freeze_registry_path: Optional[Path] = None

    def normalized(self) -> "Experimental10AccessPolicy":
        return Experimental10AccessPolicy(
            whitelist_path=resolved(self.whitelist_path),
            whitelist_sha256=str(self.whitelist_sha256).lower(),
            evaluation_root=resolved(self.evaluation_root),
            freeze_registry_path=self.freeze_registry_path,
        )


class Experimental10Dataset(Dataset):
    """Load the single frozen Experimental10 endpoint and no other endpoint."""

    def __init__(
        self,
        policy: Experimental10AccessPolicy,
        *,
        manifest_path: Path,
        manifest_sha256: str,
        expected_ids: Tuple[str, ...],
        expected_chains: Mapping[str, str],
        verify_component_hashes: bool = True,
    ) -> None:
        super().__init__()
        self.policy = policy.normalized()
        self.endpoint = EXPERIMENTAL10_ENDPOINT
        self.manifest_path = resolved(manifest_path)
        self.manifest_sha256 = str(manifest_sha256).lower()
        self.expected_ids = tuple(expected_ids)
        self.expected_chains = dict(expected_chains)
        self.verify_component_hashes = bool(verify_component_hashes)
        self._verified_paths: Set[Tuple[Path, str]] = set()
        self._validate_gate_and_manifest_identity()
        self.rows = tuple(self._read_and_validate_rows())
        observed_ids = tuple(row["protein_id"] for row in self.rows)
        if observed_ids != self.expected_ids:
            raise EvaluationIntegrityError(
                f"Experimental10 order/IDs differ: {observed_ids} != {self.expected_ids}"
            )

    def _validate_gate_and_manifest_identity(self) -> None:
        policy = self.policy
        from mdsop_runtime_v2_2 import PROJECT, RUN_TAG, require_frozen_models
        if policy.evaluation_root == PROJECT / "data" / RUN_TAG / "evaluation":
            if policy.freeze_registry_path is None:
                raise EvaluationAccessError("Frozen model registry required before Experimental10 access")
            require_frozen_models(policy.freeze_registry_path, EXPERIMENTAL10_ENDPOINT)
        if not SHA256_RE.fullmatch(policy.whitelist_sha256):
            raise EvaluationIntegrityError("Malformed final evaluator whitelist SHA-256")
        if not policy.whitelist_path.is_file():
            raise FileNotFoundError(policy.whitelist_path)
        if not is_within(policy.whitelist_path, policy.evaluation_root):
            raise EvaluationAccessError("Final evaluator whitelist escapes evaluation root")
        if sha256_file(policy.whitelist_path) != policy.whitelist_sha256:
            raise EvaluationAccessError("Final evaluator whitelist SHA-256 mismatch")
        whitelist = json.loads(policy.whitelist_path.read_text(encoding="utf-8"))
        entry = whitelist.get("endpoints", {}).get(EXPERIMENTAL10_ENDPOINT)
        if not isinstance(entry, Mapping):
            raise EvaluationAccessError("Whitelist has no Experimental10 entry")
        listed_manifest = resolved(Path(str(entry.get("record_manifest", ""))))
        listed_hash = str(entry.get("record_manifest_sha256", "")).lower()
        if listed_manifest != self.manifest_path or listed_hash != self.manifest_sha256:
            raise EvaluationAccessError("Requested manifest differs from frozen whitelist")
        canonical = resolved(policy.evaluation_root / EXPERIMENTAL10_ENDPOINT / "manifest.tsv")
        if self.manifest_path != canonical:
            raise EvaluationAccessError("Experimental10 manifest is not the canonical path")
        if not SHA256_RE.fullmatch(self.manifest_sha256):
            raise EvaluationIntegrityError("Malformed Experimental10 manifest SHA-256")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if sha256_file(self.manifest_path) != self.manifest_sha256:
            raise EvaluationAccessError("Experimental10 manifest SHA-256 mismatch")

    def _read_and_validate_rows(self) -> List[Dict[str, str]]:
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise EvaluationIntegrityError(f"Manifest lacks columns {sorted(missing)}")
            rows = [dict(row) for row in reader]
        seen: Set[str] = set()
        for row in rows:
            self._validate_row_before_label_open(row)
            record_id = row["record_id"]
            if record_id in seen:
                raise EvaluationIntegrityError(f"Duplicate record_id: {record_id}")
            seen.add(record_id)
        return rows

    def _validate_row_before_label_open(self, row: Mapping[str, str]) -> None:
        record_id = row.get("record_id", "<missing>")
        protein_id = row.get("protein_id", "")
        if row.get("dataset_endpoint") != EXPERIMENTAL10_ENDPOINT:
            raise EvaluationAccessError(f"{record_id}: endpoint mismatch")
        if row.get("source") != "historical_test":
            raise EvaluationAccessError(f"{record_id}: source is not historical_test")
        if row.get("role_v2_1") != "reference_experimental10":
            raise EvaluationAccessError(f"{record_id}: role is not reference_experimental10")
        if row.get("partition_v2_1") != "protected":
            raise EvaluationAccessError(f"{record_id}: partition is not protected")
        if row.get("role_v2_2") != "reference_experimental10" or row.get("partition_v2_2") != "protected":
            raise EvaluationAccessError(f"{record_id}: v2.2 protected role mismatch")
        if row.get("label_version") != "unchanged_709_position_historical_mask":
            raise EvaluationAccessError(f"{record_id}: experimental label version mismatch")
        if parse_bool(row.get("train_access", ""), "train_access", record_id):
            raise EvaluationAccessError(f"{record_id}: training access exposed")
        if not parse_bool(row.get("eval_access", ""), "eval_access", record_id):
            raise EvaluationAccessError(f"{record_id}: evaluation access absent")
        if row.get("allowed_label_access") != "final_experimental10_evaluation_only":
            raise EvaluationAccessError(f"{record_id}: label access policy mismatch")
        if row.get("qc_rule") != "frozen_historical_experimental10_mask":
            raise EvaluationIntegrityError(f"{record_id}: historical label mask rule changed")
        # The Step-3 endpoint manifest retains an empty legacy chain field; the
        # authoritative chains were frozen separately in Step 1.  Accept only
        # empty or the exact authoritative value, then emit the latter below.
        expected_chain = self.expected_chains.get(protein_id)
        if not expected_chain or row.get("chain_id") not in ("", expected_chain):
            raise EvaluationIntegrityError(f"{record_id}: authoritative chain mismatch")
        try:
            length = int(row.get("sequence_length", ""))
            raw_count = int(row.get("raw_label_count", ""))
            qc_count = int(row.get("qc_label_count", ""))
            loss_count = int(row.get("loss_label_count", ""))
            feature_dim = int(row.get("logical_feature_dimension", ""))
        except ValueError as exc:
            raise EvaluationIntegrityError(f"{record_id}: malformed integer field") from exc
        if length <= 0 or raw_count != qc_count or qc_count <= 0 or qc_count > length:
            raise EvaluationIntegrityError(f"{record_id}: invalid length/label counts")
        if loss_count != 0:
            raise EvaluationAccessError(f"{record_id}: reference labels enabled for loss")
        if feature_dim != 1182 or row.get("logical_feature_dtype") != "float32":
            raise EvaluationIntegrityError(f"{record_id}: feature schema changed")
        for field in ("sequence_sha256", "esmc_sha256", "hmm30_sha256", "label_file_sha256"):
            if not SHA256_RE.fullmatch(row.get(field, "")):
                raise EvaluationIntegrityError(f"{record_id}: malformed {field}")
        label_path = resolved(Path(row.get("label_path", "")))
        label_root = resolved(self.policy.evaluation_root / EXPERIMENTAL10_ENDPOINT / "labels")
        if not Path(row.get("label_path", "")).is_absolute() or not is_within(label_path, label_root):
            raise EvaluationAccessError(f"{record_id}: label path escapes endpoint label root")

    def __len__(self) -> int:
        return len(self.rows)

    def _verify_hash_once(self, path: Path, expected: str) -> None:
        key = (resolved(path), expected)
        if key in self._verified_paths:
            return
        observed = sha256_file(path)
        if observed != expected:
            raise EvaluationIntegrityError(
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
                raise EvaluationIntegrityError(f"{record_id}: missing component {path}")
        if self.verify_component_hashes:
            self._verify_hash_once(esmc_path, row["esmc_sha256"])
            self._verify_hash_once(hmm_path, row["hmm30_sha256"])
            self._verify_hash_once(label_path, row["label_file_sha256"])
        esmc = np.load(esmc_path, allow_pickle=False)
        hmm = np.load(hmm_path, allow_pickle=False)
        if esmc.shape != (length, 1152) or hmm.shape != (length, 30):
            raise EvaluationIntegrityError(f"{record_id}: feature shape mismatch")
        x_np = np.concatenate(
            (np.asarray(esmc, dtype=np.float32), np.asarray(hmm, dtype=np.float32)), axis=1
        )
        if not np.isfinite(x_np).all():
            raise EvaluationIntegrityError(f"{record_id}: non-finite feature")
        with np.load(label_path, allow_pickle=False) as archive:
            missing = REQUIRED_LABEL_ARRAYS - set(archive.files)
            if missing:
                raise EvaluationIntegrityError(f"{record_id}: label archive lacks {sorted(missing)}")
            arrays = {name: np.array(archive[name], copy=True) for name in REQUIRED_LABEL_ARRAYS}
        if arrays["length"].shape != (1,) or int(arrays["length"][0]) != length:
            raise EvaluationIntegrityError(f"{record_id}: label length mismatch")
        for name in REQUIRED_LABEL_ARRAYS - {"length"}:
            if arrays[name].shape != (length,):
                raise EvaluationIntegrityError(f"{record_id}: {name} shape mismatch")
        bool_names = ("label_mask", "qc_mask", "raw_label_present", "train_access", "eval_access", "loss_mask")
        if any(arrays[name].dtype != np.bool_ for name in bool_names):
            raise EvaluationIntegrityError(f"{record_id}: mask dtype mismatch")
        if arrays["y"].dtype != np.float32 or arrays["sequence_ascii"].dtype != np.uint8:
            raise EvaluationIntegrityError(f"{record_id}: label/sequence dtype mismatch")
        qc = arrays["qc_mask"]
        if not np.array_equal(qc, arrays["label_mask"]) or not np.array_equal(qc, arrays["raw_label_present"]):
            raise EvaluationIntegrityError(f"{record_id}: frozen Experimental10 masks differ")
        if int(qc.sum()) != int(row["qc_label_count"]):
            raise EvaluationIntegrityError(f"{record_id}: eligible label count mismatch")
        if np.any(arrays["train_access"]) or not np.all(arrays["eval_access"]) or np.any(arrays["loss_mask"]):
            raise EvaluationAccessError(f"{record_id}: per-residue access flags violate reference role")
        if not np.isfinite(arrays["y"][qc]).all() or np.any(arrays["y"][~qc] != 0.0):
            raise EvaluationIntegrityError(f"{record_id}: invalid label values/placeholders")
        sequence = bytes(arrays["sequence_ascii"].tolist()).decode("ascii")
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != row["sequence_sha256"]:
            raise EvaluationIntegrityError(f"{record_id}: sequence SHA-256 mismatch")
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x_np)),
            "y": torch.from_numpy(arrays["y"]),
            "qc_mask": torch.from_numpy(qc),
            "length": length,
            "record_id": record_id,
            "source": row["source"],
            "protein_id": row["protein_id"],
            "pdb_id": row["pdb_id"],
            "chain_id": self.expected_chains[row["protein_id"]],
            "endpoint": EXPERIMENTAL10_ENDPOINT,
            "joint_homology_group": row["joint_homology_group"],
            "sequence_sha256": row["sequence_sha256"],
            "sequence": sequence,
        }
