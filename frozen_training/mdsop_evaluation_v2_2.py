#!/usr/bin/env python3
"""Fail-closed evaluation utilities for MD-SOP v2.2 Step 6.

The development DataLoader deliberately cannot open test labels.  This module is
the separate final-evaluation gate: it accepts only manifests named in the
frozen evaluator whitelist and only endpoints explicitly enabled by the Step-6
configuration.  Experimental10 is therefore inaccessible during Step 6.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model_bilstm_window_v2_2 import (
    MDSoPBiLSTMRegressor,
    state_dict_sha256,
)


TRAINING_CHECKPOINT_FORMAT = "md_sop_training_checkpoint_v2_2"
EXPERIMENT_IDS = (
    "A_MD_w1",
    "A_MD_w15",
    "B_SOPPE_w1",
    "B_SOPPE_w15",
    "C_MIX_w1",
    "C_MIX_w15",
)
STEP6_ENDPOINTS = ("md_test", "soppe_test", "md_test_full100ns_reference")
FORBIDDEN_STEP6_ENDPOINTS = frozenset({"experimental10"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_MANIFEST_COLUMNS = frozenset(
    {
        "record_id",
        "source",
        "protein_id",
        "pdb_id",
        "chain_id",
        "role_v2_1",
        "partition_v2_1",
        "dataset_endpoint",
        "joint_homology_group",
        "sequence_sha256",
        "sequence_length",
        "esmc_path",
        "esmc_sha256",
        "hmm30_path",
        "hmm30_sha256",
        "logical_feature_dimension",
        "logical_feature_dtype",
        "label_path",
        "label_file_sha256",
        "raw_label_count",
        "qc_label_count",
        "loss_label_count",
        "train_access",
        "eval_access",
        "qc_rule",
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


class EvaluationAccessError(PermissionError):
    """Raised before an endpoint or label outside the frozen gate can be opened."""


class EvaluationIntegrityError(RuntimeError):
    """Raised when a permitted evaluation record violates the frozen schema."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_bool(value: str, field: str, record_id: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise EvaluationIntegrityError(
        f"{record_id}: {field} must be exactly True or False, got {value!r}"
    )


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: object) -> None:
    safe = _json_safe(value)
    atomic_write_text(
        Path(path),
        json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n",
    )


def atomic_write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _format_tsv_value(row.get(field)) for field in fields})
    atomic_write_text(Path(path), buffer.getvalue())


def _format_tsv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:.12g}" if math.isfinite(value) else ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return value


def atomic_write_gzip_tsv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    """Write a deterministic gzip TSV (mtime=0) via an atomic rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(descriptor)
    try:
        with open(temporary_name, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_handle:
                    writer = csv.DictWriter(
                        text_handle,
                        fieldnames=list(fields),
                        delimiter="\t",
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(
                            {field: _format_tsv_value(row.get(field)) for field in fields}
                        )
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class EvaluationAccessPolicy:
    whitelist_path: Path
    whitelist_sha256: str
    evaluation_root: Path
    allowed_endpoints: Tuple[str, ...] = STEP6_ENDPOINTS
    freeze_registry_path: Optional[Path] = None

    def normalized(self) -> "EvaluationAccessPolicy":
        return EvaluationAccessPolicy(
            whitelist_path=_resolved(self.whitelist_path),
            whitelist_sha256=str(self.whitelist_sha256).lower(),
            evaluation_root=_resolved(self.evaluation_root),
            allowed_endpoints=tuple(self.allowed_endpoints),
            freeze_registry_path=self.freeze_registry_path,
        )


class EvaluationProteinDataset(Dataset):
    """Load exactly one frozen test endpoint after whitelist validation."""

    def __init__(
        self,
        policy: EvaluationAccessPolicy,
        *,
        endpoint: str,
        manifest_path: Path,
        manifest_sha256: str,
        expected_source: str,
        verify_component_hashes: bool = True,
    ) -> None:
        super().__init__()
        self.policy = policy.normalized()
        self.endpoint = str(endpoint)
        self.manifest_path = _resolved(Path(manifest_path))
        self.manifest_sha256 = str(manifest_sha256).lower()
        self.expected_source = str(expected_source)
        self.verify_component_hashes = bool(verify_component_hashes)
        self._verified_paths: Set[Tuple[Path, str]] = set()

        self._validate_gate_and_manifest_identity()
        self.rows = tuple(self._read_and_validate_rows())
        if not self.rows:
            raise EvaluationIntegrityError(f"{self.endpoint}: evaluation manifest is empty")

    def _validate_gate_and_manifest_identity(self) -> None:
        policy = self.policy
        from mdsop_runtime_v2_2 import PROJECT, RUN_TAG, require_frozen_models
        if policy.evaluation_root == PROJECT / "data" / RUN_TAG / "evaluation":
            if policy.freeze_registry_path is None:
                raise EvaluationAccessError("Frozen model registry required before production test access")
            require_frozen_models(policy.freeze_registry_path, self.endpoint)
        if self.endpoint in FORBIDDEN_STEP6_ENDPOINTS:
            raise EvaluationAccessError(f"{self.endpoint} is forbidden during Step 6")
        if self.endpoint not in policy.allowed_endpoints or self.endpoint not in STEP6_ENDPOINTS:
            raise EvaluationAccessError(f"Endpoint {self.endpoint!r} is not enabled for Step 6")
        if set(policy.allowed_endpoints) != set(STEP6_ENDPOINTS):
            raise EvaluationAccessError("Step-6 policy must enable both source tests and the full100ns reference")
        if not SHA256_RE.fullmatch(policy.whitelist_sha256):
            raise EvaluationIntegrityError("Malformed evaluator whitelist SHA-256")
        if not policy.whitelist_path.is_file():
            raise FileNotFoundError(policy.whitelist_path)
        if not _is_within(policy.whitelist_path, policy.evaluation_root):
            raise EvaluationAccessError("Evaluator whitelist escapes the evaluation root")
        observed_whitelist_hash = sha256_file(policy.whitelist_path)
        if observed_whitelist_hash != policy.whitelist_sha256:
            raise EvaluationAccessError("Evaluator whitelist SHA-256 mismatch")

        whitelist = json.loads(policy.whitelist_path.read_text(encoding="utf-8"))
        entry = whitelist.get("endpoints", {}).get(self.endpoint)
        if not isinstance(entry, Mapping):
            raise EvaluationAccessError(f"Whitelist has no entry for {self.endpoint}")
        listed_manifest = _resolved(Path(str(entry.get("record_manifest", ""))))
        listed_hash = str(entry.get("record_manifest_sha256", "")).lower()
        if listed_manifest != self.manifest_path or listed_hash != self.manifest_sha256:
            raise EvaluationAccessError("Requested manifest differs from the frozen whitelist")
        expected_endpoint_root = _resolved(policy.evaluation_root / self.endpoint)
        if self.manifest_path != expected_endpoint_root / "manifest.tsv":
            raise EvaluationAccessError("Endpoint manifest path is not the canonical frozen path")
        if not SHA256_RE.fullmatch(self.manifest_sha256):
            raise EvaluationIntegrityError("Malformed endpoint manifest SHA-256")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        observed_manifest_hash = sha256_file(self.manifest_path)
        if observed_manifest_hash != self.manifest_sha256:
            raise EvaluationAccessError("Endpoint manifest SHA-256 mismatch")

    def _read_and_validate_rows(self) -> List[Dict[str, str]]:
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_MANIFEST_COLUMNS - columns
            if missing:
                raise EvaluationIntegrityError(
                    f"{self.endpoint}: manifest lacks columns {sorted(missing)}"
                )
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
        record_id = row.get("record_id", "<missing-record-id>")
        if row.get("dataset_endpoint") != self.endpoint:
            raise EvaluationAccessError(f"{record_id}: endpoint mismatch")
        if row.get("source") != self.expected_source:
            raise EvaluationAccessError(f"{record_id}: source mismatch")
        if row.get("role_v2_1") != "test" or row.get("partition_v2_1") != "test":
            raise EvaluationAccessError(f"{record_id}: only frozen test roles are accepted")
        if row.get("role_v2_2") != "test" or row.get("partition_v2_2") != "test":
            raise EvaluationAccessError(f"{record_id}: v2.2 test role mismatch")
        expected_version = {"md_test": "md_ired_5tauc_windowmean_v2_2", "soppe_test": "unchanged_frozen_v2_1_SOPPE_targets",
                            "md_test_full100ns_reference": "md_ired_full100ns_three_replica_mean_v2_1"}[self.endpoint]
        if row.get("label_version") != expected_version:
            raise EvaluationAccessError(f"{record_id}: evaluation target version mismatch")
        if _parse_bool(row.get("train_access", ""), "train_access", record_id):
            raise EvaluationAccessError(f"{record_id}: test record exposes training access")
        if not _parse_bool(row.get("eval_access", ""), "eval_access", record_id):
            raise EvaluationAccessError(f"{record_id}: test record lacks evaluation access")
        expected_use = "auxiliary_full100ns_evaluation_only" if self.endpoint == "md_test_full100ns_reference" else "final_evaluation_only_after_checkpoint_freeze"
        if row.get("allowed_label_access") != expected_use:
            raise EvaluationAccessError(f"{record_id}: invalid allowed_label_access")

        try:
            length = int(row.get("sequence_length", ""))
            raw_count = int(row.get("raw_label_count", ""))
            qc_count = int(row.get("qc_label_count", ""))
            loss_count = int(row.get("loss_label_count", ""))
            feature_dim = int(row.get("logical_feature_dimension", ""))
        except ValueError as exc:
            raise EvaluationIntegrityError(f"{record_id}: malformed integer field") from exc
        if length <= 0 or raw_count < 0 or qc_count < 0 or qc_count > raw_count:
            raise EvaluationIntegrityError(f"{record_id}: invalid length or label counts")
        if loss_count != 0:
            raise EvaluationAccessError(f"{record_id}: test labels must not be enabled for loss")
        if feature_dim != 1182 or row.get("logical_feature_dtype") != "float32":
            raise EvaluationIntegrityError(f"{record_id}: feature schema is not [L,1182] float32")
        if not SHA256_RE.fullmatch(row.get("sequence_sha256", "")):
            raise EvaluationIntegrityError(f"{record_id}: malformed sequence SHA-256")
        for field in ("esmc_sha256", "hmm30_sha256", "label_file_sha256"):
            if not SHA256_RE.fullmatch(row.get(field, "")):
                raise EvaluationIntegrityError(f"{record_id}: malformed {field}")

        label_path = _resolved(Path(row.get("label_path", "")))
        label_root = _resolved(self.policy.evaluation_root / self.endpoint / "labels")
        if not Path(row.get("label_path", "")).is_absolute() or not _is_within(label_path, label_root):
            raise EvaluationAccessError(f"{record_id}: label path escapes the endpoint label root")
        if "experimental10" in {part.lower() for part in label_path.parts}:
            raise EvaluationAccessError(f"{record_id}: Experimental10 path is forbidden")

    def __len__(self) -> int:
        return len(self.rows)

    def _verify_hash_once(self, path: Path, expected: str) -> None:
        key = (_resolved(path), expected)
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
                raise EvaluationIntegrityError(f"{record_id}: component is missing: {path}")
        if self.verify_component_hashes:
            self._verify_hash_once(esmc_path, row["esmc_sha256"])
            self._verify_hash_once(hmm_path, row["hmm30_sha256"])
            self._verify_hash_once(label_path, row["label_file_sha256"])

        esmc = np.load(esmc_path, allow_pickle=False)
        hmm = np.load(hmm_path, allow_pickle=False)
        if esmc.shape != (length, 1152) or hmm.shape != (length, 30):
            raise EvaluationIntegrityError(
                f"{record_id}: component shapes {esmc.shape}/{hmm.shape} do not match length"
            )
        if not np.issubdtype(esmc.dtype, np.floating) or not np.issubdtype(hmm.dtype, np.floating):
            raise EvaluationIntegrityError(f"{record_id}: features are not floating point")
        x_np = np.concatenate(
            (np.asarray(esmc, dtype=np.float32), np.asarray(hmm, dtype=np.float32)), axis=1
        )
        if not np.isfinite(x_np).all():
            raise EvaluationIntegrityError(f"{record_id}: feature tensor is non-finite")

        with np.load(label_path, allow_pickle=False) as archive:
            missing = REQUIRED_LABEL_ARRAYS - set(archive.files)
            if missing:
                raise EvaluationIntegrityError(f"{record_id}: label archive lacks {sorted(missing)}")
            arrays = {name: np.array(archive[name], copy=True) for name in REQUIRED_LABEL_ARRAYS}
        if arrays["length"].shape != (1,) or int(arrays["length"][0]) != length:
            raise EvaluationIntegrityError(f"{record_id}: label length mismatch")
        one_dimensional = REQUIRED_LABEL_ARRAYS - {"length"}
        if any(arrays[name].shape != (length,) for name in one_dimensional):
            raise EvaluationIntegrityError(f"{record_id}: label arrays do not all have length L")
        bool_names = (
            "label_mask",
            "qc_mask",
            "raw_label_present",
            "train_access",
            "eval_access",
            "loss_mask",
        )
        if any(arrays[name].dtype != np.bool_ for name in bool_names):
            raise EvaluationIntegrityError(f"{record_id}: label masks must have bool dtype")
        if arrays["y"].dtype != np.float32 or arrays["sequence_ascii"].dtype != np.uint8:
            raise EvaluationIntegrityError(f"{record_id}: label or sequence dtype mismatch")
        qc = arrays["qc_mask"]
        if np.any(qc & ~arrays["label_mask"]) or np.any(qc & ~arrays["raw_label_present"]):
            raise EvaluationIntegrityError(f"{record_id}: QC mask is not a valid subset")
        if int(qc.sum()) != int(row["qc_label_count"]):
            raise EvaluationIntegrityError(f"{record_id}: QC count differs from manifest")
        if np.any(arrays["train_access"]) or not np.all(arrays["eval_access"]):
            raise EvaluationAccessError(f"{record_id}: per-residue access flags violate test role")
        if np.any(arrays["loss_mask"]):
            raise EvaluationAccessError(f"{record_id}: test loss_mask is not empty")
        if not np.isfinite(arrays["y"][qc]).all():
            raise EvaluationIntegrityError(f"{record_id}: eligible labels are non-finite")
        if np.any(arrays["y"][~qc] != 0.0):
            raise EvaluationIntegrityError(f"{record_id}: invalid labels are not zero placeholders")
        sequence = bytes(arrays["sequence_ascii"].tolist()).decode("ascii")
        if hashlib.sha256(sequence.encode("ascii")).hexdigest() != row["sequence_sha256"]:
            raise EvaluationIntegrityError(f"{record_id}: authoritative sequence hash mismatch")

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x_np)),
            "y": torch.from_numpy(arrays["y"]),
            "qc_mask": torch.from_numpy(qc),
            "length": length,
            "record_id": record_id,
            "source": row["source"],
            "protein_id": row["protein_id"],
            "pdb_id": row["pdb_id"],
            "chain_id": row["chain_id"],
            "endpoint": self.endpoint,
            "joint_homology_group": row["joint_homology_group"],
            "sequence_sha256": row["sequence_sha256"],
            "sequence": sequence,
        }


def collate_evaluation_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not records:
        raise ValueError("Cannot collate an empty evaluation batch")
    lengths = torch.tensor([int(record["length"]) for record in records], dtype=torch.long)
    batch_size = len(records)
    max_length = int(lengths.max().item())
    x = torch.zeros((batch_size, max_length, 1182), dtype=torch.float32)
    y = torch.zeros((batch_size, max_length), dtype=torch.float32)
    qc_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    residue_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row_index, record in enumerate(records):
        length = int(record["length"])
        record_x = record["x"]
        record_y = record["y"]
        record_qc = record["qc_mask"]
        if not isinstance(record_x, Tensor) or record_x.shape != (length, 1182):
            raise EvaluationIntegrityError("Feature tensor changed before collation")
        if not isinstance(record_y, Tensor) or record_y.shape != (length,):
            raise EvaluationIntegrityError("Label tensor changed before collation")
        if not isinstance(record_qc, Tensor) or record_qc.shape != (length,):
            raise EvaluationIntegrityError("QC mask changed before collation")
        x[row_index, :length] = record_x
        y[row_index, :length] = record_y
        qc_mask[row_index, :length] = record_qc
        residue_mask[row_index, :length] = True
    metadata_fields = (
        "record_id",
        "source",
        "protein_id",
        "pdb_id",
        "chain_id",
        "endpoint",
        "joint_homology_group",
        "sequence_sha256",
        "sequence",
    )
    result: Dict[str, object] = {
        "x": x,
        "y": y,
        "qc_mask": qc_mask,
        "residue_mask": residue_mask,
        "lengths": lengths,
    }
    for field in metadata_fields:
        result[field + "s"] = [record[field] for record in records]
    return result


def make_evaluation_loader(
    dataset: EvaluationProteinDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        collate_fn=collate_evaluation_records,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
    )


def safe_pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2 or np.ptp(prediction) == 0.0 or np.ptp(target) == 0.0:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def safe_spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2 or np.ptp(prediction) == 0.0 or np.ptp(target) == 0.0:
        return float("nan")
    return float(spearmanr(prediction, target).statistic)


def nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if finite.any() else float("nan")


def record_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    qc_mask: np.ndarray,
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    qc_mask = np.asarray(qc_mask, dtype=bool)
    if prediction.ndim != 1 or prediction.shape != target.shape or prediction.shape != qc_mask.shape:
        raise ValueError("Prediction, target and QC mask must be aligned one-dimensional arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target[qc_mask]).all():
        raise ValueError("Prediction or eligible target contains non-finite values")
    eligible_prediction = prediction[qc_mask]
    eligible_target = target[qc_mask]
    row: Dict[str, object] = dict(metadata)
    row.update(
        {
            "sequence_length": int(prediction.size),
            "eligible_residues": int(qc_mask.sum()),
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std(ddof=0)),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "out_of_range_fraction_all_residues": float(
                np.mean((prediction < 0.0) | (prediction > 1.0))
            ),
        }
    )
    if eligible_prediction.size == 0:
        row.update(
            {
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "pcc": float("nan"),
                "spearman": float("nan"),
                "out_of_range_fraction": float("nan"),
            }
        )
        return row
    error = eligible_prediction - eligible_target
    row.update(
        {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "pcc": safe_pearson(eligible_prediction, eligible_target),
            "spearman": safe_spearman(eligible_prediction, eligible_target),
            "out_of_range_fraction": float(
                np.mean((eligible_prediction < 0.0) | (eligible_prediction > 1.0))
            ),
        }
    )
    return row


METRIC_NAMES = (
    "rmse",
    "mae",
    "bias",
    "pcc",
    "spearman",
    "out_of_range_fraction",
)


def summarize_predictions(
    per_record: Sequence[Mapping[str, object]],
    pooled_prediction: Sequence[np.ndarray],
    pooled_target: Sequence[np.ndarray],
) -> Dict[str, object]:
    if not per_record:
        raise ValueError("No evaluation records were supplied")
    scored_rows = [row for row in per_record if int(row["eligible_residues"]) > 0]
    macro = {
        name: nanmean([float(row[name]) for row in scored_rows]) for name in METRIC_NAMES
    }
    macro["pcc_valid_records"] = sum(
        math.isfinite(float(row["pcc"])) for row in scored_rows
    )
    macro["spearman_valid_records"] = sum(
        math.isfinite(float(row["spearman"])) for row in scored_rows
    )

    prediction_all = np.concatenate([np.asarray(x, dtype=np.float64) for x in pooled_prediction])
    target_all = np.concatenate([np.asarray(x, dtype=np.float64) for x in pooled_target])
    error_all = prediction_all - target_all
    pooled = {
        "rmse": float(np.sqrt(np.mean(error_all**2))),
        "mae": float(np.mean(np.abs(error_all))),
        "bias": float(np.mean(error_all)),
        "pcc": safe_pearson(prediction_all, target_all),
        "spearman": safe_spearman(prediction_all, target_all),
        "out_of_range_fraction": float(
            np.mean((prediction_all < 0.0) | (prediction_all > 1.0))
        ),
    }

    by_sequence: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    by_group: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in scored_rows:
        by_sequence[str(row["sequence_sha256"])].append(row)
        by_group[str(row["joint_homology_group"])].append(row)
    unique_sequence_macro = {
        name: nanmean(
            [nanmean([float(row[name]) for row in rows]) for rows in by_sequence.values()]
        )
        for name in METRIC_NAMES
    }
    homology_group_macro = {
        name: nanmean(
            [nanmean([float(row[name]) for row in rows]) for rows in by_group.values()]
        )
        for name in METRIC_NAMES
    }
    return {
        "record_count": len(per_record),
        "scored_record_count": len(scored_rows),
        "unique_sequence_count": len(by_sequence),
        "joint_homology_group_count": len(by_group),
        "complete_residue_count": int(sum(int(row["sequence_length"]) for row in per_record)),
        "eligible_residue_count": int(sum(int(row["eligible_residues"]) for row in per_record)),
        "protein_record_macro": macro,
        "pooled_residue": pooled,
        "unique_sequence_macro": unique_sequence_macro,
        "homology_group_macro": homology_group_macro,
    }


def evaluation_input_digest(records: Sequence[Mapping[str, object]]) -> str:
    """Hash IDs, sequences, masks and target values to prove identical scoring inputs."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["record_id"])):
        digest.update(str(record["record_id"]).encode("utf-8"))
        digest.update(str(record["sequence_sha256"]).encode("ascii"))
        target = np.asarray(record["target"], dtype=np.float32).copy(order="C")
        mask = np.asarray(record["qc_mask"], dtype=np.bool_).copy(order="C")
        digest.update(mask.tobytes(order="C"))
        digest.update(target.tobytes(order="C"))
    return digest.hexdigest()


def load_frozen_checkpoint(
    *,
    experiment_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    expected_window_size: int,
    expected_training_config_sha256: str,
    device: torch.device,
) -> Tuple[MDSoPBiLSTMRegressor, Dict[str, object]]:
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(f"Unknown experiment {experiment_id}")
    checkpoint_path = Path(checkpoint_path)
    observed_file_hash = sha256_file(checkpoint_path)
    if observed_file_hash != checkpoint_sha256:
        raise EvaluationIntegrityError(f"{experiment_id}: checkpoint file hash mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != TRAINING_CHECKPOINT_FORMAT:
        raise EvaluationIntegrityError(f"{experiment_id}: training checkpoint format mismatch")
    if payload.get("experiment_id") != experiment_id:
        raise EvaluationIntegrityError(f"{experiment_id}: checkpoint experiment ID mismatch")
    if payload.get("config_sha256") != expected_training_config_sha256:
        raise EvaluationIntegrityError(f"{experiment_id}: training config hash mismatch")
    if payload.get("model_version") != "v2.2" or payload.get("run_tag") != "v2_2_20261107" or payload.get("md_label_version") != "md_ired_5tauc_windowmean_v2_2":
        raise EvaluationIntegrityError(f"{experiment_id}: model/data version identity mismatch")
    if int(payload.get("epoch", -1)) != int(payload.get("best_epoch", -2)):
        raise EvaluationIntegrityError(f"{experiment_id}: best checkpoint is not stored at best epoch")

    model = MDSoPBiLSTMRegressor(window_size=int(expected_window_size))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    model.to(device)
    metadata = {
        "checkpoint_file_sha256": observed_file_hash,
        "model_state_sha256": state_dict_sha256(model.state_dict()),
        "checkpoint_epoch": int(payload["epoch"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_metric": float(payload["best_metric"]),
        "global_step": int(payload["global_step"]),
        "window_size": int(expected_window_size),
        "device": str(device),
    }
    return model, metadata


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def read_gzip_tsv(path: Path) -> List[Dict[str, str]]:
    with gzip.open(Path(path), "rt", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]

