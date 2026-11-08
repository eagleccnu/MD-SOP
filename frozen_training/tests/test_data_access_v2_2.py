import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mdsop_dataset_v2_2 import (
    AccessControlledProteinDataset,
    AccessViolationError,
    FORMAL_MANIFEST_BASENAME,
    ManifestAccessPolicy,
    collate_protein_records,
)


FIELDS = [
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
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticRelease:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dev = root / "development"
        self.labels = self.dev / "labels"
        self.evaluation = root / "evaluation"
        self.manifest = self.dev / "manifests" / FORMAL_MANIFEST_BASENAME
        for path in (self.labels, self.evaluation, self.manifest.parent, self.dev / "features"):
            path.mkdir(parents=True, exist_ok=True)

    def make_row(
        self,
        record_id: str,
        *,
        role: str = "train",
        source: str = "md_ired",
        length: int = 3,
        qc: np.ndarray = None,
        label_path: Path = None,
    ):
        if qc is None:
            qc = np.array([True] + [False] * (length - 1), dtype=bool)
        esmc_path = self.dev / "features" / f"{record_id}_esmc.npy"
        hmm_path = self.dev / "features" / f"{record_id}_hmm.npy"
        np.save(esmc_path, np.full((length, 1152), 2.0, dtype=np.float32))
        np.save(hmm_path, np.full((length, 30), 3.0, dtype=np.float32))
        if label_path is None:
            label_path = self.labels / ("md" if source == "md_ired" else "soppe") / role / f"{record_id}.npz"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        train = role == "train"
        if role in {"train", "validation"}:
            np.savez_compressed(
                label_path,
                y=np.arange(length, dtype=np.float32) / 10.0,
                label_mask=qc.copy(),
                qc_mask=qc.copy(),
                raw_label_present=qc.copy(),
                train_access=np.full(length, train, dtype=bool),
                eval_access=np.full(length, not train, dtype=bool),
                loss_mask=qc.copy() if train else np.zeros(length, dtype=bool),
                length=np.array([length], dtype=np.int64),
                sequence_ascii=np.full(length, ord("A"), dtype=np.uint8),
            )
        endpoint_prefix = "md" if source == "md_ired" else "soppe"
        return {
            "record_id": record_id,
            "source": source,
            "role_v2_1": role,
            "partition_v2_1": role,
            "role_v2_2": role,
            "partition_v2_2": role,
            "label_version": "md_ired_5tauc_windowmean_v2_2" if source == "md_ired" else "unchanged_frozen_v2_1_SOPPE_targets",
            "endpoint_kind": "primary",
            "dataset_endpoint": f"{endpoint_prefix}_{role}",
            "sequence_length": str(length),
            "esmc_path": str(esmc_path),
            "esmc_sha256": "0" * 64,
            "hmm30_path": str(hmm_path),
            "hmm30_sha256": "1" * 64,
            "logical_feature_dimension": "1182",
            "logical_feature_dtype": "float32",
            "label_path": str(label_path),
            "label_file_sha256": digest(label_path) if label_path.exists() else "2" * 64,
            "qc_label_count": str(int(qc.sum())),
            "loss_label_count": str(int(qc.sum()) if train else 0),
            "train_access": str(train),
            "eval_access": str(not train),
            "allowed_label_access": (
                "training_loss" if train else "early_stopping_and_checkpoint_selection"
            ),
        }

    def write_manifest(self, rows) -> ManifestAccessPolicy:
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return ManifestAccessPolicy(
            manifest_path=self.manifest,
            manifest_sha256=digest(self.manifest),
            development_label_root=self.labels,
            evaluation_root=self.evaluation,
        )


class TestDataAccess(unittest.TestCase):
    def test_valid_train_load_and_dynamic_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            release = SyntheticRelease(Path(directory))
            rows = [release.make_row("r1", length=3), release.make_row("r2", length=5)]
            dataset = AccessControlledProteinDataset(release.write_manifest(rows), roles=["train"])
            first, second = dataset[0], dataset[1]
            self.assertEqual(first["x"].shape, (3, 1182))
            self.assertTrue(np.allclose(first["x"][:, :1152].numpy(), 2.0))
            self.assertTrue(np.allclose(first["x"][:, 1152:].numpy(), 3.0))
            batch = collate_protein_records([first, second])
            self.assertEqual(tuple(batch["x"].shape), (2, 5, 1182))
            self.assertTrue(batch["padding_mask"][0, 3:].all())
            self.assertFalse(batch["residue_mask"][0, 3:].any())
            self.assertFalse(batch["loss_mask"][0, 3:].any())

    def test_validation_uses_eval_mask_not_training_loss_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            release = SyntheticRelease(Path(directory))
            row = release.make_row("v1", role="validation")
            dataset = AccessControlledProteinDataset(release.write_manifest([row]), roles=["validation"])
            item = dataset[0]
            self.assertEqual(int(item["loss_mask"].sum()), 0)
            self.assertEqual(int(item["eval_mask"].sum()), 1)

    def test_evaluation_label_path_fails_before_file_open(self):
        with tempfile.TemporaryDirectory() as directory:
            release = SyntheticRelease(Path(directory))
            row = release.make_row("safe")
            forbidden = release.evaluation / "labels" / "nonexistent_secret.npz"
            row["label_path"] = str(forbidden)
            row["label_file_sha256"] = "3" * 64
            policy = release.write_manifest([row])
            with self.assertRaisesRegex(AccessViolationError, "evaluation labels are forbidden"):
                AccessControlledProteinDataset(policy, roles=["train"])

    def test_test_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            release = SyntheticRelease(Path(directory))
            row = release.make_row("r1")
            row.update(
                role_v2_1="test",
                partition_v2_1="test",
                dataset_endpoint="md_test",
                train_access="False",
                eval_access="True",
                allowed_label_access="test_metrics",
            )
            with self.assertRaisesRegex(AccessViolationError, "forbidden trainer role"):
                AccessControlledProteinDataset(release.write_manifest([row]), roles=["train"])

    def test_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            release = SyntheticRelease(Path(directory))
            policy = release.write_manifest([release.make_row("r1")])
            bad_policy = ManifestAccessPolicy(
                policy.manifest_path,
                "f" * 64,
                policy.development_label_root,
                policy.evaluation_root,
            )
            with self.assertRaisesRegex(AccessViolationError, "manifest hash mismatch"):
                AccessControlledProteinDataset(bad_policy, roles=["train"])

    def test_zero_supervision_record_is_tracked_but_not_batched(self):
        with tempfile.TemporaryDirectory() as directory:
            release = SyntheticRelease(Path(directory))
            zero = np.zeros(3, dtype=bool)
            rows = [release.make_row("zero", qc=zero), release.make_row("positive")]
            dataset = AccessControlledProteinDataset(release.write_manifest(rows), roles=["train"])
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.excluded_zero_supervision_records, ("zero",))


if __name__ == "__main__":
    unittest.main()
