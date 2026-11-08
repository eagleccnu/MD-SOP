#!/usr/bin/env python3
"""Prepare isolated MD-SOP v2.1 development and evaluation datasets.

The command is intentionally phased:

* ``prepare`` verifies frozen inputs and prepares only sequence-level work lists.
* ``extract-esmc`` generates the missing frozen ESM-C 600M arrays.
* ``extract-hmm`` generates the missing legacy HMM30 arrays with HH-suite.
* ``build`` applies frozen source-QC rules, packages labels/masks, and publishes
  component feature indices.  Test labels are written only below evaluation/.

No model is trained and no predictive score is computed by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "02_prepare_features_labels_v2_1_20261007"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
AUTHORITY_AA = STANDARD_AA | {"X"}
ESMC_DIM = 1152
HMM_DIM = 30
LOGICAL_DIM = 1182
ESMC_REVISION = "fcc9d36f5fff97d9bb69a36e3252de61cc78c968"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        + "\n",
    )


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False, na_rep="", lineterminator="\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fasta_text(identifier: str, sequence: str) -> str:
    return f">{identifier}\n" + "\n".join(
        sequence[index : index + 80] for index in range(0, len(sequence), 80)
    ) + "\n"


def load_config(path: Path) -> dict[str, Any]:
    # The file is JSON syntax and therefore valid YAML while avoiding a PyYAML dependency.
    return json.loads(path.read_text(encoding="utf-8"))


def work_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    data_root = Path(config["paths"]["output_data_root"])
    return {
        "data_root": data_root,
        "manifests": data_root / "manifests",
        "audits": data_root / "audits",
        "development": data_root / "development",
        "evaluation": data_root / "evaluation",
        "work": data_root / "feature_work",
        "esmc_generated": data_root / "evaluation" / "components" / "esmc600_generated",
        "hmm_generated": data_root / "evaluation" / "components" / "hmm30_generated",
    }


def verify_frozen_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    p = config["paths"]
    d1 = Path(p["d1_release"])
    exp = Path(p["experimental10_root"])
    checks = [
        ("protocol", Path(p["protocol"]), config["frozen_sha256"]["protocol"]),
        (
            "joint_split_manifest",
            Path(p["joint_split_manifest"]),
            config["frozen_sha256"]["joint_split_manifest"],
        ),
        ("d1_proteins", d1 / "proteins.parquet", config["frozen_sha256"]["d1_proteins"]),
        (
            "d1_residue_targets",
            d1 / "residue_targets.parquet",
            config["frozen_sha256"]["d1_residue_targets"],
        ),
        ("soppe_labels", Path(p["soppe_labels"]), config["frozen_sha256"]["soppe_labels"]),
        ("md_labels", Path(p["md_labels"]), config["frozen_sha256"]["md_labels"]),
        ("md_selection", Path(p["md_selection"]), config["frozen_sha256"]["md_selection"]),
        ("atlas_catalog", Path(p["atlas_catalog"]), config["frozen_sha256"]["atlas_catalog"]),
        ("soppe_knn", Path(p["soppe_knn"]), config["frozen_sha256"]["soppe_knn"]),
        (
            "frozen_confidence_rules",
            Path(p["frozen_confidence_rules"]),
            config["frozen_sha256"]["frozen_confidence_rules"],
        ),
        (
            "experimental10_benchmark_manifest",
            exp / "benchmark_manifest.tsv",
            config["frozen_sha256"]["experimental10_benchmark_manifest"],
        ),
        (
            "experimental10_protein_manifest",
            exp / "protein_manifest.tsv",
            config["frozen_sha256"]["experimental10_protein_manifest"],
        ),
        (
            "experimental10_residue_alignment",
            exp / "residue_alignment.tsv",
            config["frozen_sha256"]["experimental10_residue_alignment"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, expected in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        row = {
            "name": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passed": observed == expected,
        }
        rows.append(row)
        if observed != expected:
            raise RuntimeError(f"frozen input hash mismatch: {name}")
    return rows


def load_joint(config: Mapping[str, Any]) -> pd.DataFrame:
    joint = pd.read_csv(
        config["paths"]["joint_split_manifest"], sep="\t", keep_default_na=False
    )
    if joint["record_id"].duplicated().any():
        raise RuntimeError("duplicate record_id in joint split")
    for row in joint.itertuples(index=False):
        sequence = str(row.sequence)
        if len(sequence) != int(row.sequence_length):
            raise RuntimeError(f"sequence length mismatch: {row.record_id}")
        if sequence_sha256(sequence) != str(row.sequence_sha256):
            raise RuntimeError(f"sequence hash mismatch: {row.record_id}")
        # One frozen S-OPPE test sequence contains an explicit X.  It remains in
        # the encoder input, while the frozen source-QC rule excludes that
        # residue from supervision.  No other noncanonical symbol is accepted.
        if set(sequence) - AUTHORITY_AA:
            raise RuntimeError(f"unsupported sequence symbol in authority: {row.record_id}")
    return joint


def selected_records(joint: pd.DataFrame) -> pd.DataFrame:
    weak = joint[
        joint["source"].isin(["md_ired", "soppe_legacy"])
        & joint["role_v2_1"].isin(["train", "validation", "test"])
    ].copy()
    exp = joint[joint["role_v2_1"] == "reference_experimental10"].copy()
    selected = pd.concat([weak, exp], ignore_index=True)

    def endpoint(row: pd.Series) -> str:
        if row["source"] == "historical_test":
            return "experimental10"
        prefix = "md" if row["source"] == "md_ired" else "soppe"
        return f"{prefix}_{row['role_v2_1']}"

    selected["dataset_endpoint"] = selected.apply(endpoint, axis=1)
    return selected


def expected_record_counts(config: Mapping[str, Any]) -> dict[str, int]:
    expected = config["expected"]
    return {
        "md_train": int(expected["md_train_records"]),
        "md_validation": int(expected["md_validation_records"]),
        "md_test": int(expected["md_test_records"]),
        "soppe_train": int(expected["soppe_train_records"]),
        "soppe_validation": int(expected["soppe_validation_records"]),
        "soppe_test": int(expected["soppe_test_records"]),
        "experimental10": int(expected["experimental10_records"]),
    }


def read_esmc_candidates(config: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for origin, manifest_path in (
        ("soppe_esmc600_frozen_cache", config["paths"]["soppe_esmc_manifest"]),
        ("md_esmc600_frozen_cache", config["paths"]["md_esmc_manifest"]),
    ):
        frame = pd.read_csv(manifest_path, sep="\t", keep_default_na=False)
        for row in frame.itertuples(index=False):
            candidates[str(row.sequence_sha256)].append(
                {
                    "origin": origin,
                    "path": str(row.feature_file),
                    "sha256": str(row.feature_sha256),
                    "length": str(row.sequence_length),
                }
            )
    return candidates


def soppe_raw_map(config: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(config["paths"]["soppe_raw1310_release"])
    frame = pd.read_csv(root / "dataset_manifest.tsv", sep="\t", keep_default_na=False)
    result: dict[str, Path] = {}
    for row in frame.itertuples(index=False):
        path = Path(str(row.record_path))
        if not path.is_absolute():
            path = root / path
        result[str(row.protein_id).lower()] = path
    return result


def md_hmm_map(config: Mapping[str, Any]) -> dict[str, list[Path]]:
    root = Path(config["paths"]["md_raw1310_release"])
    frame = pd.read_csv(root / "dataset_manifest.tsv", sep="\t", keep_default_na=False)
    result: dict[str, list[Path]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        result[str(row.sequence_sha256)].append(Path(str(row.hmm30_path)))
    return result


def experimental_manifest(config: Mapping[str, Any]) -> pd.DataFrame:
    return pd.read_csv(
        Path(config["paths"]["experimental10_root"]) / "benchmark_manifest.tsv",
        sep="\t",
        keep_default_na=False,
    )


def hmm_coverage_hashes(config: Mapping[str, Any], joint: pd.DataFrame) -> set[str]:
    hashes = set(md_hmm_map(config))
    hashes |= set(
        joint.loc[joint["source"] == "soppe_legacy", "sequence_sha256"].astype(str)
    )
    hashes |= set(
        joint.loc[joint["source"] == "historical_test", "sequence_sha256"].astype(str)
    )
    return hashes


def prepare(config: Mapping[str, Any], config_path: Path) -> int:
    paths = work_paths(config)
    if (paths["audits"] / "step3_acceptance.json").exists():
        raise RuntimeError("immutable accepted Step-3 release already exists")
    for key in ("manifests", "audits", "development", "evaluation", "work"):
        paths[key].mkdir(parents=True, exist_ok=True)

    frozen = verify_frozen_inputs(config)
    joint = load_joint(config)
    records = selected_records(joint)
    observed_counts = records.groupby("dataset_endpoint").size().to_dict()
    expected_counts = expected_record_counts(config)
    if observed_counts != expected_counts:
        raise RuntimeError(f"endpoint record count mismatch: {observed_counts}")

    esmc = read_esmc_candidates(config)
    hmm_hashes = hmm_coverage_hashes(config, joint)
    unique = (
        records.groupby("sequence_sha256", as_index=False)
        .agg(
            sequence=("sequence", "first"),
            sequence_length=("sequence_length", "first"),
            record_count=("record_id", "size"),
            record_ids=("record_id", lambda values: ";".join(sorted(map(str, values)))),
            endpoints=("dataset_endpoint", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .sort_values("sequence_sha256")
        .reset_index(drop=True)
    )
    unique["existing_esmc_candidate_count"] = unique["sequence_sha256"].map(
        lambda value: len(esmc.get(str(value), []))
    )
    unique["existing_hmm_candidate"] = unique["sequence_sha256"].isin(hmm_hashes)
    atomic_tsv(paths["work"] / "required_unique_sequences.tsv", unique)

    missing_esmc = unique[unique["existing_esmc_candidate_count"] == 0].copy()
    missing_hmm = unique[~unique["existing_hmm_candidate"]].copy()
    expected = config["expected"]
    if len(missing_esmc) != int(expected["missing_esmc_unique_sequences"]):
        raise RuntimeError(f"missing ESM-C count changed: {len(missing_esmc)}")
    if len(missing_hmm) != int(expected["missing_hmm_unique_sequences"]):
        raise RuntimeError(f"missing HMM count changed: {len(missing_hmm)}")

    esmc_fasta = paths["work"] / "missing_esmc600_132.fasta"
    atomic_text(
        esmc_fasta,
        "".join(
            fasta_text(str(row.sequence_sha256), str(row.sequence))
            for row in missing_esmc.itertuples(index=False)
        ),
    )
    atomic_tsv(paths["work"] / "missing_esmc600_manifest.tsv", missing_esmc)

    hmm_fasta_dir = paths["hmm_generated"] / "fasta_by_sequence"
    hmm_fasta_dir.mkdir(parents=True, exist_ok=True)
    hmm_rows: list[dict[str, Any]] = []
    for row in missing_hmm.itertuples(index=False):
        fasta_path = hmm_fasta_dir / f"{row.sequence_sha256}.fasta"
        atomic_text(fasta_path, fasta_text(str(row.sequence_sha256), str(row.sequence)))
        hmm_rows.append(
            {
                "sequence_sha256": row.sequence_sha256,
                "sequence": row.sequence,
                "sequence_length": int(row.sequence_length),
                "record_count": int(row.record_count),
                "record_ids": row.record_ids,
                "endpoints": row.endpoints,
                "fasta_path": str(fasta_path),
                "fasta_sha256": sha256_file(fasta_path),
            }
        )
    atomic_tsv(paths["work"] / "missing_hmm30_manifest.tsv", pd.DataFrame(hmm_rows))
    atomic_tsv(paths["audits"] / "step3_frozen_input_audit.tsv", pd.DataFrame(frozen))
    inventory = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at": now_iso(),
        "status": "PASS_PREPARED_EXTRACTION_WORKLISTS",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "endpoint_record_counts": observed_counts,
        "selected_records": len(records),
        "required_unique_sequences": len(unique),
        "missing_esmc_unique_sequences": len(missing_esmc),
        "missing_hmm_unique_sequences": len(missing_hmm),
        "missing_esmc_total_residues": int(pd.to_numeric(missing_esmc["sequence_length"]).sum()),
        "missing_hmm_total_residues": int(pd.to_numeric(missing_hmm["sequence_length"]).sum()),
        "test_label_arrays_opened": False,
        "test_label_values_used": False,
        "next_commands": ["extract-esmc", "extract-hmm", "build", "independent audit"],
    }
    atomic_json(paths["audits"] / "step3_prepare_inventory.json", inventory)
    print(json.dumps(inventory, indent=2))
    return 0


def run_esmc(config: Mapping[str, Any]) -> int:
    paths = work_paths(config)
    source_fasta = paths["work"] / "missing_esmc600_132.fasta"
    if not source_fasta.is_file():
        raise FileNotFoundError("run prepare first")
    output = paths["esmc_generated"] / "features"
    logs = paths["esmc_generated"] / "logs"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        config["paths"]["esmc_extract"],
        "--input",
        str(source_fasta),
        "--output-dir",
        str(output),
        "--model",
        "600m",
        "--device",
        str(config["esmc"]["device"]),
        "--batch-size",
        str(config["esmc"]["batch_size"]),
        "--inference-dtype",
        str(config["esmc"]["inference_dtype"]),
        "--output-dtype",
        str(config["esmc"]["cache_dtype"]),
        "--local-files-only",
        "--resume",
    ]
    started = time.monotonic()
    with (logs / "extraction.stdout.log").open("ab") as stdout, (
        logs / "extraction.stderr.log"
    ).open("ab") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    run = {
        "script_version": SCRIPT_VERSION,
        "completed_at": now_iso(),
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "input_fasta": str(source_fasta),
        "input_fasta_sha256": sha256_file(source_fasta),
        "output_directory": str(output),
    }
    atomic_json(logs / "extraction_run.json", run)
    print(json.dumps(run, indent=2))
    return int(result.returncode)


def parse_hhm_legacy(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    markers = [index for index, line in enumerate(lines) if line.split() == ["#"]]
    if not markers:
        raise RuntimeError(f"HHM marker not found: {path}")
    values: list[list[str]] = []
    for index, line in enumerate(lines):
        if index > markers[-1] + 4:
            tokens = line.split()[0:22]
            if len(tokens) > 1:
                values.append(tokens)
    if len(values) % 2:
        raise RuntimeError(f"odd HHM data-line count: {path}")
    output: list[list[float]] = []
    for index in range(0, len(values), 2):
        fields = (values[index] + values[index + 1])[2:]
        if len(fields) != 30:
            raise RuntimeError(f"HHM residue {index // 2 + 1} has {len(fields)} fields")
        row = []
        for token in fields:
            if token == "0":
                row.append(1.0)
            elif token == "*":
                row.append(0.0)
            else:
                row.append(round(2 ** (int(token) / -1000), 2))
        output.append(row)
    return np.asarray(output, dtype=np.float32)


def hmm_status_valid(status_path: Path, row: Mapping[str, Any]) -> bool:
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] != "passed" or status["sequence_sha256"] != row["sequence_sha256"]:
            return False
        array_path = Path(status["hmm30_path"])
        if not array_path.is_file() or sha256_file(array_path) != status["hmm30_sha256"]:
            return False
        array = np.load(array_path, allow_pickle=False)
        return (
            array.shape == (int(row["sequence_length"]), HMM_DIM)
            and array.dtype == np.float32
            and np.isfinite(array).all()
        )
    except Exception:
        return False


def execute_hmm_one(
    row: Mapping[str, Any], config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    sequence_hash = str(row["sequence_sha256"])
    length = int(row["sequence_length"])
    root = paths["hmm_generated"]
    for name in ("a3m_by_sequence", "hhm_by_sequence", "hmm30_by_sequence", "status", "attempts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    attempt = root / "attempts" / sequence_hash / (
        datetime.now().strftime("%Y%m%dT%H%M%S%f") + f"_{os.getpid()}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    local_fasta = attempt / f"{sequence_hash}.fasta"
    shutil.copy2(Path(str(row["fasta_path"])), local_fasta)
    a3m_tmp = attempt / f"{sequence_hash}.a3m.tmp"
    hhm_tmp = attempt / f"{sequence_hash}.hhm.tmp"
    a3m_final = root / "a3m_by_sequence" / f"{sequence_hash}.a3m"
    hhm_final = root / "hhm_by_sequence" / f"{sequence_hash}.hhm"
    hmm_final = root / "hmm30_by_sequence" / f"{sequence_hash}.npy"
    hhblits = [
        config["paths"]["hhblits"],
        "-i",
        str(local_fasta),
        "-d",
        config["paths"]["hmm_database_prefix"],
        "-n",
        str(config["hmm"]["iterations"]),
        "-mact",
        str(config["hmm"]["mact"]),
        "-cpu",
        str(config["hmm"]["cpu_per_job"]),
        "-oa3m",
        str(a3m_tmp),
    ]
    hhmake = [config["paths"]["hhmake"], "-i", str(a3m_tmp), "-o", str(hhm_tmp)]
    status: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "sequence_sha256": sequence_hash,
        "sequence_length": length,
        "started_at": now_iso(),
        "hhblits_command": hhblits,
        "hhmake_command": hhmake,
        "test_label_files_opened": False,
    }
    try:
        start = time.monotonic()
        with (attempt / "hhblits.stdout.log").open("wb") as out, (
            attempt / "hhblits.stderr.log"
        ).open("wb") as err:
            result = subprocess.run(
                hhblits, cwd="/home/amax/hh-suite/build", stdout=out, stderr=err, check=False
            )
        status["hhblits_seconds"] = round(time.monotonic() - start, 3)
        if result.returncode:
            raise RuntimeError(f"hhblits return code {result.returncode}")
        start = time.monotonic()
        with (attempt / "hhmake.stdout.log").open("wb") as out, (
            attempt / "hhmake.stderr.log"
        ).open("wb") as err:
            result = subprocess.run(hhmake, stdout=out, stderr=err, check=False)
        status["hhmake_seconds"] = round(time.monotonic() - start, 3)
        if result.returncode:
            raise RuntimeError(f"hhmake return code {result.returncode}")
        array = parse_hhm_legacy(hhm_tmp)
        if array.shape != (length, HMM_DIM):
            raise RuntimeError(f"HMM shape {array.shape} != {(length, HMM_DIM)}")
        if not np.isfinite(array).all() or float(array.min()) < 0 or float(array.max()) > 1:
            raise RuntimeError("invalid HMM values")
        if not np.allclose(array * 100, np.rint(array * 100), atol=1e-5, rtol=0):
            raise RuntimeError("HMM values do not preserve legacy two-decimal semantics")
        os.replace(a3m_tmp, a3m_final)
        os.replace(hhm_tmp, hhm_final)
        atomic_npy(hmm_final, array)
        status.update(
            {
                "status": "passed",
                "completed_at": now_iso(),
                "a3m_path": str(a3m_final),
                "a3m_sha256": sha256_file(a3m_final),
                "hhm_path": str(hhm_final),
                "hhm_sha256": sha256_file(hhm_final),
                "hmm30_path": str(hmm_final),
                "hmm30_sha256": sha256_file(hmm_final),
                "error": "",
            }
        )
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "completed_at": now_iso(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
    atomic_json(root / "status" / f"{sequence_hash}.json", status)
    return status


def run_hmm(config: Mapping[str, Any], workers: int | None = None) -> int:
    paths = work_paths(config)
    manifest = paths["work"] / "missing_hmm30_manifest.tsv"
    if not manifest.is_file():
        raise FileNotFoundError("run prepare first")
    rows = pd.read_csv(manifest, sep="\t", keep_default_na=False).to_dict("records")
    status_dir = paths["hmm_generated"] / "status"
    pending = [
        row
        for row in rows
        if not hmm_status_valid(status_dir / f"{row['sequence_sha256']}.json", row)
    ]
    workers = int(workers or config["hmm"]["workers"])
    print(
        json.dumps(
            {
                "total": len(rows),
                "already_valid": len(rows) - len(pending),
                "pending": len(pending),
                "workers": workers,
                "cpu_per_job": config["hmm"]["cpu_per_job"],
            },
            indent=2,
        ),
        flush=True,
    )
    failed = 0
    durations: list[float] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute_hmm_one, row, config, paths): row for row in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            if status["status"] == "passed":
                durations.append(
                    float(status["hhblits_seconds"]) + float(status["hhmake_seconds"])
                )
            else:
                failed += 1
                print(f"[FAILED] {status['sequence_sha256']} {status['error']}", flush=True)
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                mean = float(np.mean(durations)) if durations else math.nan
                print(
                    f"[HMM] {completed}/{len(pending)} failed={failed} "
                    f"elapsed_s={time.monotonic() - started:.1f} mean_job_s={mean:.2f}",
                    flush=True,
                )
    summary = {
        "script_version": SCRIPT_VERSION,
        "completed_at": now_iso(),
        "total_sequences": len(rows),
        "newly_attempted": len(pending),
        "failed": failed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "passed": failed == 0
        and all(
            hmm_status_valid(status_dir / f"{row['sequence_sha256']}.json", row)
            for row in rows
        ),
        "test_label_files_opened": False,
    }
    atomic_json(paths["hmm_generated"] / "hmm_extraction_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


def import_module(path: str, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def qc_roles(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["joint_training_role"] = result["role_v2_1"]
    result["joint_use"] = "evaluation_only"
    exact_count = result.groupby("sequence_sha256")["record_id"].transform("size")
    group_count = result.groupby("joint_homology_group")["record_id"].transform("size")
    result["exact_sequence_active_multiplicity"] = exact_count
    result["exact_sequence_balance_factor"] = 1.0 / exact_count
    result["homology_group_active_multiplicity"] = group_count
    result["homology_group_balance_factor"] = 1.0 / group_count
    return result


def build_frozen_test_qc(
    config: Mapping[str, Any], joint: pd.DataFrame, paths: Mapping[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    g2 = import_module(config["paths"]["stage_g2_script"], "mdsop_stage_g2_frozen")
    g3 = import_module(config["paths"]["stage_g3_script"], "mdsop_stage_g3_frozen")
    rules = json.loads(Path(config["paths"]["frozen_confidence_rules"]).read_text())
    md_roles = qc_roles(
        joint[(joint["source"] == "md_ired") & (joint["role_v2_1"] == "test")]
    )
    soppe_roles = qc_roles(
        joint[(joint["source"] == "soppe_legacy") & (joint["role_v2_1"] == "test")]
    )

    selection = pd.read_csv(config["paths"]["md_selection"], sep="\t", keep_default_na=False)
    catalog = pd.read_csv(config["paths"]["atlas_catalog"], sep="\t", keep_default_na=False)
    archive_residue, archive_protein, archive_audit = g2.archive_features(
        Path("/data/jiangbin/iRED"),
        md_roles,
        selection,
        catalog,
        "data/raw/atlas/atlas_live_20260730/analysis/{protein_id}/analysis.zip",
    )
    md_features = g2.build_md_features(
        Path(config["paths"]["md_labels"]),
        md_roles,
        selection,
        catalog,
        archive_residue,
        archive_protein,
    )
    md_confidence = g3.apply_md_rules(md_features, rules)

    soppe_features, missing_ids = g2.build_soppe_features(
        Path(config["paths"]["soppe_labels"]),
        Path(config["paths"]["soppe_knn"]),
        Path(config["paths"]["soppe_stage2_quarantine"]),
        soppe_roles,
    )
    soppe_confidence = g3.apply_soppe_rules(soppe_features, rules)
    if missing_ids:
        raise RuntimeError(f"S-OPPE test lacks frozen kNN QC: {missing_ids}")

    qc_dir = paths["evaluation"] / "qc"
    atomic_parquet(qc_dir / "md_test_frozen_confidence_by_residue.parquet", md_confidence)
    atomic_parquet(
        qc_dir / "soppe_test_frozen_confidence_by_residue.parquet", soppe_confidence
    )
    atomic_tsv(qc_dir / "md_test_archive_member_audit.tsv", archive_audit)

    # Most S-test records were formerly development records.  Their freshly
    # applied frozen rules must exactly reproduce D1 labels and eligibility.
    d1 = pd.read_parquet(Path(config["paths"]["d1_release"]) / "residue_targets.parquet")
    d1_s = d1[d1["record_id"].isin(set(soppe_confidence["record_id"]))][
        ["record_id", "sequence_position", "label_s2", "label_qc_eligible"]
    ].copy()
    current = soppe_confidence[
        ["record_id", "sequence_position", "label_s2", "confidence_grade"]
    ].copy()
    overlap = current.merge(
        d1_s,
        on=["record_id", "sequence_position"],
        how="inner",
        suffixes=("_new", "_d1"),
        validate="one_to_one",
    )
    overlap_label_equal = bool(
        np.allclose(overlap["label_s2_new"], overlap["label_s2_d1"], equal_nan=True)
    )
    overlap_mask_equal = bool(
        np.array_equal(
            overlap["confidence_grade"].isin(["A", "B", "C"]).to_numpy(),
            overlap["label_qc_eligible"].astype(bool).to_numpy(),
        )
    )
    audit = {
        "frozen_rule_sha256": sha256_file(Path(config["paths"]["frozen_confidence_rules"])),
        "frozen_rule_fit_scope": rules.get("fit_scope"),
        "md_test_records": int(md_confidence["record_id"].nunique()),
        "md_test_label_rows": len(md_confidence),
        "md_test_grade_counts": md_confidence["confidence_grade"].value_counts().to_dict(),
        "soppe_test_records": int(soppe_confidence["record_id"].nunique()),
        "soppe_test_label_rows": len(soppe_confidence),
        "soppe_test_grade_counts": soppe_confidence["confidence_grade"].value_counts().to_dict(),
        "soppe_test_knn_missing_ids": missing_ids,
        "soppe_old_development_overlap_rows": len(overlap),
        "soppe_old_development_overlap_records": int(overlap["record_id"].nunique()),
        "soppe_overlap_labels_equal": overlap_label_equal,
        "soppe_overlap_qc_masks_equal": overlap_mask_equal,
        "qc_rules_refit_on_test": False,
        "test_scores_computed": False,
    }
    if not overlap_label_equal or not overlap_mask_equal:
        raise RuntimeError("frozen S-OPPE test QC does not reproduce overlapping D1 rows")
    atomic_json(paths["audits"] / "step3_frozen_test_qc_audit.json", audit)
    return md_confidence, soppe_confidence, audit


def generated_esmc_candidates(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, list[dict[str, str]]]:
    feature_dir = paths["esmc_generated"] / "features"
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing generated ESM-C manifest; run extract-esmc")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config["expected"]
    checks = {
        "model_repo": config["esmc"]["model"],
        "model_revision": config["esmc"]["model_revision"],
        "feature_dim": int(expected["esmc_dimension"]),
        "output_dtype": config["esmc"]["cache_dtype"],
        "inference_dtype": config["esmc"]["inference_dtype"],
        "sequence_count": int(expected["missing_esmc_unique_sequences"]),
    }
    for key, expected_value in checks.items():
        if manifest.get(key) != expected_value:
            raise RuntimeError(
                f"generated ESM-C manifest {key}={manifest.get(key)!r} != {expected_value!r}"
            )
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest["files"]:
        sequence_hash = str(row["protein_id"])
        path = feature_dir / str(row["feature_file"])
        result[sequence_hash].append(
            {
                "origin": "md_sop_step3_generated_esmc600",
                "path": str(path),
                "sha256": sha256_file(path),
                "length": str(row["sequence_length"]),
            }
        )
    return result


def load_hmm_candidate(kind: str, path: Path) -> np.ndarray:
    if kind in ("npy_hmm30", "generated_hmm30"):
        array = np.load(path, allow_pickle=False)
    elif kind == "soppe_raw1310_npz":
        with np.load(path, allow_pickle=False) as package:
            array = package["features"][:, 1280:1310]
    elif kind == "experimental10_raw1310_npy":
        array = np.load(path, allow_pickle=False)[:, 1280:1310]
    else:
        raise ValueError(kind)
    return np.asarray(array, dtype=np.float32)


def hmm_candidates_for_sequence(
    sequence_hash: str,
    config: Mapping[str, Any],
    joint: pd.DataFrame,
    s_raw: Mapping[str, Path],
    md_hmm: Mapping[str, list[Path]],
    exp_by_id: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> list[tuple[str, Path, str]]:
    candidates: list[tuple[str, Path, str]] = []
    for path in md_hmm.get(sequence_hash, []):
        candidates.append(("npy_hmm30", path, "md_d1_frozen_hmm30"))
    for row in joint[
        (joint["source"] == "soppe_legacy") & (joint["sequence_sha256"] == sequence_hash)
    ].itertuples(index=False):
        path = s_raw[str(row.protein_id).lower()]
        candidates.append(("soppe_raw1310_npz", path, "soppe_frozen_raw1310_tail30"))
    for row in joint[
        (joint["source"] == "historical_test") & (joint["sequence_sha256"] == sequence_hash)
    ].itertuples(index=False):
        entry = exp_by_id[str(row.protein_id)]
        candidates.append(
            (
                "experimental10_raw1310_npy",
                Path(str(entry["raw1310_path"])),
                "experimental10_frozen_raw1310_tail30",
            )
        )
    generated = paths["hmm_generated"] / "hmm30_by_sequence" / f"{sequence_hash}.npy"
    if generated.is_file():
        candidates.append(("generated_hmm30", generated, "md_sop_step3_generated_hmm30"))
    deduplicated: list[tuple[str, Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, path, origin in candidates:
        key = (kind, str(path))
        if key not in seen:
            seen.add(key)
            deduplicated.append((kind, path, origin))
    return deduplicated


def label_package_path(paths: Mapping[str, Path], endpoint: str, record_id: str) -> Path:
    safe = record_id.replace("/", "_").replace("\\", "_")
    if endpoint.startswith("md_") or endpoint.startswith("soppe_"):
        source = "md" if endpoint.startswith("md_") else "soppe"
        role = endpoint.split("_", 1)[1]
        if role in ("train", "validation"):
            return paths["development"] / "labels" / source / role / f"{safe}.npz"
    return paths["evaluation"] / endpoint / "labels" / f"{safe}.npz"


def feature_group_dir(paths: Mapping[str, Path], endpoint: str) -> Path:
    if endpoint.endswith("_train"):
        return paths["development"] / "components" / "hmm30" / "train"
    if endpoint.endswith("_validation"):
        return paths["development"] / "components" / "hmm30" / "validation"
    return paths["evaluation"] / endpoint / "components" / "hmm30"


def feature_manifest_path(paths: Mapping[str, Path], endpoint: str) -> Path:
    if endpoint.endswith("_train") or endpoint.endswith("_validation"):
        return paths["development"] / "feature_indices" / f"{endpoint}.tsv"
    return paths["evaluation"] / endpoint / "feature_index.tsv"


def record_manifest_path(paths: Mapping[str, Path], endpoint: str) -> Path:
    if endpoint.endswith("_train") or endpoint.endswith("_validation"):
        return paths["development"] / "manifests" / f"{endpoint}.tsv"
    return paths["evaluation"] / endpoint / "manifest.tsv"


def package_label_arrays(
    row: Any,
    endpoint: str,
    label_frame: pd.DataFrame | None,
    experimental_entry: Mapping[str, Any] | None,
    paths: Mapping[str, Path],
) -> tuple[Path, dict[str, Any]]:
    length = int(row.sequence_length)
    sequence = str(row.sequence)
    y = np.zeros(length, dtype=np.float32)
    qc_mask = np.zeros(length, dtype=bool)
    raw_present = np.zeros(length, dtype=bool)
    label_source_path = ""
    label_source_sha = ""
    qc_rule = ""

    if endpoint == "experimental10":
        assert experimental_entry is not None
        label_path = Path(str(experimental_entry["label_path"]))
        if sha256_file(label_path) != str(experimental_entry["label_sha256"]):
            raise RuntimeError(f"Experimental10 label hash mismatch: {row.protein_id}")
        with np.load(label_path, allow_pickle=False) as package:
            labels = np.asarray(package["labels"], dtype=np.float64)
            mask = np.asarray(package["label_mask"], dtype=bool)
            amino = "".join(np.asarray(package["aa"]).astype(str).tolist())
        if labels.shape != (length,) or mask.shape != (length,) or amino != sequence:
            raise RuntimeError(f"Experimental10 label alignment mismatch: {row.protein_id}")
        raw_present = np.isfinite(labels)
        qc_mask = mask & raw_present
        y[qc_mask] = labels[qc_mask].astype(np.float32)
        label_source_path = str(label_path)
        label_source_sha = str(experimental_entry["label_sha256"])
        qc_rule = "frozen_historical_experimental10_mask"
    else:
        assert label_frame is not None
        current = label_frame.sort_values("sequence_position")
        positions = pd.to_numeric(current["sequence_position"]).to_numpy(dtype=int)
        if len(positions) and (positions.min() < 1 or positions.max() > length):
            raise RuntimeError(f"label position outside sequence: {row.record_id}")
        if len(np.unique(positions)) != len(positions):
            raise RuntimeError(f"duplicate label position: {row.record_id}")
        amino = current["amino_acid"].astype(str).to_numpy()
        for position, residue in zip(positions, amino):
            if residue != sequence[position - 1]:
                raise RuntimeError(
                    f"label residue mismatch: {row.record_id} position {position}"
                )
        values = pd.to_numeric(current["label_s2"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        raw_present[positions[finite] - 1] = True
        if "label_qc_eligible" in current.columns:
            eligible = current["label_qc_eligible"].astype(bool).to_numpy() & finite
            qc_rule = "d1_frozen_label_qc_eligible"
        else:
            eligible = current["confidence_grade"].isin(["A", "B", "C"]).to_numpy() & finite
            qc_rule = "stage_g3_frozen_source_confidence_ABC"
        qc_mask[positions[eligible] - 1] = True
        y[positions[eligible] - 1] = values[eligible].astype(np.float32)
        label_source_path = str(current.attrs.get("source_path", ""))
        label_source_sha = str(current.attrs.get("source_sha256", ""))

    train_access = np.full(length, str(row.role_v2_1) == "train", dtype=bool)
    eval_access = np.full(
        length,
        str(row.role_v2_1) in ("validation", "test", "reference_experimental10"),
        dtype=bool,
    )
    loss_mask = qc_mask & train_access
    package_path = label_package_path(paths, endpoint, str(row.record_id))
    atomic_npz(
        package_path,
        y=y,
        label_mask=qc_mask,
        qc_mask=qc_mask,
        raw_label_present=raw_present,
        train_access=train_access,
        eval_access=eval_access,
        loss_mask=loss_mask,
        length=np.asarray([length], dtype=np.int64),
        sequence_ascii=np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).copy(),
    )
    return package_path, {
        "label_file_sha256": sha256_file(package_path),
        "raw_label_count": int(raw_present.sum()),
        "qc_label_count": int(qc_mask.sum()),
        "loss_label_count": int(loss_mask.sum()),
        "train_access": bool(train_access[0]) if length else False,
        "eval_access": bool(eval_access[0]) if length else False,
        "qc_rule": qc_rule,
        "upstream_label_path": label_source_path,
        "upstream_label_sha256": label_source_sha,
    }


def build(config: Mapping[str, Any], config_path: Path) -> int:
    paths = work_paths(config)
    if (paths["audits"] / "step3_acceptance.json").exists():
        raise RuntimeError("immutable accepted Step-3 release already exists")
    verify_frozen_inputs(config)
    joint = load_joint(config)
    records = selected_records(joint)
    if len(records) != int(config["expected"]["total_dataset_records"]):
        raise RuntimeError("selected record total changed")

    # Verify both extraction phases are complete before labels are opened.
    esmc_candidates = read_esmc_candidates(config)
    generated_esmc = generated_esmc_candidates(config, paths)
    for key, values in generated_esmc.items():
        esmc_candidates[key].extend(values)
    required_hashes = set(records["sequence_sha256"].astype(str))
    if set(required_hashes) - set(esmc_candidates):
        raise RuntimeError("ESM-C coverage incomplete")
    missing_hmm = pd.read_csv(
        paths["work"] / "missing_hmm30_manifest.tsv", sep="\t", keep_default_na=False
    ).to_dict("records")
    for row in missing_hmm:
        if not hmm_status_valid(
            paths["hmm_generated"] / "status" / f"{row['sequence_sha256']}.json", row
        ):
            raise RuntimeError(f"HMM extraction incomplete: {row['sequence_sha256']}")

    md_test_qc, soppe_test_qc, qc_audit = build_frozen_test_qc(config, joint, paths)
    d1_path = Path(config["paths"]["d1_release"]) / "residue_targets.parquet"
    d1 = pd.read_parquet(d1_path)
    d1_sha = sha256_file(d1_path)
    md_test_path = paths["evaluation"] / "qc" / "md_test_frozen_confidence_by_residue.parquet"
    s_test_path = paths["evaluation"] / "qc" / "soppe_test_frozen_confidence_by_residue.parquet"
    md_test_sha = sha256_file(md_test_path)
    s_test_sha = sha256_file(s_test_path)

    label_groups: dict[str, pd.DataFrame] = {}
    for record_id, frame in d1.groupby("record_id", sort=False):
        current = frame.copy()
        current.attrs["source_path"] = str(d1_path)
        current.attrs["source_sha256"] = d1_sha
        label_groups[str(record_id)] = current
    for source_frame, source_path, source_sha in (
        (md_test_qc, md_test_path, md_test_sha),
        (soppe_test_qc, s_test_path, s_test_sha),
    ):
        for record_id, frame in source_frame.groupby("record_id", sort=False):
            current = frame.copy()
            current.attrs["source_path"] = str(source_path)
            current.attrs["source_sha256"] = source_sha
            label_groups[str(record_id)] = current

    exp_frame = experimental_manifest(config)
    exp_by_id = {
        str(row.protein_id): row._asdict() for row in exp_frame.itertuples(index=False)
    }
    s_raw = soppe_raw_map(config)
    md_hmm = md_hmm_map(config)

    endpoint_feature_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    endpoint_record_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_cache: dict[tuple[str, str], dict[str, Any]] = {}
    esmc_consistency_checks = 0
    hmm_consistency_checks = 0

    for endpoint, endpoint_records in records.groupby("dataset_endpoint", sort=True):
        print(f"[build] endpoint={endpoint} records={len(endpoint_records)}", flush=True)
        for sequence_hash, seq_records in endpoint_records.groupby("sequence_sha256", sort=True):
            first = seq_records.iloc[0]
            length = int(first["sequence_length"])
            sequence = str(first["sequence"])
            candidates = esmc_candidates[str(sequence_hash)]
            observed_arrays: list[np.ndarray] = []
            observed_hashes: list[str] = []
            for candidate in candidates:
                path = Path(candidate["path"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                observed_sha = sha256_file(path)
                if candidate.get("sha256") and observed_sha != candidate["sha256"]:
                    raise RuntimeError(f"ESM-C hash mismatch: {path}")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != (length, ESMC_DIM) or array.dtype != np.float32:
                    raise RuntimeError(f"ESM-C shape/dtype mismatch: {path}")
                if not np.isfinite(array).all():
                    raise RuntimeError(f"non-finite ESM-C array: {path}")
                observed_arrays.append(array)
                observed_hashes.append(observed_sha)
            if len(set(observed_hashes)) > 1:
                reference = np.asarray(observed_arrays[0])
                if any(not np.array_equal(reference, np.asarray(value)) for value in observed_arrays[1:]):
                    raise RuntimeError(f"inconsistent ESM-C arrays for {sequence_hash}")
            esmc_consistency_checks += max(0, len(candidates) - 1)
            selected_esmc = sorted(
                zip(candidates, observed_hashes), key=lambda item: (item[0]["origin"].startswith("md_sop"), item[0]["path"])
            )[0]
            esmc_row, esmc_sha = selected_esmc

            hmm_specs = hmm_candidates_for_sequence(
                str(sequence_hash), config, joint, s_raw, md_hmm, exp_by_id, paths
            )
            if not hmm_specs:
                raise RuntimeError(f"no HMM candidate for {sequence_hash}")
            hmm_arrays: list[np.ndarray] = []
            hmm_origins: list[str] = []
            for kind, path, origin in hmm_specs:
                if not path.is_file():
                    raise FileNotFoundError(path)
                array = load_hmm_candidate(kind, path)
                if array.shape != (length, HMM_DIM) or not np.isfinite(array).all():
                    raise RuntimeError(f"HMM shape/value mismatch: {path}")
                hmm_arrays.append(array)
                hmm_origins.append(origin)
            reference_hmm = hmm_arrays[0]
            if any(not np.array_equal(reference_hmm, value) for value in hmm_arrays[1:]):
                raise RuntimeError(f"inconsistent legacy HMM30 arrays for {sequence_hash}")
            hmm_consistency_checks += max(0, len(hmm_arrays) - 1)
            canonical_hmm = feature_group_dir(paths, str(endpoint)) / f"{sequence_hash}.npy"
            atomic_npy(canonical_hmm, reference_hmm.astype(np.float32, copy=False))
            hmm_sha = sha256_file(canonical_hmm)
            feature_key = f"{endpoint}::{sequence_hash}"
            feature_row = {
                "feature_key": feature_key,
                "dataset_endpoint": endpoint,
                "sequence_sha256": sequence_hash,
                "sequence_length": length,
                "record_count": len(seq_records),
                "record_ids": ";".join(sorted(seq_records["record_id"].astype(str))),
                "esmc_path": esmc_row["path"],
                "esmc_sha256": esmc_sha,
                "esmc_origin": esmc_row["origin"],
                "esmc_model": config["esmc"]["model"],
                "esmc_revision": config["esmc"]["model_revision"],
                "esmc_dimension": ESMC_DIM,
                "hmm30_path": str(canonical_hmm),
                "hmm30_sha256": hmm_sha,
                "hmm30_origin_candidates": ";".join(sorted(set(hmm_origins))),
                "hmm30_dimension": HMM_DIM,
                "logical_x_shape": f"[{length},{LOGICAL_DIM}]",
                "logical_x_dtype": "float32",
                "concatenation_order": "esmc600[:,0:1152] || hmm_legacy30[:,0:30]",
                "sequence_identity_verified": True,
                "component_alignment_verified": True,
            }
            endpoint_feature_rows[str(endpoint)].append(feature_row)
            feature_cache[(str(endpoint), str(sequence_hash))] = feature_row

        for ordinal, row in enumerate(endpoint_records.sort_values("record_id").itertuples(index=False), 1):
            if ordinal == 1 or ordinal % 250 == 0 or ordinal == len(endpoint_records):
                print(f"[labels] {endpoint} {ordinal}/{len(endpoint_records)}", flush=True)
            if str(endpoint) == "experimental10":
                label_frame = None
                exp_entry = exp_by_id[str(row.protein_id)]
            else:
                label_frame = label_groups.get(str(row.record_id))
                if label_frame is None:
                    raise RuntimeError(f"missing label frame: {row.record_id}")
                exp_entry = None
            label_path, stats = package_label_arrays(
                row, str(endpoint), label_frame, exp_entry, paths
            )
            feature = feature_cache[(str(endpoint), str(row.sequence_sha256))]
            endpoint_record_rows[str(endpoint)].append(
                {
                    "record_id": row.record_id,
                    "source": row.source,
                    "protein_id": row.protein_id,
                    "pdb_id": row.pdb_id,
                    "chain_id": row.chain_id,
                    "role_v1": row.role_v1,
                    "role_v2_1": row.role_v2_1,
                    "partition_v2_1": row.partition_v2_1,
                    "dataset_endpoint": endpoint,
                    "joint_homology_group": row.joint_homology_group,
                    "sequence_sha256": row.sequence_sha256,
                    "sequence_length": int(row.sequence_length),
                    "feature_key": feature["feature_key"],
                    "feature_index_path": str(feature_manifest_path(paths, str(endpoint))),
                    "esmc_path": feature["esmc_path"],
                    "esmc_sha256": feature["esmc_sha256"],
                    "hmm30_path": feature["hmm30_path"],
                    "hmm30_sha256": feature["hmm30_sha256"],
                    "logical_feature_dimension": LOGICAL_DIM,
                    "logical_feature_dtype": "float32",
                    "label_path": str(label_path),
                    **stats,
                    "allowed_label_access": row.allowed_label_access,
                }
            )

    all_feature_rows: list[dict[str, Any]] = []
    all_record_rows: list[dict[str, Any]] = []
    dataset_index_rows: list[dict[str, Any]] = []
    for endpoint in sorted(endpoint_record_rows):
        feature_frame = pd.DataFrame(endpoint_feature_rows[endpoint]).sort_values("feature_key")
        record_frame = pd.DataFrame(endpoint_record_rows[endpoint]).sort_values("record_id")
        feature_path = feature_manifest_path(paths, endpoint)
        record_path = record_manifest_path(paths, endpoint)
        atomic_tsv(feature_path, feature_frame)
        atomic_tsv(record_path, record_frame)
        all_feature_rows.extend(feature_frame.to_dict("records"))
        all_record_rows.extend(record_frame.to_dict("records"))
        dataset_index_rows.append(
            {
                "dataset_endpoint": endpoint,
                "record_count": len(record_frame),
                "unique_sequence_count": record_frame["sequence_sha256"].nunique(),
                "full_sequence_residues": int(record_frame["sequence_length"].sum()),
                "raw_label_count": int(record_frame["raw_label_count"].sum()),
                "qc_label_count": int(record_frame["qc_label_count"].sum()),
                "loss_label_count": int(record_frame["loss_label_count"].sum()),
                "record_manifest": str(record_path),
                "record_manifest_sha256": sha256_file(record_path),
                "feature_index": str(feature_path),
                "feature_index_sha256": sha256_file(feature_path),
                "access_scope": "training_or_validation"
                if endpoint.endswith("train") or endpoint.endswith("validation")
                else "final_evaluation_only",
            }
        )

    # This is the only manifest the later trainer is allowed to receive.
    development_records = pd.DataFrame(
        [
            row
            for row in all_record_rows
            if str(row["dataset_endpoint"]).endswith("train")
            or str(row["dataset_endpoint"]).endswith("validation")
        ]
    ).sort_values(["dataset_endpoint", "record_id"])
    trainer_manifest = paths["development"] / "manifests" / "trainer_access_manifest.tsv"
    atomic_tsv(trainer_manifest, development_records)
    all_features_path = paths["data_root"] / "feature_index.tsv"
    atomic_tsv(all_features_path, pd.DataFrame(all_feature_rows).sort_values("feature_key"))
    dataset_index_path = paths["manifests"] / "step3_dataset_index.tsv"
    dataset_index = pd.DataFrame(dataset_index_rows).sort_values("dataset_endpoint")
    atomic_tsv(dataset_index_path, dataset_index)
    evaluator_whitelist = {
        "schema_version": 1,
        "created_at": now_iso(),
        "policy": "Only the final evaluator may open these label manifests after checkpoints freeze.",
        "endpoints": {
            endpoint: {
                "record_manifest": str(record_manifest_path(paths, endpoint)),
                "record_manifest_sha256": sha256_file(record_manifest_path(paths, endpoint)),
            }
            for endpoint in ("md_test", "soppe_test", "experimental10")
        },
    }
    atomic_json(paths["evaluation"] / "final_evaluator_whitelist.json", evaluator_whitelist)

    expected_qc = {
        "md_train": int(config["expected"]["md_train_qc_labels"]),
        "md_validation": int(config["expected"]["md_validation_qc_labels"]),
        "md_test": int(config["expected"]["md_test_qc_labels"]),
        "soppe_train": int(config["expected"]["soppe_train_qc_labels"]),
        "soppe_validation": int(config["expected"]["soppe_validation_qc_labels"]),
        "soppe_test": int(config["expected"]["soppe_test_qc_labels"]),
        "experimental10": int(config["expected"]["experimental10_valid_labels"]),
    }
    observed_qc = dataset_index.set_index("dataset_endpoint")["qc_label_count"].astype(int).to_dict()
    checks = {
        "endpoint_record_counts_match": dataset_index.set_index("dataset_endpoint")[
            "record_count"
        ].astype(int).to_dict()
        == expected_record_counts(config),
        "endpoint_qc_label_counts_match": observed_qc == expected_qc,
        "all_required_esmc_present": len(required_hashes - set(esmc_candidates)) == 0,
        "all_required_hmm_present": True,
        "logical_dimension_1182": all(
            int(row["logical_feature_dimension"]) == LOGICAL_DIM for row in all_record_rows
        ),
        "soppe_frozen_qc_reproduces_d1_overlap": qc_audit["soppe_overlap_qc_masks_equal"],
        "trainer_manifest_contains_only_train_validation": set(
            development_records["role_v2_1"]
        )
        <= {"train", "validation"},
        "evaluation_label_paths_absent_from_trainer_manifest": not development_records[
            "label_path"
        ].astype(str).str.contains("/evaluation/").any(),
        "experimental10_982_residues": int(
            dataset_index.set_index("dataset_endpoint").loc[
                "experimental10", "full_sequence_residues"
            ]
        )
        == int(config["expected"]["experimental10_residues"]),
        "test_scores_not_computed": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Step-3 build checks failed: {checks}")
    build_audit = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "generated_at": now_iso(),
        "status": "PASS_BUILD_PENDING_INDEPENDENT_AUDIT",
        "checks": checks,
        "endpoint_summary": dataset_index_rows,
        "esmc_duplicate_candidate_comparisons": esmc_consistency_checks,
        "hmm_duplicate_candidate_comparisons": hmm_consistency_checks,
        "feature_storage_policy": "ESM-C arrays referenced by immutable path/hash; canonical HMM30 arrays materialized; x is logical concatenation at load time.",
        "label_storage_policy": "Development and evaluation labels are physically separated; invalid labels are zero placeholders with explicit masks.",
        "test_label_values_used_for_split_or_configuration": False,
        "test_predictive_scores_computed": False,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_access_manifest": str(trainer_manifest),
        "trainer_access_manifest_sha256": sha256_file(trainer_manifest),
        "feature_index": str(all_features_path),
        "feature_index_sha256": sha256_file(all_features_path),
        "dataset_index": str(dataset_index_path),
        "dataset_index_sha256": sha256_file(dataset_index_path),
        "python": sys.version,
        "platform": platform.platform(),
    }
    atomic_json(paths["audits"] / "step3_build_audit.json", build_audit)
    print(json.dumps(build_audit, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, required=True)
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", parents=[common])
    commands.add_parser("extract-esmc", parents=[common])
    hmm = commands.add_parser("extract-hmm", parents=[common])
    hmm.add_argument("--workers", type=int)
    commands.add_parser("build", parents=[common])
    return root


def main() -> int:
    args = parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.command == "prepare":
        return prepare(config, config_path)
    if args.command == "extract-esmc":
        return run_esmc(config)
    if args.command == "extract-hmm":
        return run_hmm(config, args.workers)
    if args.command == "build":
        return build(config, config_path)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
