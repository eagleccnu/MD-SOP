#!/usr/bin/env python3
"""Training utilities frozen for MD-SOP v2.2 step 5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from mdsop_dataset_v2_2 import (
    AccessControlledProteinDataset,
    ManifestAccessPolicy,
    collate_protein_records,
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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def atomic_torch_save(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class CyclingShuffleStream:
    """Independent deterministic shuffled queue that never drops a tail."""

    def __init__(self, population_size: int, seed: int) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        self.population_size = int(population_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.order = self.rng.permutation(self.population_size).astype(np.int64)
        self.cursor = 0
        self.draws = 0
        self.completed_cycles = 0

    def next_indices(self, count: int) -> List[int]:
        if count <= 0:
            raise ValueError("count must be positive")
        result: List[int] = []
        while len(result) < count:
            if self.cursor == self.population_size:
                self.completed_cycles += 1
                self.order = self.rng.permutation(self.population_size).astype(np.int64)
                self.cursor = 0
            take = min(count - len(result), self.population_size - self.cursor)
            result.extend(int(value) for value in self.order[self.cursor : self.cursor + take])
            self.cursor += take
            self.draws += take
        return result

    def state_dict(self) -> Dict[str, object]:
        return {
            "population_size": self.population_size,
            "seed": self.seed,
            "order": self.order.copy(),
            "cursor": self.cursor,
            "draws": self.draws,
            "completed_cycles": self.completed_cycles,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["population_size"]) != self.population_size or int(state["seed"]) != self.seed:
            raise ValueError("Cycling stream state is incompatible with this population/seed")
        order = np.asarray(state["order"], dtype=np.int64)
        if order.shape != (self.population_size,) or sorted(order.tolist()) != list(
            range(self.population_size)
        ):
            raise ValueError("Cycling stream state has an invalid permutation")
        cursor = int(state["cursor"])
        if not 0 <= cursor <= self.population_size:
            raise ValueError("Cycling stream cursor is out of range")
        self.order = order
        self.cursor = cursor
        self.draws = int(state["draws"])
        self.completed_cycles = int(state["completed_cycles"])
        self.rng.bit_generator.state = state["rng_state"]

    @property
    def equivalent_full_traversals(self) -> float:
        return self.draws / self.population_size


class MutableBatchSampler(Sampler[List[int]]):
    """Batch sampler whose frozen epoch batches are replaced between iterators."""

    def __init__(self) -> None:
        self.batches: List[List[int]] = []

    def set_batches(self, batches: Sequence[Sequence[int]]) -> None:
        self.batches = [list(map(int, batch)) for batch in batches]
        if not self.batches or any(not batch for batch in self.batches):
            raise ValueError("Epoch batch list must be non-empty and contain non-empty batches")

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_training_loader(
    dataset: AccessControlledProteinDataset,
    sampler: MutableBatchSampler,
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    worker_seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(worker_seed))
    kwargs: Dict[str, object] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": int(num_workers),
        "collate_fn": collate_protein_records,
        "pin_memory": bool(pin_memory),
        "worker_init_fn": seed_dataloader_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def make_validation_loader(
    dataset: AccessControlledProteinDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    worker_seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(worker_seed))
    kwargs: Dict[str, object] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(num_workers),
        "collate_fn": collate_protein_records,
        "pin_memory": bool(pin_memory),
        "worker_init_fn": seed_dataloader_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def move_batch_to_device(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def safe_pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2 or np.ptp(prediction) == 0.0 or np.ptp(target) == 0.0:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def safe_spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2 or np.ptp(prediction) == 0.0 or np.ptp(target) == 0.0:
        return float("nan")
    return float(spearmanr(prediction, target).statistic)


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if finite.any() else float("nan")


def summarize_validation_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not records:
        raise ValueError("No validation records were supplied")
    per_record = []
    pooled_prediction = []
    pooled_target = []
    for record in records:
        prediction = np.asarray(record["prediction"], dtype=np.float64)
        target = np.asarray(record["target"], dtype=np.float64)
        if prediction.shape != target.shape or prediction.ndim != 1 or prediction.size == 0:
            raise ValueError("Each validation record must contain aligned non-empty vectors")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Validation vectors contain non-finite values")
        error = prediction - target
        per_record.append(
            {
                "record_id": str(record["record_id"]),
                "sequence_sha256": str(record.get("sequence_sha256", "")),
                "eligible_residues": int(prediction.size),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "pcc": safe_pearson(prediction, target),
                "spearman": safe_spearman(prediction, target),
                "out_of_range_fraction": float(np.mean((prediction < 0.0) | (prediction > 1.0))),
            }
        )
        pooled_prediction.append(prediction)
        pooled_target.append(target)

    metrics = ("rmse", "mae", "bias", "pcc", "spearman", "out_of_range_fraction")
    macro = {name: _nanmean([float(row[name]) for row in per_record]) for name in metrics}
    macro["pcc_valid_records"] = sum(np.isfinite(float(row["pcc"])) for row in per_record)
    macro["spearman_valid_records"] = sum(
        np.isfinite(float(row["spearman"])) for row in per_record
    )

    prediction_all = np.concatenate(pooled_prediction)
    target_all = np.concatenate(pooled_target)
    pooled_error = prediction_all - target_all
    pooled = {
        "rmse": float(np.sqrt(np.mean(pooled_error**2))),
        "mae": float(np.mean(np.abs(pooled_error))),
        "bias": float(np.mean(pooled_error)),
        "pcc": safe_pearson(prediction_all, target_all),
        "spearman": safe_spearman(prediction_all, target_all),
        "out_of_range_fraction": float(
            np.mean((prediction_all < 0.0) | (prediction_all > 1.0))
        ),
    }

    by_sequence: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in per_record:
        sequence_key = str(row["sequence_sha256"]) or str(row["record_id"])
        by_sequence[sequence_key].append(row)
    unique_sequence_macro = {}
    for name in metrics:
        sequence_means = [
            _nanmean([float(row[name]) for row in sequence_rows])
            for sequence_rows in by_sequence.values()
        ]
        unique_sequence_macro[name] = _nanmean(sequence_means)

    return {
        "record_count": len(per_record),
        "unique_sequence_count": len(by_sequence),
        "eligible_residue_count": int(sum(row["eligible_residues"] for row in per_record)),
        "protein_record_macro": macro,
        "pooled": pooled,
        "unique_sequence_macro": unique_sequence_macro,
        "per_record": per_record,
    }


def capture_rng_state(device: torch.device) -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: Mapping[str, object], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def history_to_tsv(history: Sequence[Mapping[str, object]]) -> str:
    if not history:
        return ""
    preferred = [
        "virtual_epoch",
        "global_step",
        "train_loss",
        "train_md_loss",
        "train_soppe_loss",
        "selection_metric",
        "md_validation_macro_rmse",
        "soppe_validation_macro_rmse",
        "md_validation_macro_mae",
        "soppe_validation_macro_mae",
        "md_validation_macro_pcc",
        "soppe_validation_macro_pcc",
        "learning_rate",
        "gradient_norm_mean",
        "gradient_norm_max",
        "best_epoch",
        "best_selection_metric",
        "early_stop_bad_epochs",
        "md_record_exposures",
        "soppe_record_exposures",
        "md_effective_residue_exposures",
        "soppe_effective_residue_exposures",
        "epoch_seconds",
        "elapsed_seconds",
    ]
    extras = sorted(set().union(*(row.keys() for row in history)) - set(preferred))
    fields = [field for field in preferred if any(field in row for row in history)] + extras
    output = []
    output.append("\t".join(fields))
    for row in history:
        values = []
        for field in fields:
            value = row.get(field, "")
            if value is None:
                value = ""
            elif isinstance(value, float):
                value = f"{value:.12g}"
            values.append(str(value))
        output.append("\t".join(values))
    return "\n".join(output) + "\n"


def _history_series(history: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    values = []
    for row in history:
        value = row.get(key)
        try:
            values.append(float(value) if value not in (None, "") else np.nan)
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def save_training_curves(
    history: Sequence[Mapping[str, object]],
    png_path: Path,
    svg_path: Path,
    experiment_id: str,
) -> None:
    if not history:
        return
    epochs = _history_series(history, "virtual_epoch")
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 10.5), sharex=True)

    for key, label, color in (
        ("train_loss", "total train loss", "#1f77b4"),
        ("train_md_loss", "MD train loss", "#2ca02c"),
        ("train_soppe_loss", "S-OPPE train loss", "#ff7f0e"),
    ):
        values = _history_series(history, key)
        if np.isfinite(values).any():
            axes[0].plot(epochs, values, marker="o", markersize=2.5, linewidth=1.4, label=label, color=color)
    axes[0].set_ylabel("SmoothL1 loss")
    axes[0].set_title(f"{experiment_id}: training-loss trend")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    for key, label, color in (
        ("selection_metric", "checkpoint selection metric", "#d62728"),
        ("md_validation_macro_rmse", "MD validation macro RMSE", "#2ca02c"),
        ("soppe_validation_macro_rmse", "S-OPPE validation macro RMSE", "#ff7f0e"),
    ):
        values = _history_series(history, key)
        if np.isfinite(values).any():
            axes[1].plot(epochs, values, marker="o", markersize=2.5, linewidth=1.4, label=label, color=color)
    axes[1].set_ylabel("Validation RMSE")
    axes[1].set_title("Development validation trend")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    learning_rate = _history_series(history, "learning_rate")
    axes[2].plot(epochs, learning_rate, marker="o", markersize=2.5, linewidth=1.4, color="#9467bd")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Virtual epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning-rate schedule")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    for path, file_format in ((Path(png_path), "png"), (Path(svg_path), "svg")):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        fig.savefig(temporary, format=file_format, dpi=180 if file_format == "png" else None)
        os.replace(temporary, path)
    plt.close(fig)


def make_access_policy(config: Mapping[str, object]) -> ManifestAccessPolicy:
    access = config["data_access"]
    return ManifestAccessPolicy(
        manifest_path=Path(access["trainer_manifest"]),
        manifest_sha256=str(access["trainer_manifest_sha256"]),
        development_label_root=Path(access["development_label_root"]),
        evaluation_root=Path(access["evaluation_root"]),
    )


def validate_frozen_training_config(config: Mapping[str, object]) -> None:
    if tuple(config["experiments"].keys()) != EXPERIMENT_IDS:
        raise ValueError("Experiment matrix/order differs from the frozen six-run protocol")
    training = config["training"]
    access = config["data_access"]
    calculated = math.ceil(
        max(int(access["expected_supervised_train_records"]["md_ired"]), int(access["expected_supervised_train_records"]["soppe_legacy"]))
        / int(training["total_batch_records"])
    )
    if calculated != 69 or int(training["steps_per_virtual_epoch"]) != calculated:
        raise ValueError(f"Frozen steps_per_virtual_epoch must be 69, got {calculated}")
    if int(training["max_optimizer_steps"]) != int(training["max_virtual_epochs"]) * calculated:
        raise ValueError("max_optimizer_steps does not match virtual-epoch budget")
    if int(training["mixed_md_records_per_step"]) != 16 or int(training["mixed_soppe_records_per_step"]) != 16:
        raise ValueError("Mixed batches must contain exactly 16 records from each source")
    if int(training["single_source_records_per_step"]) != 32:
        raise ValueError("Single-source batches must contain exactly 32 records")
    for experiment_id, experiment in config["experiments"].items():
        if int(experiment["window_size"]) not in (1, 15):
            raise ValueError(f"{experiment_id}: unsupported window size")


def experiment_output_paths(config: Mapping[str, object], experiment_id: str) -> Dict[str, Path]:
    artifacts = config["artifacts"]
    model_dir = Path(config["model_root"]) / experiment_id
    result_dir = Path(config["result_root"]) / experiment_id
    return {
        "model_dir": model_dir,
        "result_dir": result_dir,
        "best": model_dir / artifacts["best_checkpoint_name"],
        "last": model_dir / artifacts["last_checkpoint_name"],
        "history": result_dir / artifacts["history_name"],
        "progress": result_dir / artifacts["progress_name"],
        "run_manifest": result_dir / artifacts["run_manifest_name"],
        "exposures": result_dir / artifacts["exposure_counts_name"],
        "curve_png": result_dir / artifacts["curve_png_name"],
        "curve_svg": result_dir / artifacts["curve_svg_name"],
        "validation_summary": result_dir / artifacts["validation_summary_name"],
    }


def checkpoint_payload(
    *,
    experiment_id: str,
    config_sha256: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    early_stop_reference: float,
    early_stop_bad_epochs: int,
    streams: Mapping[str, CyclingShuffleStream],
    history: Sequence[Mapping[str, object]],
    exposures: Mapping[str, object],
    device: torch.device,
    status: str,
    identity: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return {
        **dict(identity or {}),
        "format": TRAINING_CHECKPOINT_FORMAT,
        "experiment_id": experiment_id,
        "config_sha256": config_sha256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "early_stop_reference": float(early_stop_reference),
        "early_stop_bad_epochs": int(early_stop_bad_epochs),
        "stream_states": {name: stream.state_dict() for name, stream in streams.items()},
        "history": list(history),
        "exposures": dict(exposures),
        "rng_state": capture_rng_state(device),
        "status": status,
    }

