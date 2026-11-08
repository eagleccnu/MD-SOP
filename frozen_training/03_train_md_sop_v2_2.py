#!/usr/bin/env python3
"""Train one frozen MD-SOP v2.2 development experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import socket
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

from mdsop_dataset_v2_2 import AccessControlledProteinDataset
from mdsop_loss_v2_2 import (
    mixed_source_record_macro_smooth_l1,
    record_macro_smooth_l1,
)
from mdsop_training_v2_2 import (
    EXPERIMENT_IDS,
    TRAINING_CHECKPOINT_FORMAT,
    CyclingShuffleStream,
    MutableBatchSampler,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    checkpoint_payload,
    experiment_output_paths,
    history_to_tsv,
    make_access_policy,
    make_training_loader,
    make_validation_loader,
    move_batch_to_device,
    optimizer_to_device,
    restore_rng_state,
    save_training_curves,
    sha256_file,
    summarize_validation_records,
    validate_frozen_training_config,
)
from model_bilstm_window_v2_2 import load_model_checkpoint, state_dict_sha256
from mdsop_runtime_v2_2 import load_runtime_config, runtime_identity, validate_resume_checkpoint


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def validate_initialization(config: Mapping[str, object], window_size: int) -> Path:
    initialization = config["initialization"]
    key = "window_1" if window_size == 1 else "window_15"
    path = Path(initialization[key]["checkpoint"])
    observed_file_hash = sha256_file(path)
    if observed_file_hash != initialization[key]["file_sha256"]:
        raise RuntimeError(f"Initialization checkpoint hash mismatch: {path}")
    model, payload = load_model_checkpoint(path, expected_window_size=window_size)
    observed_backbone = state_dict_sha256(model.backbone_state_dict())
    if observed_backbone != initialization["common_backbone_state_sha256"]:
        raise RuntimeError("Initialization backbone hash differs from the frozen common backbone")
    if int(payload["seed"]) != int(config["random_streams"]["model_initialization_seed"]):
        raise RuntimeError("Initialization seed differs from the frozen model seed")
    return path


def source_row_indices(dataset: AccessControlledProteinDataset) -> Dict[str, List[int]]:
    result = {"md_ired": [], "soppe_legacy": []}
    for index, row in enumerate(dataset.rows):
        result[row["source"]].append(index)
    return {key: value for key, value in result.items() if value}


def build_epoch_batches(
    *,
    train_sources: Sequence[str],
    source_indices: Mapping[str, Sequence[int]],
    streams: Mapping[str, CyclingShuffleStream],
    steps: int,
    single_batch_size: int,
    mixed_md_size: int,
    mixed_soppe_size: int,
) -> List[List[int]]:
    batches = []
    if len(train_sources) == 1:
        source = train_sources[0]
        index_map = source_indices[source]
        for _ in range(steps):
            local = streams[source].next_indices(single_batch_size)
            batches.append([int(index_map[index]) for index in local])
    elif list(train_sources) == ["md_ired", "soppe_legacy"]:
        md_map = source_indices["md_ired"]
        soppe_map = source_indices["soppe_legacy"]
        for _ in range(steps):
            md_local = streams["md_ired"].next_indices(mixed_md_size)
            soppe_local = streams["soppe_legacy"].next_indices(mixed_soppe_size)
            batch = [int(md_map[index]) for index in md_local]
            batch.extend(int(soppe_map[index]) for index in soppe_local)
            batches.append(batch)
    else:
        raise ValueError(f"Unsupported train source route: {train_sources}")
    return batches


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    sequence_hash_by_record: Mapping[str, str],
) -> Dict[str, object]:
    model.eval()
    records = []
    for batch_cpu in loader:
        batch = move_batch_to_device(batch_cpu, device)
        prediction = model(batch["x"], batch["lengths"], batch["residue_mask"])
        if prediction.shape != batch["y"].shape or not torch.isfinite(prediction).all():
            raise RuntimeError("Validation prediction is malformed or non-finite")
        for row_index, record_id in enumerate(batch["record_ids"]):
            valid = batch["eval_mask"][row_index]
            if int(valid.sum().item()) <= 0:
                raise RuntimeError(f"Validation record {record_id} has no evaluation labels")
            records.append(
                {
                    "record_id": record_id,
                    "sequence_sha256": sequence_hash_by_record[record_id],
                    "prediction": prediction[row_index][valid].detach().cpu().numpy(),
                    "target": batch["y"][row_index][valid].detach().cpu().numpy(),
                }
            )
    return summarize_validation_records(records)


def update_exposures(exposures: Dict[str, object], batch: Mapping[str, object]) -> None:
    record_counts: Counter = exposures["record_counts"]
    source_records: Counter = exposures["source_record_exposures"]
    source_residues: Counter = exposures["source_effective_residue_exposures"]
    residue_counts = batch["loss_mask"].sum(dim=1).detach().cpu().tolist()
    for record_id, source, residue_count in zip(
        batch["record_ids"], batch["sources"], residue_counts
    ):
        record_counts[record_id] += 1
        source_records[source] += 1
        source_residues[source] += int(residue_count)


def serialize_exposures(
    exposures: Mapping[str, object], streams: Mapping[str, CyclingShuffleStream]
) -> Dict[str, object]:
    return {
        "record_counts": dict(sorted(exposures["record_counts"].items())),
        "source_record_exposures": dict(exposures["source_record_exposures"]),
        "source_effective_residue_exposures": dict(
            exposures["source_effective_residue_exposures"]
        ),
        "source_equivalent_full_traversals": {
            source: stream.equivalent_full_traversals for source, stream in streams.items()
        },
        "source_queue_draws": {source: stream.draws for source, stream in streams.items()},
    }


def restore_exposures(saved: Mapping[str, object]) -> Dict[str, object]:
    return {
        "record_counts": Counter(saved.get("record_counts", {})),
        "source_record_exposures": Counter(saved.get("source_record_exposures", {})),
        "source_effective_residue_exposures": Counter(
            saved.get("source_effective_residue_exposures", {})
        ),
    }


def compact_validation(summary: Mapping[str, object]) -> Dict[str, object]:
    return {
        "record_count": summary["record_count"],
        "unique_sequence_count": summary["unique_sequence_count"],
        "eligible_residue_count": summary["eligible_residue_count"],
        "protein_record_macro": summary["protein_record_macro"],
        "pooled": summary["pooled"],
        "unique_sequence_macro": summary["unique_sequence_macro"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True, choices=EXPERIMENT_IDS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--validate-only", action="store_true", help="Validate bindings and development identities without creating formal outputs or training")
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit last-checkpoint path",
    )
    args = parser.parse_args()

    raw_config_text = args.config.read_text(encoding="utf-8")
    config = load_runtime_config(args.config)
    validate_frozen_training_config(config)
    config_sha256 = sha256_file(args.config)
    experiment_id = args.experiment_id
    experiment = config["experiments"][experiment_id]
    train_sources = list(experiment["train_sources"])
    validation_sources = list(experiment["validation_sources"])
    window_size = int(experiment["window_size"])
    training = config["training"]
    streams_config = config["random_streams"]
    paths = experiment_output_paths(config, experiment_id)
    if args.validate_only:
        validate_initialization(config, window_size)
        policy = make_access_policy(config)
        counts = {source: len(AccessControlledProteinDataset(policy, roles=["train"], sources=[source])) for source in train_sources}
        log(json.dumps({"status": "PASS", "validate_only": True, "experiment_id": experiment_id, "train_counts": counts}))
        return 0
    for directory_key in ("model_dir", "result_dir"):
        paths[directory_key].mkdir(parents=True, exist_ok=True)
    snapshot_path = paths["result_dir"] / "run_config_snapshot_v2_2_20261107.yaml"
    if snapshot_path.exists() and sha256_file(snapshot_path) != config_sha256:
        raise RuntimeError("Existing run config snapshot differs from the requested frozen config")
    atomic_write_text(snapshot_path, raw_config_text)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal step-5 training requires an explicitly selected CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    initialization_path = validate_initialization(config, window_size)
    policy = make_access_policy(config)
    train_dataset = AccessControlledProteinDataset(
        policy, roles=["train"], sources=train_sources, drop_zero_supervision=True
    )
    indices_by_source = source_row_indices(train_dataset)
    expected_train = config["data_access"]["expected_supervised_train_records"]
    for source in train_sources:
        if len(indices_by_source[source]) != int(expected_train[source]):
            raise RuntimeError(
                f"{source} supervised train count {len(indices_by_source[source])} "
                f"!= frozen {expected_train[source]}"
            )

    validation_datasets = {
        source: AccessControlledProteinDataset(
            policy, roles=["validation"], sources=[source], drop_zero_supervision=True
        )
        for source in validation_sources
    }
    expected_validation = config["data_access"]["expected_validation_records"]
    for source, dataset in validation_datasets.items():
        if len(dataset) != int(expected_validation[source]):
            raise RuntimeError(f"{source} validation count differs from frozen config")

    sequence_hash_by_record = {
        row["record_id"]: row.get("sequence_sha256", "")
        for dataset in validation_datasets.values()
        for row in dataset.rows
    }

    seed_by_source = {
        "md_ired": int(streams_config["md_shuffle_seed"]),
        "soppe_legacy": int(streams_config["soppe_shuffle_seed"]),
    }
    streams = {
        source: CyclingShuffleStream(len(indices_by_source[source]), seed_by_source[source])
        for source in train_sources
    }
    mutable_sampler = MutableBatchSampler()
    # Seed it with a structurally valid batch before DataLoader construction.
    mutable_sampler.set_batches([[0]])
    train_loader = make_training_loader(
        train_dataset,
        mutable_sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=bool(training["pin_memory"]),
        persistent_workers=bool(training["persistent_workers"]),
        worker_seed=int(streams_config["dataloader_worker_seed"]),
    )
    validation_loaders = {
        source: make_validation_loader(
            dataset,
            batch_size=int(training["total_batch_records"]),
            num_workers=int(training["validation_num_workers"]),
            pin_memory=bool(training["pin_memory"]),
            persistent_workers=bool(training["persistent_workers"]),
            worker_seed=int(streams_config["dataloader_worker_seed"]) + 100 + index,
        )
        for index, (source, dataset) in enumerate(validation_datasets.items())
    }

    model, _ = load_model_checkpoint(initialization_path, expected_window_size=window_size)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
    )
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_config["mode"]),
        patience=int(scheduler_config["patience"]),
        factor=float(scheduler_config["factor"]),
        min_lr=float(scheduler_config["min_learning_rate"]),
    )

    start_epoch = 1
    global_step = 0
    best_metric = float("inf")
    best_epoch = 0
    early_stop_reference = float("inf")
    early_stop_bad_epochs = 0
    history: List[Dict[str, object]] = []
    exposures: Dict[str, object] = {
        "record_counts": Counter(),
        "source_record_exposures": Counter(),
        "source_effective_residue_exposures": Counter(),
    }
    resume_path = None
    if args.resume == "auto" and paths["last"].is_file():
        resume_path = paths["last"]
    elif args.resume not in ("auto", "none"):
        resume_path = Path(args.resume)
    elif args.resume == "none" and any(paths[key].exists() for key in ("last", "best", "history")):
        raise RuntimeError("Formal output already exists; use --resume auto rather than overwrite it")

    if resume_path is not None:
        if resume_path.resolve() != paths["last"].resolve():
            raise RuntimeError("Resume is restricted to this experiment's v2.2 last checkpoint")
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(saved, config, config_sha256, experiment_id)
        if saved.get("status") in {"early_stopped", "max_epochs_complete"}:
            log(f"{experiment_id} is already complete in {resume_path}; no changes made")
            return 0
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        global_step = int(saved["global_step"])
        best_metric = float(saved["best_metric"])
        best_epoch = int(saved["best_epoch"])
        early_stop_reference = float(saved["early_stop_reference"])
        early_stop_bad_epochs = int(saved["early_stop_bad_epochs"])
        history = [dict(row) for row in saved["history"]]
        exposures = restore_exposures(saved["exposures"])
        for source, stream in streams.items():
            stream.load_state_dict(saved["stream_states"][source])
        restore_rng_state(saved["rng_state"], device)
        log(f"resumed {experiment_id} from epoch {start_epoch - 1}, global_step {global_step}")
    else:
        dropout_seed = int(streams_config["dropout_seed"])
        random.seed(dropout_seed)
        np.random.seed(dropout_seed % (2**32))
        torch.manual_seed(dropout_seed)
        torch.cuda.manual_seed_all(dropout_seed)

    run_started = now_iso()
    wall_started = time.perf_counter()
    run_manifest = {
        **runtime_identity(config),
        "schema_version": "1.0",
        "status": "running",
        "experiment_id": experiment_id,
        "run_tag": config["run_tag"],
        "plan_version": config["plan_version"],
        "config_path": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "initialization_checkpoint": str(initialization_path),
        "initialization_sha256": sha256_file(initialization_path),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "started_at": run_started,
        "train_sources": train_sources,
        "validation_sources": validation_sources,
        "window_size": window_size,
        "test_or_reference_labels_opened": False,
    }
    atomic_write_json(paths["run_manifest"], run_manifest)
    log(
        f"start {experiment_id}: device={device}, window={window_size}, "
        f"train_records={len(train_dataset)}, validation="
        f"{ {source: len(dataset) for source, dataset in validation_datasets.items()} }"
    )

    terminal_status = "running"
    try:
        for epoch in range(start_epoch, int(training["max_virtual_epochs"]) + 1):
            epoch_started = time.perf_counter()
            batches = build_epoch_batches(
                train_sources=train_sources,
                source_indices=indices_by_source,
                streams=streams,
                steps=int(training["steps_per_virtual_epoch"]),
                single_batch_size=int(training["single_source_records_per_step"]),
                mixed_md_size=int(training["mixed_md_records_per_step"]),
                mixed_soppe_size=int(training["mixed_soppe_records_per_step"]),
            )
            mutable_sampler.set_batches(batches)
            model.train()
            total_losses = []
            md_losses = []
            soppe_losses = []
            gradient_norms = []
            for epoch_step, batch_cpu in enumerate(train_loader, start=1):
                if epoch_step > int(training["steps_per_virtual_epoch"]):
                    raise RuntimeError("Training DataLoader yielded too many batches")
                batch = move_batch_to_device(batch_cpu, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["x"], batch["lengths"], batch["residue_mask"])
                if len(train_sources) == 2:
                    loss_result = mixed_source_record_macro_smooth_l1(
                        prediction,
                        batch["y"],
                        batch["loss_mask"],
                        batch["sources"],
                        beta=float(training["smooth_l1_beta"]),
                    )
                    loss = loss_result.loss
                    md_losses.append(float(loss_result.md_loss.loss.detach().cpu()))
                    soppe_losses.append(float(loss_result.soppe_loss.loss.detach().cpu()))
                else:
                    loss_result = record_macro_smooth_l1(
                        prediction,
                        batch["y"],
                        batch["loss_mask"],
                        beta=float(training["smooth_l1_beta"]),
                    )
                    loss = loss_result.loss
                    if train_sources[0] == "md_ired":
                        md_losses.append(float(loss.detach().cpu()))
                    else:
                        soppe_losses.append(float(loss.detach().cpu()))
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss became non-finite")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_global_norm"])
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Gradient norm became non-finite")
                optimizer.step()
                global_step += 1
                total_losses.append(float(loss.detach().cpu()))
                gradient_norms.append(float(gradient_norm.detach().cpu()))
                update_exposures(exposures, batch)
                if epoch_step % int(training["progress_every_steps"]) == 0:
                    log(
                        f"{experiment_id} epoch={epoch:03d} step={epoch_step:02d}/"
                        f"{training['steps_per_virtual_epoch']} global={global_step} "
                        f"loss={np.mean(total_losses[-10:]):.6f}"
                    )
            if len(total_losses) != int(training["steps_per_virtual_epoch"]):
                raise RuntimeError("Training DataLoader yielded too few batches")

            validation = {
                source: run_validation(
                    model, validation_loaders[source], device, sequence_hash_by_record
                )
                for source in validation_sources
            }
            md_rmse = (
                float(validation["md_ired"]["protein_record_macro"]["rmse"])
                if "md_ired" in validation
                else None
            )
            soppe_rmse = (
                float(validation["soppe_legacy"]["protein_record_macro"]["rmse"])
                if "soppe_legacy" in validation
                else None
            )
            if md_rmse is not None and soppe_rmse is not None:
                selection_metric = 0.5 * md_rmse + 0.5 * soppe_rmse
            elif md_rmse is not None:
                selection_metric = md_rmse
            elif soppe_rmse is not None:
                selection_metric = soppe_rmse
            else:
                raise RuntimeError("No validation selection metric is available")
            if not math.isfinite(selection_metric):
                raise RuntimeError("Validation selection metric is non-finite")

            scheduler.step(selection_metric)
            is_best = selection_metric < best_metric
            if is_best:
                best_metric = selection_metric
                best_epoch = epoch
            min_delta = float(config["early_stopping"]["min_delta"])
            if selection_metric < early_stop_reference - min_delta:
                early_stop_reference = selection_metric
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1
            should_stop = early_stop_bad_epochs >= int(
                config["early_stopping"]["patience_virtual_epochs"]
            )
            if should_stop:
                terminal_status = "early_stopped"
            elif epoch == int(training["max_virtual_epochs"]):
                terminal_status = "max_epochs_complete"
            else:
                terminal_status = "running"

            serialized_exposures = serialize_exposures(exposures, streams)
            elapsed = time.perf_counter() - wall_started
            row: Dict[str, object] = {
                "virtual_epoch": epoch,
                "global_step": global_step,
                "train_loss": float(np.mean(total_losses)),
                "train_md_loss": float(np.mean(md_losses)) if md_losses else None,
                "train_soppe_loss": float(np.mean(soppe_losses)) if soppe_losses else None,
                "selection_metric": selection_metric,
                "md_validation_macro_rmse": md_rmse,
                "soppe_validation_macro_rmse": soppe_rmse,
                "md_validation_macro_mae": (
                    float(validation["md_ired"]["protein_record_macro"]["mae"])
                    if "md_ired" in validation
                    else None
                ),
                "soppe_validation_macro_mae": (
                    float(validation["soppe_legacy"]["protein_record_macro"]["mae"])
                    if "soppe_legacy" in validation
                    else None
                ),
                "md_validation_macro_pcc": (
                    float(validation["md_ired"]["protein_record_macro"]["pcc"])
                    if "md_ired" in validation
                    else None
                ),
                "soppe_validation_macro_pcc": (
                    float(validation["soppe_legacy"]["protein_record_macro"]["pcc"])
                    if "soppe_legacy" in validation
                    else None
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "gradient_norm_max": float(np.max(gradient_norms)),
                "best_epoch": best_epoch,
                "best_selection_metric": best_metric,
                "early_stop_bad_epochs": early_stop_bad_epochs,
                "md_record_exposures": int(
                    serialized_exposures["source_record_exposures"].get("md_ired", 0)
                ),
                "soppe_record_exposures": int(
                    serialized_exposures["source_record_exposures"].get("soppe_legacy", 0)
                ),
                "md_effective_residue_exposures": int(
                    serialized_exposures["source_effective_residue_exposures"].get(
                        "md_ired", 0
                    )
                ),
                "soppe_effective_residue_exposures": int(
                    serialized_exposures["source_effective_residue_exposures"].get(
                        "soppe_legacy", 0
                    )
                ),
                "epoch_seconds": time.perf_counter() - epoch_started,
                "elapsed_seconds": elapsed,
            }
            history.append(row)
            atomic_write_text(paths["history"], history_to_tsv(history))
            atomic_write_json(paths["exposures"], serialized_exposures)
            if epoch % int(training["plot_every_epochs"]) == 0 or terminal_status != "running":
                save_training_curves(
                    history, paths["curve_png"], paths["curve_svg"], experiment_id
                )

            payload = checkpoint_payload(
                experiment_id=experiment_id,
                config_sha256=config_sha256,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                best_epoch=best_epoch,
                early_stop_reference=early_stop_reference,
                early_stop_bad_epochs=early_stop_bad_epochs,
                streams=streams,
                history=history,
                exposures=serialized_exposures,
                device=device,
                status=terminal_status,
                identity=runtime_identity(config),
            )
            if is_best:
                atomic_torch_save(paths["best"], payload)
                atomic_write_json(
                    paths["validation_summary"],
                    {
                        "experiment_id": experiment_id,
                        "best_epoch": best_epoch,
                        "best_selection_metric": best_metric,
                        "validation": {
                            source: compact_validation(summary)
                            for source, summary in validation.items()
                        },
                    },
                )
            atomic_torch_save(paths["last"], payload)
            atomic_write_json(
                paths["progress"],
                {
                    "status": terminal_status,
                    "experiment_id": experiment_id,
                    "virtual_epoch": epoch,
                    "global_step": global_step,
                    "train_loss": row["train_loss"],
                    "selection_metric": selection_metric,
                    "best_epoch": best_epoch,
                    "best_selection_metric": best_metric,
                    "early_stop_bad_epochs": early_stop_bad_epochs,
                    "learning_rate": row["learning_rate"],
                    "updated_at": now_iso(),
                    "curve_png": str(paths["curve_png"]),
                },
            )
            log(
                f"{experiment_id} epoch={epoch:03d} train={row['train_loss']:.6f} "
                f"selection={selection_metric:.6f} best={best_metric:.6f}@{best_epoch} "
                f"lr={row['learning_rate']:.3g} bad={early_stop_bad_epochs} "
                f"seconds={row['epoch_seconds']:.1f} status={terminal_status}"
            )
            if terminal_status != "running":
                break

        if terminal_status == "running":
            raise RuntimeError("Training loop ended without a terminal state")
        finished_at = now_iso()
        run_manifest.update(
            {
                "status": terminal_status,
                "finished_at": finished_at,
                "duration_seconds": time.perf_counter() - wall_started,
                "completed_virtual_epochs": len(history),
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_selection_metric": best_metric,
                "artifacts": {
                    "best_checkpoint": str(paths["best"]),
                    "best_checkpoint_sha256": sha256_file(paths["best"]),
                    "last_checkpoint": str(paths["last"]),
                    "last_checkpoint_sha256": sha256_file(paths["last"]),
                    "history": str(paths["history"]),
                    "history_sha256": sha256_file(paths["history"]),
                    "curve_png": str(paths["curve_png"]),
                    "curve_png_sha256": sha256_file(paths["curve_png"]),
                    "curve_svg": str(paths["curve_svg"]),
                    "curve_svg_sha256": sha256_file(paths["curve_svg"]),
                    "exposure_counts": str(paths["exposures"]),
                    "exposure_counts_sha256": sha256_file(paths["exposures"]),
                    "best_validation_summary": str(paths["validation_summary"]),
                    "best_validation_summary_sha256": sha256_file(
                        paths["validation_summary"]
                    ),
                },
            }
        )
        atomic_write_json(paths["run_manifest"], run_manifest)
        log(
            f"completed {experiment_id}: status={terminal_status}, epochs={len(history)}, "
            f"steps={global_step}, best={best_metric:.6f}@{best_epoch}"
        )
        return 0
    except Exception as exc:
        run_manifest.update(
            {
                "status": "failed",
                "failed_at": now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "last_completed_epoch": history[-1]["virtual_epoch"] if history else 0,
                "global_step": global_step,
            }
        )
        atomic_write_json(paths["run_manifest"], run_manifest)
        log(f"FAILED {experiment_id}: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

