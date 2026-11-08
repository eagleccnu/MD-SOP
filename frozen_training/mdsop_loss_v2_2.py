#!/usr/bin/env python3
"""Mask-safe, protein-record-macro SmoothL1 losses for MD-SOP v2.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


class NoSupervisionError(ValueError):
    """Raised when no record contains an eligible supervised residue."""


class LossAccessError(PermissionError):
    """Raised when validation/test roles are presented to the training loss."""


@dataclass
class RecordMacroLoss:
    loss: Tensor
    record_losses: Tensor
    record_indices: Tensor
    valid_counts: Tensor


@dataclass
class MixedSourceLoss:
    loss: Tensor
    md_loss: RecordMacroLoss
    soppe_loss: RecordMacroLoss


def _validate_shapes(prediction: Tensor, target: Tensor, mask: Tensor) -> None:
    if prediction.ndim != 2 or target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, target and mask must all have shape [B,L]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have bool dtype")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point")


def record_macro_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    beta: float = 0.05,
) -> RecordMacroLoss:
    """Mean residues within each record, then mean non-empty records.

    Boolean indexing occurs before the loss is evaluated; invalid NaN labels
    therefore never participate in arithmetic.
    """

    _validate_shapes(prediction, target, mask)
    if beta <= 0:
        raise ValueError("beta must be positive")
    losses = []
    indices = []
    counts = []
    for record_index in range(prediction.shape[0]):
        valid = mask[record_index]
        count = int(valid.sum().item())
        if count == 0:
            continue
        selected_prediction = prediction[record_index][valid]
        selected_target = target[record_index][valid]
        if not torch.isfinite(selected_prediction).all() or not torch.isfinite(selected_target).all():
            raise ValueError(f"Record {record_index} has non-finite values at an eligible position")
        losses.append(
            F.smooth_l1_loss(
                selected_prediction,
                selected_target,
                reduction="mean",
                beta=beta,
            )
        )
        indices.append(record_index)
        counts.append(count)
    if not losses:
        raise NoSupervisionError("No record has an eligible supervised residue")
    record_losses = torch.stack(losses)
    return RecordMacroLoss(
        loss=record_losses.mean(),
        record_losses=record_losses,
        record_indices=torch.tensor(indices, device=prediction.device, dtype=torch.long),
        valid_counts=torch.tensor(counts, device=prediction.device, dtype=torch.long),
    )


def mixed_source_record_macro_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    sources: Sequence[str],
    *,
    beta: float = 0.05,
) -> MixedSourceLoss:
    """Return 0.5*MD-record-macro + 0.5*S-OPPE-record-macro."""

    _validate_shapes(prediction, target, mask)
    if len(sources) != prediction.shape[0]:
        raise ValueError("sources must contain one entry per record")
    md_indices = [i for i, source in enumerate(sources) if source == "md_ired"]
    soppe_indices = [i for i, source in enumerate(sources) if source == "soppe_legacy"]
    unknown = sorted(set(sources) - {"md_ired", "soppe_legacy"})
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}")
    if not md_indices or not soppe_indices:
        raise NoSupervisionError("Mixed loss requires at least one record from each source")
    md = record_macro_smooth_l1(
        prediction[md_indices], target[md_indices], mask[md_indices], beta=beta
    )
    soppe = record_macro_smooth_l1(
        prediction[soppe_indices], target[soppe_indices], mask[soppe_indices], beta=beta
    )
    return MixedSourceLoss(loss=0.5 * md.loss + 0.5 * soppe.loss, md_loss=md, soppe_loss=soppe)


def training_loss_from_batch(
    prediction: Tensor,
    batch: Mapping[str, object],
    *,
    mixed_sources: bool,
    beta: float = 0.05,
) -> Tensor:
    roles = list(batch["roles"])  # type: ignore[arg-type]
    if any(role != "train" for role in roles):
        raise LossAccessError("Only train-role records may enter the optimization loss")
    target = batch["y"]
    mask = batch["loss_mask"]
    if not isinstance(target, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("Batch y and loss_mask must be tensors")
    if mixed_sources:
        sources = list(batch["sources"])  # type: ignore[arg-type]
        return mixed_source_record_macro_smooth_l1(
            prediction, target, mask, sources, beta=beta
        ).loss
    return record_macro_smooth_l1(prediction, target, mask, beta=beta).loss


@torch.no_grad()
def validation_loss_from_batch(
    prediction: Tensor,
    batch: Mapping[str, object],
    *,
    beta: float = 0.05,
) -> Tensor:
    roles = list(batch["roles"])  # type: ignore[arg-type]
    if any(role != "validation" for role in roles):
        raise LossAccessError("Validation loss accepts validation-role records only")
    target = batch["y"]
    mask = batch["eval_mask"]
    if not isinstance(target, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("Batch y and eval_mask must be tensors")
    return record_macro_smooth_l1(prediction, target, mask, beta=beta).loss

