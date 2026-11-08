#!/usr/bin/env python3
"""Frozen MD-SOP v2.2 BiLSTM backbone with window-1 or window-15 head."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


CHECKPOINT_FORMAT = "md_sop_bilstm_window_v2_2"
EXPECTED_PARAMETER_COUNTS = {1: 2_587_069, 15: 3_504_573}


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 1182
    projection_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    projection_dropout: float = 0.10
    recurrent_dropout: float = 0.20
    head_dim: int = 128
    head_dropout: float = 0.20
    window_size: int = 1

    def validate(self) -> None:
        if self.input_dim != 1182:
            raise ValueError("MD-SOP v2.2 input_dim is frozen at 1182")
        if self.projection_dim != 128 or self.hidden_size != 256 or self.num_layers != 2:
            raise ValueError("MD-SOP v2.2 projection and recurrent dimensions are frozen")
        if self.head_dim != 128 or self.window_size not in (1, 15):
            raise ValueError("MD-SOP v2.2 supports only 128-unit window-1/window-15 heads")
        if not (0.0 <= self.projection_dropout < 1.0 and 0.0 <= self.head_dropout < 1.0):
            raise ValueError("Dropout probabilities must be in [0,1)")


class MDSoPBiLSTMRegressor(nn.Module):
    """Run a packed whole-chain BiLSTM once, then apply one local regression head."""

    def __init__(self, config: Optional[ModelConfig] = None, **overrides: object) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("Pass either ModelConfig or keyword overrides, not both")
        self.config = config or ModelConfig(**overrides)
        self.config.validate()

        self.input_norm = nn.LayerNorm(self.config.input_dim)
        self.input_projection = nn.Linear(self.config.input_dim, self.config.projection_dim)
        self.input_activation = nn.GELU()
        self.input_dropout = nn.Dropout(self.config.projection_dropout)
        self.bilstm = nn.LSTM(
            input_size=self.config.projection_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.recurrent_dropout,
        )

        recurrent_dim = 2 * self.config.hidden_size
        if self.config.window_size == 1:
            self.window_projection: nn.Module = nn.Linear(recurrent_dim, self.config.head_dim)
            self.output_layer: nn.Module = nn.Linear(self.config.head_dim, 1)
        else:
            self.window_projection = nn.Conv1d(
                recurrent_dim,
                self.config.head_dim,
                kernel_size=self.config.window_size,
                padding=self.config.window_size // 2,
            )
            self.output_layer = nn.Conv1d(self.config.head_dim, 1, kernel_size=1)
        self.head_activation = nn.GELU()
        self.head_dropout = nn.Dropout(self.config.head_dropout)

    @property
    def window_size(self) -> int:
        return self.config.window_size

    def expected_residue_mask(self, lengths: Tensor, max_length: int, device: torch.device) -> Tensor:
        return torch.arange(max_length, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)

    def encode(self, x: Tensor, lengths: Tensor, residue_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError(f"x must have shape [B,L,{self.config.input_dim}]")
        if lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
            raise ValueError("lengths must have shape [B]")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError("lengths must be an integer tensor")
        if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
            raise ValueError("Each length must fall in [1,L]")
        expected_mask = self.expected_residue_mask(lengths, x.shape[1], x.device)
        if residue_mask is None:
            residue_mask = expected_mask
        elif residue_mask.dtype != torch.bool or residue_mask.shape != expected_mask.shape:
            raise ValueError("residue_mask must be bool [B,L]")
        elif not torch.equal(residue_mask.to(x.device), expected_mask):
            raise ValueError("residue_mask must represent every real residue and only padding")

        projected = self.input_dropout(self.input_activation(self.input_projection(self.input_norm(x))))
        packed = pack_padded_sequence(
            projected,
            lengths.detach().to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.bilstm(packed)
        hidden, _ = pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=x.shape[1],
        )
        hidden = hidden.masked_fill(~expected_mask.unsqueeze(-1), 0.0)
        return hidden, expected_mask

    def forward_from_hidden(self, hidden: Tensor, residue_mask: Tensor) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != 2 * self.config.hidden_size:
            raise ValueError("hidden must have shape [B,L,512]")
        if residue_mask.dtype != torch.bool or residue_mask.shape != hidden.shape[:2]:
            raise ValueError("residue_mask must be bool [B,L]")
        hidden = hidden.masked_fill(~residue_mask.unsqueeze(-1), 0.0)
        if self.config.window_size == 1:
            projected = self.window_projection(hidden)
            output = self.output_layer(self.head_dropout(self.head_activation(projected))).squeeze(-1)
        else:
            channels_first = hidden.transpose(1, 2)
            projected = self.window_projection(channels_first)
            output = self.output_layer(self.head_dropout(self.head_activation(projected))).squeeze(1)
        return output.masked_fill(~residue_mask, 0.0)

    def forward(self, x: Tensor, lengths: Tensor, residue_mask: Optional[Tensor] = None) -> Tensor:
        hidden, real_residue_mask = self.encode(x, lengths, residue_mask)
        return self.forward_from_hidden(hidden, real_residue_mask)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def backbone_state_dict(self) -> Dict[str, Tensor]:
        prefixes = ("input_norm.", "input_projection.", "bilstm.")
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }


def state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Hash tensor names, schemas and raw CPU bytes in deterministic key order."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def save_model_checkpoint(
    path: Path,
    model: MDSoPBiLSTMRegressor,
    *,
    seed: int,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "model_state_sha256": state_dict_sha256(model.state_dict()),
        "backbone_state_sha256": state_dict_sha256(model.backbone_state_dict()),
        "seed": int(seed),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)
    return payload


def load_model_checkpoint(
    path: Path,
    *,
    expected_window_size: Optional[int] = None,
    map_location: str = "cpu",
) -> Tuple[MDSoPBiLSTMRegressor, Dict[str, object]]:
    path = Path(path)
    legacy_initializations = {
        "/data/jiangbin/SOPPCL/MD-SOP/model/v2_1_20261007/seed_20261007/initialization/initial_w1_seed_20261007.pt": "535147bfea0e8f9d60de4d63dd4ecbf66d7ccc2daf6edefd4b9fa7400b39e34e",
        "/data/jiangbin/SOPPCL/MD-SOP/model/v2_1_20261007/seed_20261007/initialization/initial_w15_seed_20261007.pt": "251ecf916fae4c06d7cdd90ef546c01c9fa07daacb871b349f3cf9b2b0b3ab58",
    }
    expected_legacy_hash = legacy_initializations.get(str(path.resolve()))
    if expected_legacy_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_legacy_hash:
        raise ValueError("Frozen untrained initialization file changed")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    legacy_ok = (isinstance(payload, dict) and expected_legacy_hash is not None and payload.get("format") == "md_sop_bilstm_window_v2_1"
                 and payload.get("metadata", {}).get("purpose") == "untrained paired initialization"
                 and payload.get("seed") == 20261007)
    if not isinstance(payload, dict) or (payload.get("format") != CHECKPOINT_FORMAT and not legacy_ok):
        raise ValueError("Checkpoint format is not MD-SOP BiLSTM window v2.2")
    config = ModelConfig(**payload["model_config"])
    config.validate()
    if expected_window_size is not None and config.window_size != expected_window_size:
        raise ValueError(
            f"Checkpoint window_size={config.window_size}, expected {expected_window_size}"
        )
    model = MDSoPBiLSTMRegressor(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    observed_state_hash = state_dict_sha256(model.state_dict())
    if observed_state_hash != payload.get("model_state_sha256"):
        raise ValueError("Checkpoint state hash does not match the loaded tensors")
    return model, payload


def build_paired_initializations(output_dir: Path, seed: int) -> Dict[str, object]:
    """Create one reusable initialization per window with an identical backbone."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models: Dict[int, MDSoPBiLSTMRegressor] = {}
    for window_size in (1, 15):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        models[window_size] = MDSoPBiLSTMRegressor(window_size=window_size)
        observed_count = models[window_size].parameter_count()
        if observed_count != EXPECTED_PARAMETER_COUNTS[window_size]:
            raise RuntimeError(
                f"window={window_size}: parameter count {observed_count} != "
                f"{EXPECTED_PARAMETER_COUNTS[window_size]}"
            )

    backbone_hashes = {
        window: state_dict_sha256(model.backbone_state_dict()) for window, model in models.items()
    }
    if len(set(backbone_hashes.values())) != 1:
        raise RuntimeError("Paired window models do not share an identical initialized backbone")

    checkpoints: Dict[str, Dict[str, object]] = {}
    for window_size, model in models.items():
        path = output_dir / f"initial_w{window_size}_seed_{seed}.pt"
        payload = save_model_checkpoint(
            path,
            model,
            seed=seed,
            metadata={
                "purpose": "untrained paired initialization",
                "source_routes": ["A_MD", "B_SOPPE", "C_MIX"],
                "window_size": window_size,
            },
        )
        checkpoints[f"w{window_size}"] = {
            "path": str(path),
            "parameter_count": model.parameter_count(),
            "model_state_sha256": payload["model_state_sha256"],
            "backbone_state_sha256": payload["backbone_state_sha256"],
        }

    return {
        "format": "md_sop_paired_initialization_manifest_v2_2",
        "seed": int(seed),
        "common_backbone_state_sha256": next(iter(backbone_hashes.values())),
        "checkpoints": checkpoints,
        "route_mapping": {
            "A_MD_w1": "w1",
            "B_SOPPE_w1": "w1",
            "C_MIX_w1": "w1",
            "A_MD_w15": "w15",
            "B_SOPPE_w15": "w15",
            "C_MIX_w15": "w15",
        },
    }

