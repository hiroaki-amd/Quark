#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Checkpoint export/reload for BLASST (skip-softmax) calibration results.

Splits writing calibration metadata into a checkpoint (``export_sparse_attention_config``)
from resolving it back at load/serving time (``load_sparse_attention_config_from_checkpoint`` /
``resolve_layer_sparse_config``). This module has no dependency on ATOM or any serving stack: it
only writes/reads the ``sparse_attention_config`` block, using the ``config_groups``
schema built by :func:`~quark.torch.sparsity.attention_sparsity.config.build_sparse_attention_config_dict`.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

import torch

from quark.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig, build_sparse_attention_config_dict

__all__ = [
    "export_sparse_attention_config",
    "load_sparse_attention_config_from_checkpoint",
    "resolve_layer_sparse_config",
]


def export_sparse_attention_config(
    model: torch.nn.Module,
    sparse_cfg: dict[str, SparseAttentionAttributeConfig],
) -> None:
    """
    Attach calibrated BLASST config groups to a model's HF config, under
    ``sparse_attention_config``, analogous to how Quark's quantization exporters set
    ``model.config.update({"quantization_config": ...})``.

    This only updates the in-memory ``model.config``; writing it to disk (``config.json``)
    is left to the caller's usual checkpoint-saving path (e.g. ``model.config.save_pretrained``
    or Quark's own safetensors exporter).

    :param torch.nn.Module model: A model exposing a HF-style ``config`` attribute with an
        ``update()`` method.
    :param dict[str, SparseAttentionAttributeConfig] sparse_cfg: Named config groups, as
        returned by :class:`~quark.torch.sparsity.attention_sparsity.api.ModelSparseAttentionCalibrator`.
    """
    model.config.update({"sparse_attention_config": build_sparse_attention_config_dict(sparse_cfg)})


def load_sparse_attention_config_from_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """
    Read the ``sparse_attention_config`` block from a checkpoint's ``config.json``, if present.

    :param str | Path checkpoint_dir: Directory containing ``config.json``.
    :return: The raw ``sparse_attention_config`` dict (``config_groups`` + ``producer``), or
        ``None`` if the checkpoint has no such block.
    :rtype: dict[str, Any] | None
    """
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    result = config.get("sparse_attention_config")
    return result if isinstance(result, dict) else None


def resolve_layer_sparse_config(layer_name: str, sparse_attention_config: dict[str, Any]) -> dict[str, Any] | None:
    """
    Resolve a layer's BLASST config group by matching ``layer_name`` against each config
    group's ``ignore`` and ``targets`` fnmatch patterns.

    Groups are checked in ``config_groups`` insertion order; within a group, a match against
    ``ignore`` takes precedence over ``targets`` (the layer is left dense/unmatched). The first
    group with a ``targets`` match wins.

    :param str layer_name: Fully-qualified layer name (e.g. ``"model.layers.3.self_attn"``).
    :param dict[str, Any] sparse_attention_config: The dict returned by
        :func:`load_sparse_attention_config_from_checkpoint` (or
        :func:`~quark.torch.sparsity.attention_sparsity.config.build_sparse_attention_config_dict`).
    :return: The matched group's raw dict (``algorithm``, ``threshold_scale_factor``,
        ``target_sparsity``, ...), or ``None`` if no group matches (dense fallback).
    :rtype: dict[str, Any] | None
    """
    config_groups = sparse_attention_config.get("config_groups", {})
    for group in config_groups.values():
        if any(fnmatch.fnmatch(layer_name, pattern) for pattern in group.get("ignore", [])):
            continue
        if any(fnmatch.fnmatch(layer_name, pattern) for pattern in group.get("targets", [])):
            return group
    return None
