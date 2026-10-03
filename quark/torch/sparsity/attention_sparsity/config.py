#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Attention Sparsity (BLASST / Skip Softmax Attention) Config API for PyTorch.

The ``sparse_attention_config`` checkpoint schema uses named ``config_groups``,
each with an ``algorithm`` name, named threshold-scale-factor coefficients per
phase, and an fnmatch ``targets``/``ignore`` pair to select layers. There is no
GPU-architecture axis: the threshold-to-sparsity relationship is treated as a
property of the model and phase (prefill/decode) only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quark import __version__
from quark.common.config import BaseAlgoConfig, BaseConfigImpl

# Default sweep of skip-softmax thresholds used during calibration to collect
# (scale_factor, sparsity) data points for curve fitting. Not part of the
# exported checkpoint schema -- purely a calibration-time hyperparameter.
DEFAULT_THRESHOLD_TRIALS: list[float] = [
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    5e-3,
    1e-2,
    2e-2,
    3e-2,
    5e-2,
    7e-2,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.75,
    0.9,
    0.99,
]

_SKIP_SOFTMAX_FORMULA = "a * exp(b * target_sparsity)"


@dataclass(eq=True)
class ThresholdScaleFactor(BaseConfigImpl):
    """
    Fitted coefficients for one phase (prefill/decode) of the skip-softmax
    threshold-scale-factor formula ``scale_factor = a * exp(b * target_sparsity)``.

    :param float a: Fitted multiplicative coefficient.
    :param float b: Fitted exponential-rate coefficient.
    """

    a: float
    b: float


@dataclass(eq=True)
class SparseAttentionAttributeConfig(BaseConfigImpl):
    """
    One ``config_groups`` entry: the skip-softmax calibration result for a set of
    attention layers matched by ``targets``/``ignore``.

    :param str algorithm: Sparse attention algorithm name. Only ``"skip_softmax"`` (BLASST) is
        currently supported.
    :param list[str] targets: Fnmatch patterns (or class names) identifying which attention
        layers this group applies to.
    :param dict[str, ThresholdScaleFactor] threshold_scale_factor: Per-phase (``"prefill"``,
        ``"decode"``) fitted coefficients. A phase absent from this dict has not been
        calibrated.
    :param dict[str, float] target_sparsity: Per-phase target sparsity used to derive the
        runtime threshold (``threshold = scale_factor / seq_len``).
    :param list[str] ignore: Fnmatch patterns for layers excluded from this group (dense
        fallback).
    """

    algorithm: str = "skip_softmax"
    targets: list[str] = field(default_factory=list)
    threshold_scale_factor: dict[str, ThresholdScaleFactor] = field(default_factory=dict)
    target_sparsity: dict[str, float] = field(default_factory=dict)
    ignore: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "targets": list(self.targets),
            "threshold_scale_factor": {
                "formula": _SKIP_SOFTMAX_FORMULA,
                **{phase: {"a": coeffs.a, "b": coeffs.b} for phase, coeffs in self.threshold_scale_factor.items()},
            },
            "target_sparsity": dict(self.target_sparsity),
            "ignore": list(self.ignore),
        }


@dataclass
class CalibrationConfig(BaseAlgoConfig):
    """
    Configuration for the BLASST/skip-softmax dynamic-threshold calibration algorithm.

    :param str name: Algorithm name.
    :param dict[str, float] target_sparse_ratio: Per-phase target sparsity to calibrate for.
        A phase whose ratio is ``0.0`` is skipped (left dense).
    :param int samples: Number of calibration samples (default mirrors "1 per RULER task per
        length bin").
    :param int max_seqlen: Maximum sequence length used during calibration.
    :param int num_length_bins: Number of sequence-length bins sampled during calibration.
    :param int chunk_size: Chunked-prefill size to bound memory use; ``-1`` disables chunking.
    :param int num_decode_tokens: Number of manual decode steps measured during decode-phase
        calibration.
    :param list[float] threshold_trials: Threshold values swept in a single calibration forward
        pass to collect ``(scale_factor, sparsity)`` data points.
    :param int block_size: Key-dimension block size used for the skip-decision granularity
        (analogous to a FlashAttention ``BLOCK_N``).
    :param int query_block_size: Query-dimension tile size (analogous to a FlashAttention
        ``BLOCK_M``). A key block is only skippable for a query tile if every row within that
        tile agrees, matching the all-or-nothing skip granularity of a real tiled kernel.
        Defaults to ``1`` (per-row decisions, no tiling). To calibrate against a specific
        deployed kernel's actual skip behavior (e.g. TokenSpeed's GFX950 prefill kernel, which
        uses ``BLOCK_M=128``/``BLOCK_N=64``), set this to that kernel's real query tile size and
        ``block_size`` to its key block size.
    :param bool fit_logspace: If ``True``, fit ``log(scale_factor) = log(a) + b * sparsity``
        (minimizes relative error; useful when ``scale_factor`` spans many orders of
        magnitude). If ``False`` (default), fit in linear space (minimizes absolute error).
    :param str | None cache_dir: Optional cache directory for calibration model artifacts.
    :param str | None data_dir: Optional cache directory for calibration sample data.
    """

    name: str = "skip_softmax_calibration"
    target_sparse_ratio: dict[str, float] = field(default_factory=lambda: {"prefill": 0.5, "decode": 0.5})
    samples: int = 24
    max_seqlen: int = 8192
    num_length_bins: int = 8
    chunk_size: int = -1
    num_decode_tokens: int = 8
    threshold_trials: list[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLD_TRIALS))
    block_size: int = 64
    query_block_size: int = 1
    fit_logspace: bool = False
    cache_dir: str | None = None
    data_dir: str | None = None


@dataclass
class SparseAttentionConfig(BaseConfigImpl):
    """
    Top-level attention-sparsity configuration.

    :param dict[str, SparseAttentionAttributeConfig] sparse_cfg: Named ``config_groups`` entries
        (e.g. ``"group_0"``), each covering a set of attention layers.
    :param CalibrationConfig | None calibration_config: Calibration hyperparameters. ``None``
        means no calibration is to be run (e.g. when loading an already-calibrated checkpoint).
    """

    sparse_cfg: dict[str, SparseAttentionAttributeConfig] = field(default_factory=dict)
    calibration_config: CalibrationConfig | None = None


def build_sparse_attention_config_dict(sparse_cfg: dict[str, SparseAttentionAttributeConfig]) -> dict[str, Any]:
    """
    Build the ``sparse_attention_config`` dict written to a checkpoint's ``config.json``.

    Pure data-shaping: takes already-fitted config groups and produces the
    Model-Optimizer-compatible JSON structure. Does not touch a model or run any calibration.

    :param dict[str, SparseAttentionAttributeConfig] sparse_cfg: Named config groups to export.
    :return: A dict suitable for ``model.config.update({"sparse_attention_config": ...})``.
    :rtype: dict[str, Any]
    """
    return {
        "config_groups": {name: group.to_dict() for name, group in sparse_cfg.items()},
        "producer": {"name": "quark", "version": __version__},
    }
