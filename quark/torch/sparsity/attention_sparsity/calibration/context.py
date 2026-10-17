#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""F.softmax monkeypatch used to observe skip-softmax sparsity during a real forward pass.

Temporarily replaces ``torch.nn.functional.softmax`` with a wrapper that
inspects the pre-softmax score tensor passed in by the caller, then forwards
to the original ``softmax`` unchanged. No Q/K/V is captured; the model's
real output is untouched, only observed.

This requires the model to run with ``attn_implementation="eager"``: fused
backends (SDPA, FlashAttention) never call ``F.softmax`` as a distinct step,
so there is nothing for this monkeypatch to intercept.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from quark.torch.sparsity.attention_sparsity.calibration.calibrator import DynamicThresholdCalibrator
from quark.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import FlashSkipSoftmax


@contextlib.contextmanager
def record_softmax_sparsity(
    calibrator: DynamicThresholdCalibrator,
    threshold_trials: list[float],
    block_size: int = 64,
    query_block_size: int = 1,
) -> Iterator[None]:
    """
    Monkeypatch ``torch.nn.functional.softmax`` to observe skip-softmax sparsity for every 4D
    score tensor softmax is called on during this context, while leaving the wrapped code's real
    output unchanged.

    Averages sparsity across every call intercepted within this context (one per attention
    layer, all sharing the same key/value length) and, when the context exits, records a single
    ``(threshold, seq_len, sparsity)`` observation per threshold into ``calibrator`` -- matching
    NVIDIA Model Optimizer's own per-forward-pass (not per-layer) aggregation
    (``DynamicThresholdCalibrator.calibrate``'s ``_extract_calibration_stats``). Per-layer
    sparsity is known to vary substantially layer-to-layer; treating each layer as an independent
    calibration point would pollute the fit with that layer-identity variance instead of the
    threshold/context-length relationship calibration is meant to capture.

    Open one ``with record_softmax_sparsity(...)`` block per forward pass at a single key/value
    length (one prefill chunk, or one decode step) -- never around multiple forward passes at
    different lengths, which would silently average across those different lengths too.

    :param DynamicThresholdCalibrator calibrator: Accumulator the averaged observation(s) are
        recorded into when this context exits.
    :param list[float] threshold_trials: Thresholds swept against every observed score tensor.
    :param int block_size: Key-dimension block size used for the skip-decision granularity.
    :param int query_block_size: Query-dimension tile size; a key block is only skippable for a
        tile if every row within it agrees. See :class:`FlashSkipSoftmax` for details.
    """
    method = FlashSkipSoftmax(block_size=block_size, query_block_size=query_block_size)
    original_softmax = F.softmax
    accumulated: list[list[float]] = [[] for _ in threshold_trials]
    seq_lens_seen: list[int] = []

    def sparse_softmax(input: torch.Tensor, dim: int | None = None, *args: object, **kwargs: object) -> torch.Tensor:
        if input.ndim == 4:
            seq_lens_seen.append(input.shape[-1])
            sparsities = method.calculate_sparsities(input, threshold_trials)
            for values, sparsity in zip(accumulated, sparsities):
                values.append(sparsity)
        return original_softmax(input, dim, *args, **kwargs)

    F.softmax = sparse_softmax
    try:
        yield
    finally:
        F.softmax = original_softmax
        if seq_lens_seen:
            seq_len = seq_lens_seen[0]
            averaged = [sum(values) / len(values) for values in accumulated]
            calibrator.add_observations(thresholds=threshold_trials, seq_len=seq_len, sparsities=averaged)
