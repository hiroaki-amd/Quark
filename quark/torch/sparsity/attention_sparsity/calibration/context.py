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
    Monkeypatch ``torch.nn.functional.softmax`` to record ``(threshold, seq_len, sparsity)``
    observations into ``calibrator`` for every 4D score tensor softmax is called on, while
    leaving the wrapped code's real output unchanged.

    :param DynamicThresholdCalibrator calibrator: Accumulator that observations are recorded
        into.
    :param list[float] threshold_trials: Thresholds swept against each observed score tensor in
        a single pass.
    :param int block_size: Key-dimension block size used for the skip-decision granularity.
    :param int query_block_size: Query-dimension tile size; a key block is only skippable for a
        tile if every row within it agrees. See :class:`FlashSkipSoftmax` for details.
    """
    method = FlashSkipSoftmax(block_size=block_size, query_block_size=query_block_size)
    original_softmax = F.softmax

    def sparse_softmax(input: torch.Tensor, dim: int | None = None, *args: object, **kwargs: object) -> torch.Tensor:
        if input.ndim == 4:
            seq_len = input.shape[-1]
            for threshold in threshold_trials:
                _, sparsity = method.calculate_sparsity(input, threshold)
                calibrator.add_observation(threshold=threshold, seq_len=seq_len, sparsity=sparsity)
        return original_softmax(input, dim, *args, **kwargs)

    F.softmax = sparse_softmax
    try:
        yield
    finally:
        F.softmax = original_softmax
