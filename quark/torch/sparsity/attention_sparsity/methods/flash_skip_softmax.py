#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Pure-PyTorch reference implementation of BLASST (skip-softmax) sparsity.

Computes a block-wise skip decision on a pre-softmax attention score tensor. A key block is
skippable for a query row if ``exp(block_max - running_max) < threshold``, where ``running_max``
is the running max over key blocks seen so far (causal, left-to-right). A key block is only
skipped for a whole query tile if every row in the tile agrees, matching a real tiled kernel's
all-or-nothing skip granularity (``query_block_size``/``block_size`` are the ``BLOCK_M``/
``BLOCK_N`` equivalents).

Masked-out positions must be detected by magnitude, not ``torch.isfinite``: HF-style masks use a
large finite sentinel (``torch.finfo(dtype).min``), not literal ``-inf``.
"""

from __future__ import annotations

import torch


def compute_skip_mask_and_sparsity(
    scores: torch.Tensor,
    threshold: float,
    block_size: int,
    query_block_size: int = 1,
) -> tuple[torch.Tensor, float]:
    """
    Compute the block-skip mask and observed sparsity for one attention score tensor.

    :param torch.Tensor scores: Pre-softmax attention scores, shape ``[batch, heads, q_len,
        kv_len]``. Positions already masked out (e.g. causal, padding) are expected to hold
        ``-inf``.
    :param float threshold: Skip threshold. A key block is skipped for a given query row if
        ``exp(block_max - running_max) < threshold``.
    :param int block_size: Number of key positions per block (the granularity at which skip
        decisions are made, analogous to a FlashAttention ``BLOCK_N``).
    :param int query_block_size: Number of query positions per tile (analogous to a
        FlashAttention ``BLOCK_M``). A key block is only skippable for a query tile if *every*
        row within that tile is individually skippable -- matching the all-or-nothing skip
        granularity of a real tiled kernel. Defaults to ``1`` (per-row decisions, no tiling).
    :return: ``(skip_mask, sparsity)`` where ``skip_mask`` is a boolean tensor of shape
        ``[batch, heads, num_q_blocks, num_kv_blocks]`` (``True`` where the block is skipped for
        the whole query tile) and ``sparsity`` is the fraction of *valid* (non-fully-masked)
        query-tile/key-block pairs that were skipped.
    :rtype: tuple[torch.Tensor, float]
    """
    if scores.ndim != 4:
        raise ValueError(f"scores must be 4D [batch, heads, q_len, kv_len], got shape {tuple(scores.shape)}")

    q_len = scores.shape[-2]
    kv_len = scores.shape[-1]
    num_kv_blocks = (kv_len + block_size - 1) // block_size
    num_q_blocks = (q_len + query_block_size - 1) // query_block_size
    pad_kv = num_kv_blocks * block_size - kv_len
    pad_q = num_q_blocks * query_block_size - q_len
    if pad_kv or pad_q:
        scores = torch.nn.functional.pad(scores, (0, pad_kv, 0, pad_q), value=float("-inf"))

    blocked = scores.reshape(*scores.shape[:-2], num_q_blocks, query_block_size, num_kv_blocks, block_size)
    block_max = blocked.amax(dim=-1)  # [batch, heads, num_q_blocks, query_block_size, num_kv_blocks]
    # HF masks use a large finite sentinel (torch.finfo(dtype).min), not -inf, so detect masked
    # positions by magnitude rather than torch.isfinite (see module docstring).
    block_valid = block_max > -1e30

    log_threshold = float(torch.log(torch.tensor(threshold)))

    # A tile with no valid row (fully masked/padded) is not a real skip decision.
    tile_valid = block_valid.any(dim=-2)  # [batch, heads, num_q_blocks, num_kv_blocks]

    # Explicit recurrence over kv blocks (mirrors a real kernel's loop); mathematically
    # equivalent to a vectorized torch.cummax, since freezing running_max on a fully skipped
    # tile is a no-op (every row's block_max is already below it by construction).
    running_max = torch.full_like(block_max[..., 0], float("-inf"))  # [batch, heads, num_q_blocks, query_block_size]
    skip_mask_steps = []
    for b in range(num_kv_blocks):
        block_max_b = block_max[..., b]
        valid_b = block_valid[..., b]

        has_predecessor = torch.isfinite(running_max)
        row_skippable = (block_max_b - running_max) < log_threshold
        row_skip_ok = row_skippable & has_predecessor
        # A masked-out row carries no skip requirement, so it must not block the rest of the tile.
        row_skip_ok_effective = row_skip_ok | ~valid_b

        tile_skip_b = row_skip_ok_effective.all(dim=-1) & tile_valid[..., b]  # [batch, heads, num_q_blocks]
        skip_mask_steps.append(tile_skip_b)

        advanced = torch.maximum(running_max, block_max_b)
        running_max = torch.where(tile_skip_b.unsqueeze(-1), running_max, advanced)

    skip_mask = torch.stack(skip_mask_steps, dim=-1)  # [batch, heads, num_q_blocks, num_kv_blocks]

    num_valid = tile_valid.sum()
    if num_valid == 0:
        sparsity = 0.0
    else:
        sparsity = float((skip_mask & tile_valid).sum() / num_valid)

    return skip_mask, sparsity


class FlashSkipSoftmax:
    """
    Reference (calibration-time) skip-softmax method.

    Stateless helper wrapping :func:`compute_skip_mask_and_sparsity`. Operates purely on an
    already-computed pre-softmax score tensor -- the same tensor a caller would otherwise pass
    straight to ``torch.nn.functional.softmax`` -- with no dependency on how ``scores`` was
    produced (no Q/K/V access needed).

    :param int block_size: Key-dimension block size used for skip decisions (FlashAttention
        ``BLOCK_N``-equivalent).
    :param int query_block_size: Query-dimension tile size (FlashAttention ``BLOCK_M``-
        equivalent). A key block is only skippable for a tile if every row in the tile agrees.
        Defaults to ``1`` (per-row decisions). Pass the target kernel's real tile size (e.g.
        ``128`` for TokenSpeed's GFX950 prefill kernel) to match its actual skip granularity.
    """

    def __init__(self, block_size: int = 64, query_block_size: int = 1) -> None:
        self.block_size = block_size
        self.query_block_size = query_block_size

    def calculate_sparsity(self, scores: torch.Tensor, threshold: float) -> tuple[torch.Tensor, float]:
        """Compute the skip mask and observed sparsity for ``scores`` at ``threshold``."""
        return compute_skip_mask_and_sparsity(scores, threshold, self.block_size, self.query_block_size)
