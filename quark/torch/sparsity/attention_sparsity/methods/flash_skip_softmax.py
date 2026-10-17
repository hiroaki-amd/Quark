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


def _block_max_and_validity(
    scores: torch.Tensor,
    block_size: int,
    query_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared, threshold-independent setup for both the single- and multi-threshold skip
    computation: pad to a whole number of blocks/tiles, reduce to per-block maxima, and work
    out which (query-tile, key-block) pairs are real (not fully masked/padded) decisions.

    :return: ``(block_max, block_valid, tile_valid)``, where ``block_max``/``block_valid`` have
        shape ``[batch, heads, num_q_blocks, query_block_size, num_kv_blocks]`` and ``tile_valid``
        has shape ``[batch, heads, num_q_blocks, num_kv_blocks]``.
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

    # A tile with no valid row (fully masked/padded) is not a real skip decision.
    tile_valid = block_valid.any(dim=-2)  # [batch, heads, num_q_blocks, num_kv_blocks]

    return block_max, block_valid, tile_valid


def _running_max_prev_and_predecessor(block_max: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Causal running max over key blocks seen *before* each block, and whether such a predecessor
    exists at all.

    A block is only ever skipped when its own max is already ``<= running_max`` (skip requires
    ``block_max - running_max < log(threshold) <= 0``), so freezing ``running_max`` on a skipped
    block is a no-op: the true running max is exactly an exclusive ``cummax`` of ``block_max``
    over the key-block axis, independent of any threshold. This replaces what used to be an
    explicit sequential Python loop over key blocks (mirroring a real kernel's loop) with a
    single vectorized op.
    """
    inclusive = torch.cummax(block_max, dim=-1).values
    running_max_prev = torch.cat(
        [torch.full_like(inclusive[..., :1], float("-inf")), inclusive[..., :-1]], dim=-1
    )
    has_predecessor = torch.isfinite(running_max_prev)
    return running_max_prev, has_predecessor


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
    block_max, block_valid, tile_valid = _block_max_and_validity(scores, block_size, query_block_size)
    running_max_prev, has_predecessor = _running_max_prev_and_predecessor(block_max)

    log_threshold = float(torch.log(torch.tensor(threshold)))

    row_skip_ok = ((block_max - running_max_prev) < log_threshold) & has_predecessor
    # A masked-out row carries no skip requirement, so it must not block the rest of the tile.
    row_skip_ok_effective = row_skip_ok | ~block_valid
    skip_mask = row_skip_ok_effective.all(dim=-2) & tile_valid  # all over query_block_size axis

    num_valid = tile_valid.sum()
    if num_valid == 0:
        sparsity = 0.0
    else:
        sparsity = float((skip_mask & tile_valid).sum() / num_valid)

    return skip_mask, sparsity


def compute_skip_sparsity_multi_threshold(
    scores: torch.Tensor,
    thresholds: list[float],
    block_size: int,
    query_block_size: int = 1,
) -> list[float]:
    """
    Vectorized multi-threshold counterpart to :func:`compute_skip_mask_and_sparsity`: computes
    the observed sparsity for every threshold in ``thresholds`` from a single pass over
    ``scores``, instead of looping this module's single-threshold function once per threshold
    (each of which redundantly recomputes the shared, threshold-independent setup). Does not
    return a skip mask -- calibration only ever needs the sparsity scalar.

    :param torch.Tensor scores: Same as :func:`compute_skip_mask_and_sparsity`.
    :param list[float] thresholds: Threshold values to evaluate, all against the same ``scores``.
    :param int block_size: Same as :func:`compute_skip_mask_and_sparsity`.
    :param int query_block_size: Same as :func:`compute_skip_mask_and_sparsity`.
    :return: One sparsity value per entry in ``thresholds``, same order.
    :rtype: list[float]
    """
    block_max, block_valid, tile_valid = _block_max_and_validity(scores, block_size, query_block_size)
    running_max_prev, has_predecessor = _running_max_prev_and_predecessor(block_max)

    num_valid = tile_valid.sum()
    if num_valid == 0:
        return [0.0] * len(thresholds)

    log_thresholds = torch.log(torch.tensor(thresholds, dtype=block_max.dtype, device=block_max.device))

    diff = (block_max - running_max_prev).unsqueeze(-1)  # [..., query_block_size, num_kv_blocks, 1]
    row_skip_ok = (diff < log_thresholds) & has_predecessor.unsqueeze(-1)
    row_skip_ok_effective = row_skip_ok | (~block_valid).unsqueeze(-1)
    # all() over the query_block_size axis, now third-from-last since the threshold axis was
    # appended at the end.
    tile_skip = row_skip_ok_effective.all(dim=-3) & tile_valid.unsqueeze(-1)  # [..., num_kv_blocks, T]

    sparsities = tile_skip.sum(dim=tuple(range(tile_skip.ndim - 1))).float() / num_valid
    return sparsities.tolist()


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

    def calculate_sparsities(self, scores: torch.Tensor, thresholds: list[float]) -> list[float]:
        """Compute the observed sparsity for ``scores`` at every threshold in ``thresholds``."""
        return compute_skip_sparsity_multi_threshold(scores, thresholds, self.block_size, self.query_block_size)
