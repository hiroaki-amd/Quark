#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Pure-PyTorch reference implementation of BLASST (skip-softmax) sparsity.

A block-wise skip decision computed directly on the pre-softmax attention
score tensor, with no capture/replay of Q, K, or V. This is a measurement
reference, not a production kernel -- it exists to compute exact sparsity
statistics for a given threshold, not to save compute.

The skip decision follows the FlashAttention-style running-max argument: for
a query row, a key block only meaningfully contributes to the softmax
normalizer if its own max score is within ``log(threshold)`` of the running
max established by key blocks seen so far (causal, left-to-right). A block
that falls short of that bar is treated as skippable -- masked to ``-inf``
before the real softmax is applied, so the returned output is numerically
consistent with what a real skip-softmax kernel would produce.

The running max is computed as an explicit sequential recurrence over key
blocks (mirroring a real kernel's own loop) rather than a one-shot cumulative
max over the raw per-block scores, for clarity of correspondence. This does
not change the result: a block that is fully (unanimously) skipped for a
tile provably cannot have raised the running max anyway (unanimous skip
requires every row's block max to already be below it), so freezing vs.
unconditionally advancing are mathematically identical here -- verified by a
direct before/after diagnostic. Positions the caller has masked out (e.g.
causal, padding) are expected to be far below any real score after scaling;
this module treats anything below a large negative magnitude threshold as
masked, since HF-style attention masks commonly use a large *finite*
sentinel (``torch.finfo(dtype).min``) rather than literal ``-inf`` -- a
previous version of this function checked ``torch.isfinite`` instead, which
missed that sentinel and roughly doubled reported sparsity end to end (see
``memory/tokenspeed_calibration_current_goal.md`` in the ``blasst-calib``
workspace for the diagnostic that found this).

A real tiled kernel (e.g. FlashAttention-style ``BLOCK_M`` x ``BLOCK_N``
tiling, which is what TokenSpeed's GFX950 kernel uses) does not make this
decision independently per query row: a
key block (``BLOCK_N``/``block_size``) is only skippable for an entire query
tile (``BLOCK_M``/``query_block_size``) if *every* row within that tile agrees
it is skippable. If even one row in the tile needs the block, the whole tile
must compute it. ``query_block_size`` defaults to ``1`` (a tile of one row,
equivalent to the old per-row behavior) for callers that don't care about
matching a specific kernel's tiling; pass the kernel's actual query tile size
(e.g. ``128`` for TokenSpeed's GFX950 prefill kernel) to get a calibration
that matches the deployed kernel's real skip granularity.
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
    # Masked-out positions (causal future, or this function's own padding) must be told apart from
    # real scores. This function's own padding uses literal -inf (see the `pad` call above), but a
    # caller's causal/attention mask is not guaranteed to: HF's eager attention adds
    # `torch.finfo(dtype).min` (a large but *finite* sentinel, e.g. ~-3.39e38 for bf16) rather than
    # actual -inf, specifically to avoid NaN from softmax-ing an all-(-inf) row. `torch.isfinite`
    # does not see through that sentinel -- it would call those positions "valid", inflating both
    # this block-tile's denominator and its skip count with causally-nonexistent positions, which
    # was measured to roughly double reported sparsity end to end (see
    # `memory/tokenspeed_calibration_current_goal.md` in the `blasst-calib` workspace). A magnitude
    # threshold catches both conventions: no real (unmasked) attention score after scaling is ever
    # anywhere close to this negative, in fp32, bf16, or fp16.
    block_valid = block_max > -1e30

    log_threshold = float(torch.log(torch.tensor(threshold)))

    # A key block is skippable for the whole query tile only if every row in the tile agrees --
    # matches TokenSpeed's "skip only if all rows in BLOCK_M agree" tiled kernel semantics.
    # A tile with no valid row at all (fully masked/padded) is not a real skip decision, so it is
    # forced to False rather than trivially True.
    tile_valid = block_valid.any(dim=-2)  # [batch, heads, num_q_blocks, num_kv_blocks]

    # Written as an explicit sequential recurrence over kv blocks (mirroring the real kernel's own
    # loop, and `sparsity_reference.py`'s independent reference implementation) rather than a
    # vectorized `torch.cummax` over the raw per-block max, for clarity of correspondence to the
    # kernel. Note this is *not* a behavioral fix by itself: freezing `running_max` on a fully
    # (unanimously) skipped tile is provably a no-op, because unanimous-skip requires every row's
    # `block_max` to already be below `running_max` by construction, so `max(running_max,
    # block_max) == running_max` whether or not the update is applied. A plain unconditional
    # `torch.cummax` is therefore exactly equivalent to this loop, not an approximation of it --
    # confirmed by rerunning the same-sample diagnostic before and after switching to this form and
    # getting bit-identical sparsity. The actual overestimate this module had was the masked-value
    # sentinel bug fixed just above (`block_valid`).
    running_max = torch.full_like(block_max[..., 0], float("-inf"))  # [batch, heads, num_q_blocks, query_block_size]
    skip_mask_steps = []
    for b in range(num_kv_blocks):
        block_max_b = block_max[..., b]
        valid_b = block_valid[..., b]

        has_predecessor = torch.isfinite(running_max)
        row_skippable = (block_max_b - running_max) < log_threshold
        row_skip_ok = row_skippable & has_predecessor
        # A row that is itself masked out (causal/padding) carries no real skip requirement, so
        # it must not prevent the rest of the tile from skipping.
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


def apply_skip_mask(
    scores: torch.Tensor,
    skip_mask: torch.Tensor,
    block_size: int,
    query_block_size: int = 1,
) -> torch.Tensor:
    """
    Mask out skipped key blocks in ``scores`` (set to ``-inf``) ahead of a real softmax call.

    :param torch.Tensor scores: Pre-softmax attention scores, shape ``[batch, heads, q_len,
        kv_len]``.
    :param torch.Tensor skip_mask: Boolean mask, shape ``[batch, heads, num_q_blocks,
        num_kv_blocks]``, as returned by :func:`compute_skip_mask_and_sparsity`.
    :param int block_size: Same key-dimension block size used to compute ``skip_mask``.
    :param int query_block_size: Same query-dimension tile size used to compute ``skip_mask``.
    :return: A copy of ``scores`` with skipped-block positions set to ``-inf``.
    :rtype: torch.Tensor
    """
    q_len = scores.shape[-2]
    kv_len = scores.shape[-1]

    expanded_mask = skip_mask.repeat_interleave(query_block_size, dim=-2).repeat_interleave(block_size, dim=-1)
    expanded_mask = expanded_mask[..., :q_len, :kv_len]

    return scores.masked_fill(expanded_mask, float("-inf"))


class FlashSkipSoftmax:
    """
    Reference (calibration-time) skip-softmax method.

    Stateless helper wrapping :func:`compute_skip_mask_and_sparsity` and
    :func:`apply_skip_mask`. Operates purely on an already-computed pre-softmax score tensor --
    the same tensor a caller would otherwise pass straight to ``torch.nn.functional.softmax`` --
    with no dependency on how ``scores`` was produced (no Q/K/V access needed).

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

    def apply_sparsity(self, scores: torch.Tensor, skip_mask: torch.Tensor) -> torch.Tensor:
        """Mask out the skipped blocks in ``scores`` ahead of a real softmax call."""
        return apply_skip_mask(scores, skip_mask, self.block_size, self.query_block_size)
