#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from quark.torch.sparsity.attention_sparsity.methods import (
    FlashSkipSoftmax,
    compute_skip_mask_and_sparsity,
)


def _make_scores(block_values: list[float], block_size: int = 2) -> torch.Tensor:
    """Build a [1, 1, 1, num_blocks * block_size] score row where every position within
    block i holds the constant value block_values[i]."""
    row = torch.tensor(block_values, dtype=torch.float32).repeat_interleave(block_size)
    return row.view(1, 1, 1, -1)


def test_first_block_is_never_skipped():
    # Even an extremely low first block must be kept -- it has no predecessor to compare against.
    scores = _make_scores([-100.0, 0.0])
    skip_mask, sparsity = compute_skip_mask_and_sparsity(scores, threshold=0.5, block_size=2)
    assert skip_mask[0, 0, 0, 0].item() is False


def test_low_relative_block_is_skipped():
    # threshold=0.01 -> log(threshold) ~= -4.6. Block 1 (value=0) is far below block 0's max (10),
    # exp(0 - 10) ~= 4.5e-5 << 0.01, so it should be skipped.
    scores = _make_scores([10.0, 0.0])
    skip_mask, sparsity = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2)
    assert skip_mask[0, 0, 0, 1].item() is True
    assert sparsity == 0.5  # 1 skipped out of 2 valid blocks (block 0 is valid, just never skipped)


def test_close_block_is_not_skipped():
    # Block 1 (value=9.9) is very close to block 0's max (10): exp(9.9-10) ~= 0.90, well above
    # threshold=0.01, so it must NOT be skipped.
    scores = _make_scores([10.0, 9.9])
    skip_mask, sparsity = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2)
    assert skip_mask[0, 0, 0, 1].item() is False
    assert sparsity == 0.0


def test_threshold_boundary_is_monotonic():
    # A tighter (larger) threshold should never result in less sparsity than a looser one.
    scores = _make_scores([10.0, 5.0, 8.0, 2.0], block_size=2)
    _, sparsity_loose = compute_skip_mask_and_sparsity(scores, threshold=1e-6, block_size=2)
    _, sparsity_tight = compute_skip_mask_and_sparsity(scores, threshold=0.5, block_size=2)
    assert sparsity_tight >= sparsity_loose


def test_fully_masked_blocks_excluded_from_denominator():
    # Causal-style mask: the tail of the sequence is entirely -inf (e.g. padding). Those blocks
    # must not count as "valid" and must not inflate the sparsity denominator.
    scores = _make_scores([10.0, 0.0, float("-inf"), float("-inf")], block_size=2)
    skip_mask, sparsity = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2)
    assert skip_mask[0, 0, 0, 2].item() is False
    assert skip_mask[0, 0, 0, 3].item() is False
    assert sparsity == 0.5  # 1 skipped (block 1) out of 2 valid blocks; blocks 2-3 are masked out


def test_kv_len_not_multiple_of_block_size_is_padded_correctly():
    # 5 key positions, block_size=2 -> 3 blocks, last block only has 1 real position.
    scores = torch.tensor([10.0, 10.0, 0.0, 0.0, 0.0], dtype=torch.float32).view(1, 1, 1, -1)
    skip_mask, sparsity = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2)
    assert skip_mask.shape[-1] == 3
    assert skip_mask[0, 0, 0, 0].item() is False  # first block never skipped
    assert skip_mask[0, 0, 0, 1].item() is True  # 0.0 vs running max 10.0 -> skippable
    assert skip_mask[0, 0, 0, 2].item() is True  # padded value is -inf, but still real (0.0) data


def test_flash_skip_softmax_class_matches_module_functions():
    scores = _make_scores([10.0, 0.0], block_size=2)
    method = FlashSkipSoftmax(block_size=2)

    skip_mask, sparsity = method.calculate_sparsity(scores, threshold=0.01)
    expected_skip_mask, expected_sparsity = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2)
    assert torch.equal(skip_mask, expected_skip_mask)
    assert sparsity == expected_sparsity


def test_no_valid_blocks_returns_zero_sparsity():
    scores = torch.full((1, 1, 1, 4), float("-inf"))
    skip_mask, sparsity = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2)
    assert sparsity == 0.0
    assert not skip_mask.any()


def test_batched_multihead_shapes_are_independent():
    # Two (batch, head) slices with opposite skip behavior must not interfere with each other.
    row_a = torch.tensor([10.0, 0.0], dtype=torch.float32).repeat_interleave(2)
    row_b = torch.tensor([10.0, 9.9], dtype=torch.float32).repeat_interleave(2)
    scores = torch.stack([row_a, row_b]).view(2, 1, 1, -1)

    skip_mask, _ = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2)
    assert skip_mask[0, 0, 0, 1].item() is True
    assert skip_mask[1, 0, 0, 1].item() is False


def _make_query_tile_scores(rows: list[list[float]], block_size: int = 2) -> torch.Tensor:
    """Build a [1, 1, len(rows), num_blocks * block_size] score tensor, one row per query
    position, each row's blocks holding the constant values given in ``rows[i]``."""
    expanded = [
        torch.tensor(values, dtype=torch.float32).repeat_interleave(block_size) for values in rows
    ]
    return torch.stack(expanded).view(1, 1, len(rows), -1)


def test_query_tile_requires_all_rows_to_agree_before_skipping():
    # Row 0 alone would skip block 1 (0.0 far below running max 10.0), but row 1 in the same
    # tile still needs it (9.9 is close to 10.0) -- the whole tile must keep the block.
    scores = _make_query_tile_scores([[10.0, 0.0], [10.0, 9.9]], block_size=2)
    skip_mask, sparsity = compute_skip_mask_and_sparsity(
        scores, threshold=0.01, block_size=2, query_block_size=2
    )
    assert skip_mask.shape[-2:] == (1, 2)
    assert skip_mask[0, 0, 0, 1].item() is False
    assert sparsity == 0.0


def test_query_tile_skips_when_every_row_agrees():
    scores = _make_query_tile_scores([[10.0, 0.0], [10.0, 0.1]], block_size=2)
    skip_mask, sparsity = compute_skip_mask_and_sparsity(
        scores, threshold=0.01, block_size=2, query_block_size=2
    )
    assert skip_mask[0, 0, 0, 1].item() is True
    assert sparsity == 0.5  # 1 skipped out of 2 valid tile/block pairs


def test_query_tile_padding_rows_do_not_force_dense():
    # q_len=3 is not a multiple of query_block_size=2, so the last tile is padded with an
    # entirely-masked row. That padding must not prevent the real row in the same tile from
    # skipping.
    scores = _make_query_tile_scores([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]], block_size=2)
    skip_mask, _ = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2, query_block_size=2)
    assert skip_mask.shape[-2] == 2  # ceil(3 / 2) query tiles
    assert skip_mask[0, 0, 1, 1].item() is True  # tile 1 = real row 2 + padding, still skippable


def test_query_block_size_default_matches_per_row_behavior():
    # query_block_size defaults to 1, so tiling must not change any existing per-row result.
    scores = _make_query_tile_scores([[10.0, 0.0], [10.0, 9.9]], block_size=2)
    skip_mask, _ = compute_skip_mask_and_sparsity(scores, threshold=0.01, block_size=2)
    assert skip_mask[0, 0, 0, 1].item() is True
    assert skip_mask[0, 0, 1, 1].item() is False


def test_flash_skip_softmax_class_supports_query_block_size():
    scores = _make_query_tile_scores([[10.0, 0.0], [10.0, 9.9]], block_size=2)
    method = FlashSkipSoftmax(block_size=2, query_block_size=2)

    skip_mask, sparsity = method.calculate_sparsity(scores, threshold=0.01)
    expected_skip_mask, expected_sparsity = compute_skip_mask_and_sparsity(
        scores, threshold=0.01, block_size=2, query_block_size=2
    )
    assert torch.equal(skip_mask, expected_skip_mask)
    assert sparsity == expected_sparsity
    assert skip_mask[0, 0, 0, 1].item() is False  # row 1 in the tile still needs the block
