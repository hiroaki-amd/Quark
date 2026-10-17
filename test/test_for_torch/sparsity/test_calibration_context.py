#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
import torch.nn.functional as F

from quark.torch.sparsity.attention_sparsity.calibration.calibrator import DynamicThresholdCalibrator
from quark.torch.sparsity.attention_sparsity.calibration.context import record_softmax_sparsity


def test_record_softmax_sparsity_records_observations_for_4d_input():
    calibrator = DynamicThresholdCalibrator()
    scores = torch.randn(1, 2, 8, 32)

    with record_softmax_sparsity(calibrator, threshold_trials=[1e-3, 1e-2, 1e-1]):
        F.softmax(scores, dim=-1)

    assert len(calibrator._scale_factors) == 3
    assert len(calibrator._sparsities) == 3


def test_record_softmax_sparsity_ignores_non_4d_input():
    calibrator = DynamicThresholdCalibrator()
    scores_2d = torch.randn(4, 16)

    with record_softmax_sparsity(calibrator, threshold_trials=[1e-3, 1e-2]):
        F.softmax(scores_2d, dim=-1)

    assert len(calibrator._scale_factors) == 0


def test_record_softmax_sparsity_output_matches_unpatched_softmax():
    scores = torch.randn(1, 2, 8, 32)
    expected = F.softmax(scores, dim=-1)

    calibrator = DynamicThresholdCalibrator()
    with record_softmax_sparsity(calibrator, threshold_trials=[1e-3]):
        actual = F.softmax(scores, dim=-1)

    assert torch.equal(actual, expected)


def test_record_softmax_sparsity_restores_original_softmax_on_exit():
    original = F.softmax
    calibrator = DynamicThresholdCalibrator()
    with record_softmax_sparsity(calibrator, threshold_trials=[1e-3]):
        assert F.softmax is not original
    assert F.softmax is original


def test_record_softmax_sparsity_restores_original_softmax_on_exception():
    original = F.softmax
    calibrator = DynamicThresholdCalibrator()
    try:
        with record_softmax_sparsity(calibrator, threshold_trials=[1e-3]):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert F.softmax is original


def test_record_softmax_sparsity_averages_across_multiple_calls():
    """Multiple calls within one context (e.g. one per attention layer of a single forward
    pass) must be averaged into a single observation per threshold when the context exits, not
    recorded as independent points -- see the module docstring for why."""
    calibrator = DynamicThresholdCalibrator()
    with record_softmax_sparsity(calibrator, threshold_trials=[1e-3, 1e-2]):
        F.softmax(torch.randn(1, 2, 8, 32), dim=-1)
        F.softmax(torch.randn(1, 2, 8, 16), dim=-1)

    assert len(calibrator._scale_factors) == 2
    assert len(calibrator._sparsities) == 2


def test_record_softmax_sparsity_averages_sparsity_values_correctly():
    import quark.torch.sparsity.attention_sparsity.calibration.context as context_module

    class _FakeMethod:
        def __init__(self, block_size, query_block_size):
            pass

        def calculate_sparsities(self, scores, thresholds):
            # Return a distinct, known sparsity per call so the average is checkable.
            call_index = int(scores[0, 0, 0, 0].item())
            return [0.1 * call_index for _ in thresholds]

    original_method_cls = context_module.FlashSkipSoftmax
    context_module.FlashSkipSoftmax = _FakeMethod
    try:
        calibrator = DynamicThresholdCalibrator()
        with record_softmax_sparsity(calibrator, threshold_trials=[1e-3, 1e-2]):
            F.softmax(torch.full((1, 2, 8, 32), 1.0), dim=-1)  # sparsity 0.1 for both thresholds
            F.softmax(torch.full((1, 2, 8, 16), 3.0), dim=-1)  # sparsity 0.3 for both thresholds
    finally:
        context_module.FlashSkipSoftmax = original_method_cls

    assert calibrator._sparsities == pytest.approx([0.2, 0.2])
