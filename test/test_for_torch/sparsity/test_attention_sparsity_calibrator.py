#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import pytest

from quark.torch.sparsity.attention_sparsity.calibration import (
    CalibrationFitResult,
    DynamicThresholdCalibrator,
)
from quark.torch.sparsity.attention_sparsity.config import ThresholdScaleFactor

TRUE_A = 5.0
TRUE_B = 6.0


def _synthetic_points(a: float, b: float, num_points: int = 60, noise_std: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    sparsities = np.linspace(0.02, 0.98, num_points)
    scale_factors = a * np.exp(b * sparsities)
    if noise_std:
        scale_factors = scale_factors * (1.0 + rng.normal(0.0, noise_std, size=num_points))
    return sparsities, scale_factors


def _populate(calibrator: DynamicThresholdCalibrator, sparsities, scale_factors, seq_len: int = 1000):
    for sparsity, scale_factor in zip(sparsities, scale_factors):
        # add_observation stores threshold * seq_len as the scale factor, so pick
        # threshold = scale_factor / seq_len to land on the desired scale_factor exactly.
        calibrator.add_observation(threshold=scale_factor / seq_len, seq_len=seq_len, sparsity=float(sparsity))


def test_fit_recovers_ground_truth_linear_space_noiseless():
    sparsities, scale_factors = _synthetic_points(TRUE_A, TRUE_B, noise_std=0.0)
    calibrator = DynamicThresholdCalibrator()
    _populate(calibrator, sparsities, scale_factors)

    result = calibrator.fit(fit_logspace=False)

    assert result.a == pytest.approx(TRUE_A, rel=1e-3)
    assert result.b == pytest.approx(TRUE_B, rel=1e-3)
    assert result.r_squared > 0.999
    assert result.fit_logspace is False


def test_fit_recovers_ground_truth_logspace_noiseless():
    sparsities, scale_factors = _synthetic_points(TRUE_A, TRUE_B, noise_std=0.0)
    calibrator = DynamicThresholdCalibrator()
    _populate(calibrator, sparsities, scale_factors)

    result = calibrator.fit(fit_logspace=True)

    assert result.a == pytest.approx(TRUE_A, rel=1e-3)
    assert result.b == pytest.approx(TRUE_B, rel=1e-3)
    assert result.r_squared > 0.999
    assert result.fit_logspace is True


def test_fit_recovers_ground_truth_with_noise_both_modes():
    sparsities, scale_factors = _synthetic_points(TRUE_A, TRUE_B, noise_std=0.02, seed=42)

    for fit_logspace in (False, True):
        calibrator = DynamicThresholdCalibrator()
        _populate(calibrator, sparsities, scale_factors)
        result = calibrator.fit(fit_logspace=fit_logspace)
        assert result.a == pytest.approx(TRUE_A, rel=0.1)
        assert result.b == pytest.approx(TRUE_B, rel=0.1)
        assert result.r_squared > 0.9


def test_fit_filters_out_of_range_sparsity():
    calibrator = DynamicThresholdCalibrator()
    # In-range points on the true curve.
    in_range_sparsities = np.linspace(0.2, 0.8, 20)
    in_range_scale_factors = TRUE_A * np.exp(TRUE_B * in_range_sparsities)
    _populate(calibrator, in_range_sparsities, in_range_scale_factors)

    # Out-of-range points that would badly skew the fit if not filtered out.
    calibrator.add_observation(threshold=1e-9, seq_len=1000, sparsity=0.01)
    calibrator.add_observation(threshold=1e9, seq_len=1000, sparsity=0.999)

    result = calibrator.fit(fit_logspace=False)

    assert result.num_points == 20
    assert result.min_observed_sparsity >= 0.2
    assert result.max_observed_sparsity <= 0.8
    assert result.a == pytest.approx(TRUE_A, rel=1e-2)
    assert result.b == pytest.approx(TRUE_B, rel=1e-2)


def test_fit_raises_when_all_points_out_of_range():
    calibrator = DynamicThresholdCalibrator()
    calibrator.add_observation(threshold=1e-9, seq_len=1000, sparsity=0.01)
    calibrator.add_observation(threshold=1e9, seq_len=1000, sparsity=0.999)

    with pytest.raises(ValueError):
        calibrator.fit()


def test_fit_raises_with_single_point():
    calibrator = DynamicThresholdCalibrator()
    calibrator.add_observation(threshold=0.005, seq_len=1000, sparsity=0.5)

    with pytest.raises(ValueError):
        calibrator.fit()


def test_add_observations_batch_matches_add_observation_loop():
    thresholds = [0.001, 0.002, 0.003]
    sparsities = [0.3, 0.5, 0.7]

    batch_calibrator = DynamicThresholdCalibrator()
    batch_calibrator.add_observations(thresholds, seq_len=1000, sparsities=sparsities)

    loop_calibrator = DynamicThresholdCalibrator()
    for threshold, sparsity in zip(thresholds, sparsities):
        loop_calibrator.add_observation(threshold, seq_len=1000, sparsity=sparsity)

    assert batch_calibrator._scale_factors == loop_calibrator._scale_factors
    assert batch_calibrator._sparsities == loop_calibrator._sparsities


def test_add_observations_raises_on_length_mismatch():
    calibrator = DynamicThresholdCalibrator()
    with pytest.raises(ValueError):
        calibrator.add_observations([0.1, 0.2], seq_len=1000, sparsities=[0.5])


def test_calibration_fit_result_to_threshold_scale_factor():
    result = CalibrationFitResult(
        a=1.5,
        b=2.5,
        r_squared=0.99,
        min_observed_sparsity=0.2,
        max_observed_sparsity=0.8,
        num_points=10,
    )
    coeffs = result.to_threshold_scale_factor()
    assert isinstance(coeffs, ThresholdScaleFactor)
    assert coeffs.a == 1.5
    assert coeffs.b == 2.5
