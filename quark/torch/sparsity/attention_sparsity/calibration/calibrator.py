#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Threshold/sparsity curve fitting for BLASST (skip-softmax) calibration.

Pure math, no model or GPU dependency. Given observed
``(scale_factor, sparsity)`` data points collected by sweeping a fixed list of
candidate thresholds through an attention module, fits
``scale_factor = a * exp(b * sparsity)``.

Uses a fixed default sweep of thresholds, pooled (not per-threshold-averaged)
data points filtered to a reliable sparsity range, and a choice between
linear-space (absolute error) and log-space (relative error) fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from quark.common.config import BaseConfigImpl
from quark.torch.sparsity.attention_sparsity.config import ThresholdScaleFactor

# Data points with an observed sparsity outside this range are unreliable for
# fitting (near 0: almost no skip decisions to observe; near 1: attention is
# nearly fully skipped, dominated by boundary effects) and are discarded.
DEFAULT_MIN_SPARSITY = 0.10
DEFAULT_MAX_SPARSITY = 0.90

# Upper bound on the fitted exponential-rate coefficient `b`. Prevents the
# solver from finding a numerically unstable, implausibly steep fit when data
# is sparse or noisy.
_MAX_B = 20.0


def _skip_softmax_curve(sparsity: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(b * sparsity)


@dataclass(eq=True)
class CalibrationFitResult(BaseConfigImpl):
    """
    Result of fitting ``scale_factor = a * exp(b * sparsity)`` to observed data.

    :param float a: Fitted multiplicative coefficient.
    :param float b: Fitted exponential-rate coefficient.
    :param float r_squared: Coefficient of determination, computed in the space the fit was
        performed in (log space when ``fit_logspace`` is ``True``).
    :param float min_observed_sparsity: Smallest observed sparsity among the data points used
        for fitting.
    :param float max_observed_sparsity: Largest observed sparsity among the data points used
        for fitting.
    :param int num_points: Number of data points used for fitting, after range filtering.
    :param bool fit_logspace: Whether the fit was performed in log space.
    """

    a: float
    b: float
    r_squared: float
    min_observed_sparsity: float
    max_observed_sparsity: float
    num_points: int
    fit_logspace: bool = False

    def to_threshold_scale_factor(self) -> ThresholdScaleFactor:
        return ThresholdScaleFactor(a=self.a, b=self.b)


class DynamicThresholdCalibrator:
    """
    Pools ``(scale_factor, sparsity)`` observations and fits the skip-softmax
    threshold-scale-factor curve.

    Usage: call :meth:`add_observation` (or :meth:`add_observations`) once per
    ``(threshold, seq_len, sparsity)`` triple measured while sweeping
    ``threshold_trials`` through an attention module, then call :meth:`fit`.

    :param float min_sparsity: Lower bound (exclusive) of the reliable sparsity range used to
        filter data points before fitting.
    :param float max_sparsity: Upper bound (exclusive) of the reliable sparsity range used to
        filter data points before fitting.
    """

    def __init__(self, min_sparsity: float = DEFAULT_MIN_SPARSITY, max_sparsity: float = DEFAULT_MAX_SPARSITY) -> None:
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self._scale_factors: list[float] = []
        self._sparsities: list[float] = []

    def add_observation(self, threshold: float, seq_len: int, sparsity: float) -> None:
        """Record one ``(threshold, seq_len, sparsity)`` observation."""
        self._scale_factors.append(threshold * seq_len)
        self._sparsities.append(sparsity)

    def add_observations(
        self,
        thresholds: list[float],
        seq_len: int,
        sparsities: list[float],
    ) -> None:
        """Record observations from one forward pass sweeping multiple thresholds."""
        if len(thresholds) != len(sparsities):
            raise ValueError(
                f"thresholds and sparsities must have the same length, got {len(thresholds)} and {len(sparsities)}"
            )
        for threshold, sparsity in zip(thresholds, sparsities):
            self.add_observation(threshold, seq_len, sparsity)

    def _filtered_points(self) -> tuple[np.ndarray, np.ndarray]:
        scale_factors = np.asarray(self._scale_factors, dtype=np.float64)
        sparsities = np.asarray(self._sparsities, dtype=np.float64)
        mask = (sparsities > self.min_sparsity) & (sparsities < self.max_sparsity)
        return scale_factors[mask], sparsities[mask]

    def fit(self, fit_logspace: bool = False) -> CalibrationFitResult:
        """
        Fit ``scale_factor = a * exp(b * sparsity)`` over all pooled observations.

        :param bool fit_logspace: If ``True``, fit in log space (minimizes relative error;
            recommended when ``scale_factor`` spans many orders of magnitude). If ``False``
            (default), fit in linear space (minimizes absolute error).
        :raises ValueError: If fewer than 2 data points remain after filtering to the reliable
            sparsity range.
        :return: The fitted coefficients and fit diagnostics.
        :rtype: CalibrationFitResult
        """
        scale_factors, sparsities = self._filtered_points()

        if len(scale_factors) < 2:
            raise ValueError(
                f"Need at least 2 data points with sparsity in ({self.min_sparsity}, {self.max_sparsity}) to fit, "
                f"got {len(scale_factors)} (out of {len(self._scale_factors)} total observations)."
            )

        if fit_logspace:
            log_scale_factors = np.log(scale_factors)
            # log(scale_factor) = log(a) + b * sparsity is linear in (log(a), b);
            # solved analytically via least squares rather than iterative curve_fit.
            b, log_a = np.polyfit(sparsities, log_scale_factors, 1)
            a = float(np.exp(log_a))
            b = float(np.clip(b, 0.0, _MAX_B))
            predicted = np.log(a) + b * sparsities
            r_squared = _r_squared(log_scale_factors, predicted)
        else:
            (a, b), _ = curve_fit(
                _skip_softmax_curve,
                sparsities,
                scale_factors,
                p0=(1.0, 1.0),
                bounds=([0.0, 0.0], [np.inf, _MAX_B]),
                maxfev=10000,
            )
            a, b = float(a), float(b)
            predicted = _skip_softmax_curve(sparsities, a, b)
            r_squared = _r_squared(scale_factors, predicted)

        return CalibrationFitResult(
            a=a,
            b=b,
            r_squared=r_squared,
            min_observed_sparsity=float(sparsities.min()),
            max_observed_sparsity=float(sparsities.max()),
            num_points=len(scale_factors),
            fit_logspace=fit_logspace,
        )


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual_sum_squares = float(np.sum((observed - predicted) ** 2))
    total_sum_squares = float(np.sum((observed - observed.mean()) ** 2))
    if total_sum_squares == 0.0:
        return 1.0 if residual_sum_squares == 0.0 else 0.0
    return 1.0 - residual_sum_squares / total_sum_squares
