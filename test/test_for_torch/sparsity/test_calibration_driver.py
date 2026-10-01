#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""End-to-end test of ModelSparseAttentionCalibrator against a tiny HF Llama model.

Uses a randomly initialized, untrained model, so the calibration output is not
expected to be numerically meaningful -- these tests assert the driver runs a real
forward pass end-to-end and returns plausible (in-range, finite) fitted coefficients,
not that the fit matches any ground truth.
"""

import pytest
import torch

from quark.torch.sparsity.attention_sparsity.api import ModelSparseAttentionCalibrator
from quark.torch.sparsity.attention_sparsity.calibration.calibrator import DynamicThresholdCalibrator
from quark.torch.sparsity.attention_sparsity.config import CalibrationConfig

transformers = pytest.importorskip("transformers")


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=256,
        attn_implementation="eager",
        # A larger-than-default initializer_range gives this untrained model enough attention-
        # score dynamic range to produce a realistic spread of sparsity values across the
        # threshold sweep (the HF default range is tuned for training stability, not for
        # producing "interesting" random attention patterns).
        initializer_range=0.2,
    )
    return transformers.AutoModelForCausalLM.from_config(config)


@pytest.fixture
def input_ids_list():
    torch.manual_seed(0)
    return [torch.randint(0, 100, (2, 256)) for _ in range(3)]


def _plausible_fit_assertions(coeffs):
    assert coeffs.a > 0.0
    assert 0.0 <= coeffs.b <= 20.0


def test_prefill_only_calibration(tiny_model, input_ids_list):
    config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8)
    calibrator = ModelSparseAttentionCalibrator(config)

    group = calibrator.calibrate_model(tiny_model, input_ids_list)

    assert "prefill" in group.threshold_scale_factor
    assert "decode" not in group.threshold_scale_factor
    _plausible_fit_assertions(group.threshold_scale_factor["prefill"])


def test_decode_only_calibration(tiny_model, input_ids_list):
    config = CalibrationConfig(
        target_sparse_ratio={"prefill": 0.0, "decode": 0.5},
        num_decode_tokens=4,
        block_size=8,
    )
    calibrator = ModelSparseAttentionCalibrator(config)

    group = calibrator.calibrate_model(tiny_model, input_ids_list)

    assert "decode" in group.threshold_scale_factor
    assert "prefill" not in group.threshold_scale_factor
    _plausible_fit_assertions(group.threshold_scale_factor["decode"])


def test_prefill_and_decode_calibration_are_independent(tiny_model, input_ids_list):
    config = CalibrationConfig(
        target_sparse_ratio={"prefill": 0.5, "decode": 0.5},
        num_decode_tokens=4,
        block_size=8,
    )
    calibrator = ModelSparseAttentionCalibrator(config)

    group = calibrator.calibrate_model(tiny_model, input_ids_list)

    assert set(group.threshold_scale_factor.keys()) == {"prefill", "decode"}
    _plausible_fit_assertions(group.threshold_scale_factor["prefill"])
    _plausible_fit_assertions(group.threshold_scale_factor["decode"])


def test_calibration_restores_original_attn_implementation(tiny_model, input_ids_list):
    tiny_model.config._attn_implementation = "sdpa"
    config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8)
    calibrator = ModelSparseAttentionCalibrator(config)

    calibrator.calibrate_model(tiny_model, input_ids_list)

    assert tiny_model.config._attn_implementation == "sdpa"


def test_target_sparsity_recorded_in_group(tiny_model, input_ids_list):
    config = CalibrationConfig(target_sparse_ratio={"prefill": 0.3, "decode": 0.0}, block_size=8)
    calibrator = ModelSparseAttentionCalibrator(config)

    group = calibrator.calibrate_model(tiny_model, input_ids_list)

    assert group.target_sparsity == {"prefill": 0.3, "decode": 0.0}


def test_chunked_prefill_observes_growing_kv_length_per_chunk(tiny_model, input_ids_list):
    """Each chunk must be observed against its own (growing) KV-cache depth, not the
    sample's final/full length: this is what lets the resulting fit match a serving
    deployment that *also* uses chunked prefill at this chunk_size, where the threshold
    derived from the fit is applied against whatever context length has been seen so far,
    not the eventual total. A chunk_size of 96 over a 256-token sample should therefore
    yield observations at kv lengths 96, 192, and 256 -- never only 256."""
    from quark.torch.sparsity.attention_sparsity.calibration.context import record_softmax_sparsity

    config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8, chunk_size=96)
    calibrator_driver = ModelSparseAttentionCalibrator(config)
    calibrator = DynamicThresholdCalibrator()
    observed_seq_lens = []
    original_add_observation = calibrator.add_observation
    calibrator.add_observation = lambda threshold, seq_len, sparsity: (
        observed_seq_lens.append(seq_len),
        original_add_observation(threshold, seq_len, sparsity),
    )[1]

    with torch.no_grad(), record_softmax_sparsity(calibrator, config.threshold_trials, block_size=config.block_size):
        calibrator_driver._run_chunked_prefill(tiny_model, input_ids_list[0])

    assert set(observed_seq_lens) == {96, 192, 256}


def test_chunked_prefill_produces_plausible_fit(tiny_model, input_ids_list):
    config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8, chunk_size=96)
    calibrator = ModelSparseAttentionCalibrator(config)

    group = calibrator.calibrate_model(tiny_model, input_ids_list)

    _plausible_fit_assertions(group.threshold_scale_factor["prefill"])


def test_chunked_prefill_handles_non_dividing_chunk_size(tiny_model, input_ids_list):
    """chunk_size need not evenly divide the sample length (256 here)."""
    config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8, chunk_size=100)
    calibrator = ModelSparseAttentionCalibrator(config)

    group = calibrator.calibrate_model(tiny_model, input_ids_list)

    _plausible_fit_assertions(group.threshold_scale_factor["prefill"])


def test_chunk_size_equal_to_seq_len_is_single_chunk(tiny_model, input_ids_list):
    """chunk_size == seq_len should behave identically to no chunking (one chunk, same as
    disabling chunking outright)."""
    equal_config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8, chunk_size=256)
    disabled_config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8, chunk_size=-1)

    equal_group = ModelSparseAttentionCalibrator(equal_config).calibrate_model(tiny_model, input_ids_list)
    disabled_group = ModelSparseAttentionCalibrator(disabled_config).calibrate_model(tiny_model, input_ids_list)

    assert equal_group.threshold_scale_factor["prefill"].a == pytest.approx(
        disabled_group.threshold_scale_factor["prefill"].a, rel=1e-5
    )
    assert equal_group.threshold_scale_factor["prefill"].b == pytest.approx(
        disabled_group.threshold_scale_factor["prefill"].b, rel=1e-5
    )


def test_chunked_decode_warmup_prefill(tiny_model, input_ids_list):
    """The decode phase's unrecorded warm-up prefill must also chunk, so a long sample
    doesn't blow the eager-mode memory budget before any decode step runs."""
    config = CalibrationConfig(
        target_sparse_ratio={"prefill": 0.0, "decode": 0.5},
        num_decode_tokens=4,
        block_size=8,
        chunk_size=96,
    )
    calibrator = ModelSparseAttentionCalibrator(config)

    group = calibrator.calibrate_model(tiny_model, input_ids_list)

    assert "decode" in group.threshold_scale_factor
    _plausible_fit_assertions(group.threshold_scale_factor["decode"])
