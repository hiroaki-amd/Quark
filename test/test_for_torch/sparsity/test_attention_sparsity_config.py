#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.sparsity.attention_sparsity.config import (
    DEFAULT_THRESHOLD_TRIALS,
    CalibrationConfig,
    SparseAttentionAttributeConfig,
    SparseAttentionConfig,
    ThresholdScaleFactor,
    build_sparse_attention_config_dict,
)


def test_threshold_scale_factor_defaults():
    coeffs = ThresholdScaleFactor(a=7.4142, b=9.6915)
    assert coeffs.a == 7.4142
    assert coeffs.b == 9.6915


def test_sparse_attention_attribute_config_defaults():
    group = SparseAttentionAttributeConfig()
    assert group.algorithm == "skip_softmax"
    assert group.targets == []
    assert group.threshold_scale_factor == {}
    assert group.target_sparsity == {}
    assert group.ignore == []


def test_sparse_attention_attribute_config_to_dict_shape():
    group = SparseAttentionAttributeConfig(
        targets=["*self_attn*"],
        threshold_scale_factor={
            "prefill": ThresholdScaleFactor(a=7.4142, b=9.6915),
            "decode": ThresholdScaleFactor(a=0.12, b=9.85),
        },
        target_sparsity={"prefill": 0.5, "decode": 0.3},
        ignore=["model.layers.0.self_attn"],
    )

    d = group.to_dict()

    assert d["algorithm"] == "skip_softmax"
    assert d["targets"] == ["*self_attn*"]
    assert d["threshold_scale_factor"]["formula"] == "a * exp(b * target_sparsity)"
    assert d["threshold_scale_factor"]["prefill"] == {"a": 7.4142, "b": 9.6915}
    assert d["threshold_scale_factor"]["decode"] == {"a": 0.12, "b": 9.85}
    assert d["target_sparsity"] == {"prefill": 0.5, "decode": 0.3}
    assert d["ignore"] == ["model.layers.0.self_attn"]


def test_sparse_attention_attribute_config_partial_phase():
    # Only prefill calibrated -- decode absent means "not yet calibrated", not zero.
    group = SparseAttentionAttributeConfig(
        threshold_scale_factor={"prefill": ThresholdScaleFactor(a=1.0, b=2.0)},
    )
    d = group.to_dict()
    assert "prefill" in d["threshold_scale_factor"]
    assert "decode" not in d["threshold_scale_factor"]


def test_calibration_config_defaults():
    cfg = CalibrationConfig()
    assert cfg.name == "skip_softmax_calibration"
    assert cfg.target_sparse_ratio == {"prefill": 0.5, "decode": 0.5}
    assert cfg.samples == 24
    assert cfg.threshold_trials == DEFAULT_THRESHOLD_TRIALS
    assert cfg.fit_logspace is False


def test_calibration_config_threshold_trials_independent_copies():
    sentinel = 0.123456
    cfg1 = CalibrationConfig()
    cfg2 = CalibrationConfig()
    assert sentinel not in DEFAULT_THRESHOLD_TRIALS
    cfg1.threshold_trials.append(sentinel)
    assert sentinel not in cfg2.threshold_trials
    assert sentinel not in DEFAULT_THRESHOLD_TRIALS


def test_sparse_attention_config_defaults():
    cfg = SparseAttentionConfig()
    assert cfg.sparse_cfg == {}
    assert cfg.calibration_config is None


def test_build_sparse_attention_config_dict_shape():
    sparse_cfg = {
        "group_0": SparseAttentionAttributeConfig(
            targets=["*self_attn*"],
            threshold_scale_factor={
                "prefill": ThresholdScaleFactor(a=7.4142, b=9.6915),
                "decode": ThresholdScaleFactor(a=0.12, b=9.85),
            },
            target_sparsity={"prefill": 0.5},
            ignore=["model.layers.0.self_attn"],
        )
    }

    exported = build_sparse_attention_config_dict(sparse_cfg)

    assert "config_groups" in exported
    assert "group_0" in exported["config_groups"]

    group_0 = exported["config_groups"]["group_0"]
    assert group_0["algorithm"] == "skip_softmax"
    assert group_0["threshold_scale_factor"]["formula"] == "a * exp(b * target_sparsity)"
    assert group_0["threshold_scale_factor"]["prefill"] == {"a": 7.4142, "b": 9.6915}
    assert group_0["target_sparsity"] == {"prefill": 0.5}
    assert group_0["ignore"] == ["model.layers.0.self_attn"]

    assert exported["producer"]["name"] == "quark"
    assert "version" in exported["producer"]


def test_build_sparse_attention_config_dict_empty():
    exported = build_sparse_attention_config_dict({})
    assert exported["config_groups"] == {}
    assert exported["producer"]["name"] == "quark"


def test_build_sparse_attention_config_dict_multiple_groups():
    sparse_cfg = {
        "group_0": SparseAttentionAttributeConfig(targets=["*self_attn*"]),
        "group_1": SparseAttentionAttributeConfig(targets=["*cross_attn*"], ignore=["*.0.*"]),
    }
    exported = build_sparse_attention_config_dict(sparse_cfg)
    assert set(exported["config_groups"].keys()) == {"group_0", "group_1"}
    assert exported["config_groups"]["group_1"]["ignore"] == ["*.0.*"]
