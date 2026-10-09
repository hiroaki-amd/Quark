#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Checkpoint export/reload round-trip tests for BLASST calibration config.

Validates the schema contract that ATOM (Phase 5) will eventually consume, without
depending on ATOM itself: calibrate a tiny model (Phase 3) -> export to a real
``config.json`` on disk -> reload it independently of the in-memory objects -> resolve
per-layer config with fnmatch patterns -> assert it matches what was fitted.
"""

import json

import pytest
import torch

from quark.torch.sparsity.attention_sparsity.api import ModelSparseAttentionCalibrator
from quark.torch.sparsity.attention_sparsity.config import (
    CalibrationConfig,
    SparseAttentionAttributeConfig,
    ThresholdScaleFactor,
    build_sparse_attention_config_dict,
)
from quark.torch.sparsity.attention_sparsity.export import (
    export_sparse_attention_config,
    load_sparse_attention_config_from_checkpoint,
    resolve_layer_sparse_config,
)

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
        initializer_range=0.2,
    )
    return transformers.AutoModelForCausalLM.from_config(config)


@pytest.fixture
def input_ids_list():
    torch.manual_seed(0)
    return [torch.randint(0, 100, (2, 256)) for _ in range(3)]


def test_export_writes_sparse_attention_config_onto_model_config(tiny_model, input_ids_list):
    calib_config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8)
    group = ModelSparseAttentionCalibrator(calib_config).calibrate_model(tiny_model, input_ids_list)

    export_sparse_attention_config(tiny_model, {"group_0": group})

    exported = tiny_model.config.sparse_attention_config
    assert exported["config_groups"]["group_0"]["algorithm"] == "skip_softmax"
    assert "prefill" in exported["config_groups"]["group_0"]["threshold_scale_factor"]


def test_calibrate_export_reload_round_trip(tiny_model, input_ids_list, tmp_path):
    calib_config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.5}, num_decode_tokens=4, block_size=8)
    group = ModelSparseAttentionCalibrator(calib_config).calibrate_model(tiny_model, input_ids_list)

    export_sparse_attention_config(tiny_model, {"group_0": group})
    tiny_model.config.save_pretrained(tmp_path)

    reloaded = load_sparse_attention_config_from_checkpoint(tmp_path)
    assert reloaded is not None

    resolved = resolve_layer_sparse_config("model.layers.0.self_attn", reloaded)
    assert resolved is not None
    assert resolved["algorithm"] == "skip_softmax"
    assert resolved["threshold_scale_factor"]["prefill"]["a"] == pytest.approx(
        group.threshold_scale_factor["prefill"].a
    )
    assert resolved["threshold_scale_factor"]["prefill"]["b"] == pytest.approx(
        group.threshold_scale_factor["prefill"].b
    )
    assert resolved["threshold_scale_factor"]["decode"]["a"] == pytest.approx(
        group.threshold_scale_factor["decode"].a
    )
    assert resolved["target_sparsity"] == {"prefill": 0.5, "decode": 0.5}


def test_load_sparse_attention_config_from_checkpoint_returns_none_when_absent(tiny_model, tmp_path):
    tiny_model.config.save_pretrained(tmp_path)

    assert load_sparse_attention_config_from_checkpoint(tmp_path) is None


def test_resolve_layer_sparse_config_respects_ignore_before_targets():
    group = SparseAttentionAttributeConfig(
        targets=["*self_attn*"],
        ignore=["*layers.0*"],
        threshold_scale_factor={"prefill": ThresholdScaleFactor(a=1.0, b=2.0)},
        target_sparsity={"prefill": 0.5},
    )
    sparse_attention_config = build_sparse_attention_config_dict({"group_0": group})

    assert resolve_layer_sparse_config("model.layers.0.self_attn", sparse_attention_config) is None
    resolved = resolve_layer_sparse_config("model.layers.1.self_attn", sparse_attention_config)
    assert resolved is not None
    assert resolved["algorithm"] == "skip_softmax"


def test_resolve_layer_sparse_config_returns_none_for_unmatched_layer():
    group = SparseAttentionAttributeConfig(
        targets=["*self_attn*"],
        threshold_scale_factor={"prefill": ThresholdScaleFactor(a=1.0, b=2.0)},
        target_sparsity={"prefill": 0.5},
    )
    sparse_attention_config = build_sparse_attention_config_dict({"group_0": group})

    assert resolve_layer_sparse_config("model.layers.0.mlp", sparse_attention_config) is None


def test_resolve_layer_sparse_config_first_matching_group_wins():
    group_a = SparseAttentionAttributeConfig(
        targets=["*self_attn*"],
        threshold_scale_factor={"prefill": ThresholdScaleFactor(a=1.0, b=2.0)},
        target_sparsity={"prefill": 0.5},
    )
    group_b = SparseAttentionAttributeConfig(
        targets=["*self_attn*"],
        threshold_scale_factor={"prefill": ThresholdScaleFactor(a=99.0, b=99.0)},
        target_sparsity={"prefill": 0.9},
    )
    sparse_attention_config = build_sparse_attention_config_dict({"group_a": group_a, "group_b": group_b})

    resolved = resolve_layer_sparse_config("model.layers.0.self_attn", sparse_attention_config)
    assert resolved["threshold_scale_factor"]["prefill"]["a"] == 1.0


def test_exported_config_json_is_plain_json_serializable(tiny_model, input_ids_list, tmp_path):
    calib_config = CalibrationConfig(target_sparse_ratio={"prefill": 0.5, "decode": 0.0}, block_size=8)
    group = ModelSparseAttentionCalibrator(calib_config).calibrate_model(tiny_model, input_ids_list)

    export_sparse_attention_config(tiny_model, {"group_0": group})
    tiny_model.config.save_pretrained(tmp_path)

    with open(tmp_path / "config.json") as f:
        raw = json.load(f)
    assert "sparse_attention_config" in raw
