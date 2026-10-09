#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Top-level BLASST (skip-softmax) calibration API, analogous to ``ModelQuantizer`` /
``ModelPruner``.

Runs the reference skip-softmax method's threshold/sparsity observations
(:mod:`quark.torch.sparsity.attention_sparsity.calibration.context`) over real
forward passes of a model, independently for the prefill and decode phases,
then fits each phase's ``scale_factor = a * exp(b * sparsity)`` coefficients
via :class:`~quark.torch.sparsity.attention_sparsity.calibration.calibrator.DynamicThresholdCalibrator`.

This initial phase is intentionally simplified: a single ``sparse_cfg`` group
covering the whole model (no per-layer target/ignore patterns yet), and a
plain list of already-tokenized prefill samples rather than a dataset loader.
"""

from __future__ import annotations

import torch

from quark.torch.sparsity.attention_sparsity.calibration.calibrator import DynamicThresholdCalibrator
from quark.torch.sparsity.attention_sparsity.calibration.context import record_softmax_sparsity
from quark.torch.sparsity.attention_sparsity.config import CalibrationConfig, SparseAttentionAttributeConfig


class ModelSparseAttentionCalibrator:
    """
    Calibrates BLASST (skip-softmax) threshold-scale-factor coefficients for a model.

    :param CalibrationConfig config: Calibration hyperparameters (target sparsity per phase,
        threshold sweep, fit mode, etc.).
    """

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

    def calibrate_model(
        self,
        model: torch.nn.Module,
        input_ids_list: list[torch.Tensor],
    ) -> SparseAttentionAttributeConfig:
        """
        Run prefill and/or decode calibration and return a fitted config group.

        :param torch.nn.Module model: A HuggingFace causal-LM model. Its
            ``config._attn_implementation`` is temporarily forced to ``"eager"`` for the
            duration of calibration (restored afterward), since the reference method observes
            sparsity by intercepting ``torch.nn.functional.softmax``, which fused attention
            backends never call.
        :param list[torch.Tensor] input_ids_list: Tokenized calibration samples, each of shape
            ``[batch, seq_len]``.
        :return: A single config group (``targets=["*"]``, no per-layer overrides) with the
            phases that had a nonzero ``target_sparse_ratio`` calibrated.
        :rtype: SparseAttentionAttributeConfig
        """
        original_attn_implementation = getattr(model.config, "_attn_implementation", None)
        model.config._attn_implementation = "eager"
        model.eval()
        try:
            threshold_scale_factor = {}
            if self.config.target_sparse_ratio.get("prefill", 0.0) > 0.0:
                threshold_scale_factor["prefill"] = self._calibrate_prefill(model, input_ids_list).to_threshold_scale_factor()
            if self.config.target_sparse_ratio.get("decode", 0.0) > 0.0:
                threshold_scale_factor["decode"] = self._calibrate_decode(model, input_ids_list).to_threshold_scale_factor()
        finally:
            if original_attn_implementation is not None:
                model.config._attn_implementation = original_attn_implementation

        return SparseAttentionAttributeConfig(
            targets=["*"],
            threshold_scale_factor=threshold_scale_factor,
            target_sparsity=dict(self.config.target_sparse_ratio),
        )

    def _run_chunked_prefill(self, model: torch.nn.Module, input_ids: torch.Tensor):
        """
        Run prefill for one sample, splitting it into ``self.config.chunk_size``-sized chunks
        carried forward through a real KV cache (``past_key_values``) when the sample exceeds
        that size, to bound eager-mode attention memory at long context.

        Each chunk's own (causally correct) attention against the accumulated cache is computed
        and observed independently by whatever ``F.softmax`` monkeypatch is active during the
        call (see :func:`record_softmax_sparsity`); unlike reassembling a full score tensor
        across chunks, this requires no post-hoc stitching of Q/K.

        :return: The final chunk's model output (with ``past_key_values`` populated).
        """
        seq_len = input_ids.shape[-1]
        chunk_size = self.config.chunk_size
        if chunk_size <= 0 or seq_len <= chunk_size:
            return model(input_ids, use_cache=True)

        past_key_values = None
        outputs = None
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            outputs = model(input_ids[:, start:end], past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
        return outputs

    def _calibrate_prefill(self, model: torch.nn.Module, input_ids_list: list[torch.Tensor]):
        calibrator = DynamicThresholdCalibrator()
        with (
            torch.no_grad(),
            record_softmax_sparsity(
                calibrator,
                self.config.threshold_trials,
                block_size=self.config.block_size,
                query_block_size=self.config.query_block_size,
            ),
        ):
            for input_ids in input_ids_list:
                self._run_chunked_prefill(model, input_ids)
        return calibrator.fit(fit_logspace=self.config.fit_logspace)

    def _calibrate_decode(self, model: torch.nn.Module, input_ids_list: list[torch.Tensor]):
        calibrator = DynamicThresholdCalibrator()
        with torch.no_grad():
            for input_ids in input_ids_list:
                # Prefill once, unrecorded, purely to build the KV cache for decode steps.
                # Chunked the same way as `_calibrate_prefill` so a long warm-up prefill doesn't
                # itself blow the eager-mode memory budget.
                outputs = self._run_chunked_prefill(model, input_ids)
                next_token = outputs.logits[:, -1:].argmax(dim=-1)

                with record_softmax_sparsity(
                    calibrator,
                    self.config.threshold_trials,
                    block_size=self.config.block_size,
                    query_block_size=self.config.query_block_size,
                ):
                    for _ in range(self.config.num_decode_tokens):
                        outputs = model(next_token, past_key_values=outputs.past_key_values, use_cache=True)
                        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        return calibrator.fit(fit_logspace=self.config.fit_logspace)
