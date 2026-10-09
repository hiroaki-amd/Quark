# Attention Sparsity (Skip-Softmax Attention / BLASST) Calibration

This package calibrates threshold/sparsity coefficients for skip-softmax attention
(BLASST, arXiv:2512.12087): a block-sparse attention technique that skips key blocks
whose contribution to the softmax normalizer is negligible, based on a per-model,
per-phase threshold derived from a target sparsity.

## Overview

The runtime kernel decides whether to skip a key block using a single scalar
`threshold`. Calibration is the offline step that finds, for a given model, the
`(a, b)` coefficients of `scale_factor = a * exp(b * target_sparsity)`, from which the
runtime threshold is derived as `threshold = scale_factor / seq_len`. This lets a
deployment pick a `target_sparsity` (e.g. 0.5) without having to hand-tune a raw
threshold value per model and context length.

Calibration works by running real forward passes of the model in
`attn_implementation="eager"` mode, intercepting `torch.nn.functional.softmax` to
observe, for a sweep of candidate thresholds, what sparsity each one would actually
produce on the model's real attention score distributions. Eager mode is required only
for this offline step, since fused attention backends (SDPA, FlashAttention) never
materialize scores as a distinct tensor passed to `F.softmax`. The resulting
`(a, b)` fit is then used at serving time by the real (non-eager) skip-softmax kernel.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.torch.sparsity.attention_sparsity.api import ModelSparseAttentionCalibrator
from quark.torch.sparsity.attention_sparsity.config import CalibrationConfig
from quark.torch.sparsity.attention_sparsity.export import export_sparse_attention_config

model_name = "Qwen/Qwen3-8B"  # or e.g. "NousResearch/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenized calibration samples, each shape [1, seq_len]. Use real prompts spanning a
# range of lengths relevant to your deployment (see "Calibration samples" below).
input_ids_list = [tokenizer(prompt, return_tensors="pt").input_ids.cuda() for prompt in calibration_prompts]

config = CalibrationConfig(
    target_sparse_ratio={"prefill": 0.5, "decode": 0.0},
    block_size=64,          # kernel's key-block tile size (BLOCK_N)
    query_block_size=128,   # kernel's query-tile size (BLOCK_M)
    chunk_size=8192,        # bound eager-mode memory at long context; see "Calibration
                            # samples" below. -1 (the default) disables chunking.
    fit_logspace=True,
)
calibrator = ModelSparseAttentionCalibrator(config)
group = calibrator.calibrate_model(model, input_ids_list)

export_sparse_attention_config(model, {"group_0": group})
model.config.save_pretrained("/path/to/checkpoint")
```

This writes a `sparse_attention_config` block into the checkpoint's `config.json`,
which a serving engine reads back with `load_sparse_attention_config_from_checkpoint`
and `resolve_layer_sparse_config` (see `export.py`) to decide, per attention layer, which
`threshold_scale_factor` and `target_sparsity` to use.

### Generating `calibration_prompts` with RULER

RULER ([NVIDIA/RULER](https://github.com/NVIDIA/RULER)) is one convenient source of
long-context prompts at controlled exact lengths; it is not a requirement, and any
corpus of real text with the right length distribution for your deployment works
equally well (see "Calibration samples" below). To generate RULER prompts at a given
context length, use its own `scripts/data/prepare.py` data generator (clone the RULER
repo and follow its README for setup), then load the resulting `validation.jsonl`
files, one per task:

```python
import glob
import json

calibration_prompts = []
for path in glob.glob("/path/to/ruler_data/ctx_*/*/validation.jsonl"):
    with open(path) as f:
        calibration_prompts.extend(json.loads(line)["input"] for line in f)
```

Repeat `prepare.py` once per context length you want represented (e.g. 4096 through
131072) to cover the length range your deployment sees, as recommended below.

## Calibration samples

Use real tokenized prompts, not synthetic/random token IDs: the calibration observes
the actual distribution of attention scores, which depends on real text structure.
Cover a range of sequence lengths relevant to your deployment, since sparsity at a
fixed threshold varies with sequence length (this is exactly what the
`a * exp(b * target_sparsity)` fit, pooled over an `add_observation(threshold, seq_len,
sparsity)` call per `(threshold, sample)` pair, is meant to capture).

At long context, eager-mode attention's memory use grows with sequence length; set
`chunk_size` to run prefill through a real KV cache in pieces and bound this (see
`CalibrationConfig.chunk_size`). Each chunk is observed against its own (growing)
KV-cache depth rather than the sample's eventual total length, which is also the more
accurate choice: causal masking means a query's true attended-context size depends on
its own position, not on how long the sample eventually becomes, and it matches what a
real chunked-prefill serving engine would see at inference time. For this same reason,
prefer setting `chunk_size` to match your production serving engine's actual
chunked-prefill granularity, if it uses one; calibrating with an arbitrary `chunk_size`
chosen only for offline memory convenience can otherwise produce a fit that does not
faithfully match real serving behavior.

For some instruction-tuned models (observed with Llama-3.1-8B-Instruct on RULER),
wrapping prompts in the model's chat template can distort long-context aggregation-task
behavior enough to make calibration on dense accuracy numbers misleading. If accuracy
validation at long context looks off, check whether raw (non-chat-template) prompting
with any task-specific answer prefix matches the model's expected evaluation format
better.

## `block_size` / `query_block_size`

These must match the real kernel's tiling (`BLOCK_N`/`BLOCK_M`) to get a calibration
that reflects the kernel's actual all-or-nothing skip granularity per tile: a key block
is only skippable for a whole query tile if every row within that tile independently
agrees it's skippable. Calibrating with the wrong tile size (or the default
`query_block_size=1`, i.e. per-row decisions) under-counts the kernel's real sparsity
loss from tile-level granularity.

## Prefill vs. decode

Prefill and decode are calibrated independently (`target_sparse_ratio["prefill"]` /
`["decode"]`), each producing its own `(a, b)` fit, because the two phases have very
different attention score distributions (single long query vs. one query token against
a growing cache). Decode-phase skip-softmax has been observed to be substantially more
prone to over-skipping and degrading accuracy than prefill-phase skipping at the same
target sparsity; until this is independently validated for your model and deployment,
setting `target_sparse_ratio["decode"] = 0.0` (prefill-only sparsity, dense decode) is
the safer default. A phase whose `target_sparse_ratio` is `0.0` is skipped entirely by
`ModelSparseAttentionCalibrator.calibrate_model` and omitted from the exported group's
`threshold_scale_factor`.

## Worked examples

Config groups calibrated this way for two models (`block_size=64`, `query_block_size=128`,
`target_sparse_ratio={"prefill": 0.5, "decode": 0.0}`, `fit_logspace=True`, RULER prompts
at context lengths from 4096 to 131072):

| model | a | b | r_squared |
| --- | --- | --- | --- |
| Qwen3-8B | 6.278663 | 9.001448 | 0.9616 |
| Llama-3.1-8B-Instruct | 0.888507 | 7.786403 | n/a |

These coefficients are specific to the model, the sampled prompts, and the
`block_size`/`query_block_size` used above; recalibrate for any other model or kernel
tiling rather than reusing them directly.

The Llama-3.1-8B-Instruct row was produced by actually running the documented Quick
Start recipe above (`ModelSparseAttentionCalibrator.calibrate_model` with
`chunk_size=2048`), against RULER prompts generated the same way as "Generating
`calibration_prompts` with RULER" above (8 samples per context length, 4096 through
131072, round-robined across RULER's 13 tasks), so it is directly reproducible by
that recipe. The Qwen3-8B row predates the `chunk_size` feature and was produced by a
separate internal measurement path that has not yet been re-validated against the
Quick Start recipe; treat it as indicative rather than reproducible until that
validation is done.
