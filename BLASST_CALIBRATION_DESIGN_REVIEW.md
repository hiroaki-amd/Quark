# BLASST attention-sparsity calibration in Quark: design for review

## Changes since review

An earlier version of this doc was reviewed by the Quark team, who sent four points of feedback.
All four were implemented and validated (not just discussed) before this
revision. Summary, in the order raised:

1. **Placement in `quark/contrib/`, not `quark/torch/`.** Agreed outright. The
   package now lives at `quark/contrib/attention_sparsity/` (code, tests, and
   docs together, per `docs/source/intro_contrib.rst`), with its examples
   under `examples/contrib/attention_sparsity/`. Section 2 below reflects the
   new layout. CODEOWNERS, CI workflow approval, and the Intent-to-Contribute
   issue are the one part of this we can't do ourselves from this checkout:
   flagging that as a question back to you.
2. **Softmax-hook hardening.** `calibration/context.py` no longer blanket-patches
   `torch.nn.functional.softmax` and filters by tensor shape; it now wraps
   `forward` only on attention-named modules (`*attn`/`*attention`), globally
   patching `F.softmax` only for the scope of that module's own call,
   mirroring the exact pattern Quark's own `prepare_for_attention_quant`
   already uses. This closes the MoE-router false-positive risk you flagged.
   We also confirmed your `softmax_quant_spec` concern was a real, not
   hypothetical, silent-corruption risk (quantization's own hook does the
   identical patch-scoping trick on the same model) and added an explicit
   `RuntimeError` guard for it. The eager-mode/no-observations error message
   and the chunking/memory documentation you asked about were already present
   from before this round. Validated beyond unit tests: a real Qwen3-8B
   calibration re-run under the new hook reproduced the previously recorded
   fit exactly (`a=3.661323`, `b=9.350536`), so the interception mechanism
   change doesn't alter what's measured.
3. **Recording kernel tile shape.** `block_size`/`query_block_size` are now
   recorded in the exported `sparse_attention_config` (Section 4), so a
   consumer can confirm its kernel's tile shape matches before reusing fitted
   coefficients.
4. **Export placement.** Rather than folding `export.py` into
   `quark/torch/export/` (which, now that point 1 puts this package in
   `contrib/`, would mean a contrib component reaching into a core-owned
   file), we added a test proving `sparse_attention_config` and
   `quantization_config` already compose with no key collision on a single
   `model.config.update()`/`save_pretrained()` round trip. This is the one
   point where we're proposing a different answer than what you asked for
   rather than doing it as suggested; let us know if that reasoning doesn't
   hold up.

## 1. What this is

BLASST ([arXiv:2512.12087](https://arxiv.org/abs/2512.12087)) is a block-sparse attention technique from NVIDIA
(they call it Skip Softmax Attention): at inference time, a kernel skips key
blocks whose contribution to the softmax normalizer is negligible, based on a
scalar threshold. BLASST is already in the
[TokenSpeed kernel](https://github.com/lightseekorg/tokenspeed/pull/1359),
and the [AITER kernel PR](https://github.com/ROCm/aiter/pull/5868) is in
review now.

Before a model can use this at serving time, it needs an offline calibration
step: for a given model, find the `(a, b)` coefficients of
`scale_factor = a * exp(b * target_sparsity)`, so that at serving time a
deployment can pick a human-meaningful `target_sparsity` (e.g. 0.5) and derive
the runtime threshold (`threshold = scale_factor / seq_len`) instead of
hand-tuning a raw threshold per model and context length. This repository's
scope is that calibration step, not the serving kernel itself.

## 2. Where this lives, and why

NVIDIA keeps this same calibration logic in
[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), not in
TensorRT-LLM itself. Quark is AMD's equivalent of Model Optimizer, so the
calibration/PTQ-vs-serving-engine split still matches, but per reviewer
feedback this now lives in `quark/contrib/attention_sparsity/` rather than
`quark/torch/sparsity/`, following the layout
`docs/source/intro_contrib.rst` requires for a `contrib` component (code,
`test/`, and `docs/` together in the component directory; `examples/` in the
shared top-level tree).

```
quark/contrib/attention_sparsity/
├── config.py                 CalibrationConfig, SparseAttentionConfig, schema export
├── api.py                    ModelSparseAttentionCalibrator (top-level entry point)
├── export.py                 write/read sparse_attention_config in checkpoint config.json
├── README.md                 package docs, Quick Start, support/contact
├── methods/
│   └── flash_skip_softmax.py pure-PyTorch reference skip-softmax (calibration-time only)
├── calibration/
│   ├── calibrator.py         DynamicThresholdCalibrator (the curve-fitting logic)
│   └── context.py            per-attention-module softmax hook that feeds the calibrator
├── test/                     unit tests (CI only discovers contrib tests here)
└── docs/
    └── index.rst             auto-indexed landing page, no core doc files touched

examples/contrib/attention_sparsity/
├── calibrate_model.py        runnable argparse driver
├── generate_ruler_data.sh    RULER data generation helper
└── plot_calibration_fit.py   fit visualization
```

This still mirrors NVIDIA Model Optimizer's own `attention_sparsity/` layout
internally (`config.py` / `calibration/` / `methods/` / a top-level API
module) and Quark's own conventions (plain `@dataclass` configs, not
pydantic, matching `quark/torch/pruning/config.py`); only the top-level
parent directory changed.

## 3. Status

Already implemented end to end on a working branch (not yet sent as a PR).
All changes are additive: 22 new files, confined entirely to
`quark/contrib/attention_sparsity/` and `examples/contrib/attention_sparsity/`.
No existing file is touched.

What each file does:

- `config.py`: schema (`CalibrationConfig`, `SparseAttentionAttributeConfig`,
  `SparseAttentionConfig`) and checkpoint-dict shaping.
- `api.py`: top-level entry point (`ModelSparseAttentionCalibrator`),
  orchestrates calibration.
- `export.py`: writes/reads `sparse_attention_config` to/from checkpoint
  `config.json`.
- `methods/flash_skip_softmax.py`: reference, measurement-only skip-softmax
  computation.
- `calibration/calibrator.py`: threshold/sparsity curve fitting.
- `calibration/context.py`: per-attention-module `F.softmax` hook that feeds
  the calibrator during a real forward pass.
- `examples/contrib/attention_sparsity/calibrate_model.py`: runnable
  argparse driver (answers the "no examples driver yet" item from the
  previous round, see Section 6).

Size: about **1000 lines of implementation**, about **1150 lines of tests**
(70 tests), roughly one line of test per line of code.

Calibrated real models (Qwen3-8B, Llama-3.1-8B-Instruct) against real RULER
prompts, round-tripped the checkpoint export/reload, and cross-checked the
calibrated fit against the real production kernel. All 70 tests pass; actual
test execution time is about **2 minutes** (most of a standalone run's wall
time is one-time Python/torch/transformers import overhead paid once per
pytest process, not specific to this addition), negligible next to existing
test directories of similar or larger size.

## 4. Checkpoint hand-off format

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "algorithm": "skip_softmax",
        "targets": ["*"],
        "threshold_scale_factor": {
          "formula": "a * exp(b * target_sparsity)",
          "prefill": {"a": 6.278663, "b": 9.001448},
          "decode": {"a": "...", "b": "..."}
        },
        "target_sparsity": {"prefill": 0.5, "decode": 0.0},
        "ignore": ["model.layers.0.self_attn"],
        "block_size": 64,
        "query_block_size": 128
      }
    },
    "producer": {"name": "quark", "version": "0.x"}
  }
}
```

Deliberately kept structurally identical to NVIDIA's own schema (same key
names and nesting, including the `formula` string and fnmatch `ignore`
list), for compatibility, with no GPU-architecture field: the
threshold/sparsity relationship is a property of the model's
attention-score distribution, not of which accelerator executes it. A
checkpoint calibrated by this tooling
should in principle be readable by NVIDIA-compatible tooling and vice versa.

New since the previous round: `block_size`/`query_block_size` record the
key-block and query-tile size the calibration assumed (analogous to a real
kernel's `BLOCK_N`/`BLOCK_M`), so a consumer can confirm its own kernel's
tiling still matches before reusing these coefficients: a mismatch here
invalidates the fit regardless of how good `r_squared` looked at calibration
time.

## 5. Resolved from the previous round

- **Placement**: resolved by moving to `quark/contrib/attention_sparsity/`
  (Section 2); see "Changes since review" above.
- **Export placement**: `export.py` still writes/reads the checkpoint config
  on its own, separate from `BaseExporter`/`SafetensorsExporter`, since no
  weight tensors are touched here, only a `config.json` metadata key. We
  added a test proving this composes cleanly with the existing quantization
  export path (no key collision between `sparse_attention_config` and
  `quantization_config` on the same `model.config`) as evidence for keeping
  it this way rather than folding it into `quark/torch/export/`. Let us know
  if you'd still rather see it integrated differently.

## 6. Explicitly out of scope here

- **Serving-side consumption** (reading `sparse_attention_config` back out at
  inference time): handled separately as part of integrating BLASST into
  ATOM.
- **A built-in RULER/dataset loader**: the package README documents using the
  RULER benchmark as calibration input; no built-in loader is shipped.

(The `examples/` driver script originally listed here as future work now
exists: `examples/contrib/attention_sparsity/calibrate_model.py`, see
Section 3.)
