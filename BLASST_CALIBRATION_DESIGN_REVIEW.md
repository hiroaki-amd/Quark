# BLASST attention-sparsity calibration in Quark: design for review

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
TensorRT-LLM itself. Quark is AMD's equivalent of Model Optimizer, and
already separates calibration/PTQ tooling from the serving engine (ATOM) the
same way for quantization, so we followed that same split: all of this lives
in Quark. Concretely, we are proposing a new top-level `quark/torch/sparsity/`
directory, alongside the existing `quark/torch/quantization/` and
`quark/torch/pruning/`, with `attention_sparsity/` as its first subpackage
(room for other sparsity techniques to land as siblings later, rather than
overloading attention sparsity's own directory).

```
quark/torch/sparsity/attention_sparsity/
├── config.py                 CalibrationConfig, SparseAttentionConfig, schema export
├── api.py                    ModelSparseAttentionCalibrator (top-level entry point)
├── export.py                 write/read sparse_attention_config in checkpoint config.json
├── methods/
│   └── flash_skip_softmax.py pure-PyTorch reference skip-softmax (calibration-time only)
└── calibration/
    ├── calibrator.py         DynamicThresholdCalibrator (the curve-fitting logic)
    └── context.py            softmax monkeypatch that feeds the calibrator
```

This mirrors NVIDIA Model Optimizer's own `attention_sparsity/` layout
(`config.py` / `calibration/` / `methods/` / a top-level API module), and
follows Quark's own conventions rather than NVIDIA's: plain `@dataclass`
configs (not pydantic), matching `quark/torch/pruning/config.py`.

## 3. Status

Already implemented end to end on a working branch (not yet sent as a PR).
All changes are additive: 17 new files, confined entirely to
`quark/torch/sparsity/attention_sparsity/` and its tests under
`test/test_for_torch/sparsity/`. No existing file is touched.

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
- `calibration/context.py`: `F.softmax` hook that feeds the calibrator
  during a real forward pass.

Size: about **830 lines of implementation**, about **840 lines of tests**
(56 tests) -- roughly one line of test per line of code.

Calibrated real models (Qwen3-8B, Llama-3.1-8B-Instruct) against real RULER
prompts, round-tripped the checkpoint export/reload, and cross-checked the
calibrated fit against the real production kernel. All 56 tests pass; actual
test execution time is about **5 seconds** (most of a standalone run's wall
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
        "ignore": ["model.layers.0.self_attn"]
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

## 5. Open for discussion

- Does the package placement and layout above match how Quark would want a
  new optimization technique organized, or is there an existing convention we
  should follow more closely?
- Right now, `export.py` writes and reads the checkpoint config on its own,
  separate from `BaseExporter`/`SafetensorsExporter` (the existing
  quantization export path). We think that is fine, since no weight tensors
  are touched here, only a `config.json` metadata key. Does that match how
  you would want it done, or should it hook into the existing exporter
  framework some other way?

## 6. Explicitly out of scope here

- **Serving-side consumption** (reading `sparse_attention_config` back out at
  inference time): handled separately as part of integrating BLASST into
  ATOM.
- **A built-in RULER/dataset loader**: the `README.md` will include an
  example of using the RULER benchmark as calibration input.
- **An `examples/` driver script**: quantization and pruning each have a
  runnable argparse-based driver under `examples/torch/language_modeling/...`.
  This package only has the package itself under
  `quark/torch/sparsity/attention_sparsity/`, with usage documented in its
  own package README; no `examples/attention_sparsity/` driver script yet
  (mirroring NVIDIA's own `hf_sa.py` driver, if added). Can be added later to
  match convention if useful.
