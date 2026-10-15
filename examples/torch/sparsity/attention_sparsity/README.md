# BLASST attention-sparsity calibration examples

Produces the `(a, b)` threshold-scale-factor coefficients described in
`quark/torch/sparsity/attention_sparsity/README.md`, by calibrating a real model
against real RULER prompts via the public
`ModelSparseAttentionCalibrator.calibrate_model` API.

## Prerequisites

- A CUDA/ROCm GPU with enough memory to load the target model in bf16.
- Python with PyTorch already installed for your accelerator (this example does not install
  PyTorch for you; see Quark's own install docs if you need that first).

## Setup

1. Clone this repo (checking out this branch, if the change isn't merged yet) and install
   Quark from source as an editable install:

   ```bash
   git clone https://github.com/hiroaki-amd/Quark.git
   cd Quark
   git checkout blasst-head-packing-experiment
   pip install -e .
   ```

2. Clone [NVIDIA/RULER](https://github.com/NVIDIA/RULER) (used only to generate the
   calibration prompts; this example does not vendor it) and install the few dependencies
   its data-generation script (`scripts/data/prepare.py`) needs. Its `docker/requirements.txt`
   covers RULER's full benchmark suite (vllm, nemo-toolkit, etc.) -- generation itself only
   needs:

   ```bash
   git clone https://github.com/NVIDIA/RULER.git
   pip install pyyaml numpy tenacity wonderwords
   ```

## Usage

1. From `examples/torch/sparsity/attention_sparsity/` in this repo, generate RULER data at
   every context length you want to calibrate across (a `ctx_<N>/` subdirectory per length,
   all under one parent directory):

   ```bash
   RULER_DATA_DIR=/path/to/RULER/scripts/data
   for CTX in 4096 8192 16384 32768 65536 131072; do
     bash generate_ruler_data.sh Qwen/Qwen3-32B "$CTX" "./ruler_data/qwen3_32b/ctx_${CTX}"
   done
   ```

   RULER generation is tokenizer-dependent, not weights-dependent: data generated for one
   model is valid for any other model sharing the same tokenizer (e.g. Qwen3-8B and
   Qwen3-32B), but not across tokenizer families.

2. Run calibration:

   ```bash
   python calibrate_model.py \
       --model Qwen/Qwen3-32B \
       --ruler_dir ./ruler_data/qwen3_32b \
       --context_lengths 4096,8192,16384,32768,65536,131072 \
       --num_samples 8 \
       --block_size 64 --query_block_size 128 \
       --output qwen3_32b_calibration.json
   ```

   `--num_samples` is the count used *at each* context length in `--context_lengths`, not a
   total -- an earlier version of this calibration sampled unevenly across context lengths
   (undersampling the longest ones specifically) and that skewed the fit; always keep this
   count equal across every length actually used.

   `--block_size`/`--query_block_size` should match your target kernel's real tile shape
   (see `quark/torch/sparsity/attention_sparsity/README.md`'s `block_size` /
   `query_block_size` section); the defaults above match a 64x128 skip-decision tile.

   A single context length works too -- pass one value to `--context_lengths` (e.g. to
   match one specific real-kernel benchmark point rather than calibrate for general use).

   If you hit the kernel extension JIT failure below, apply that workaround first.

## Known environment issue: kernel extension JIT failure

Importing `quark.torch.sparsity.attention_sparsity` pulls in all of `quark.torch`,
including an optional hardware-emulation kernel extension
(`quark.torch.kernel.hw_emulation.extensions`) unrelated to attention sparsity. On a
machine without a prebuilt wheel for it and without a working JIT toolchain for your
GPU architecture, this raises:

```
RuntimeError: Stable-ABI hw_emulation extension could not be loaded and JIT compilation failed.
```

This is a pre-existing `quark.torch` import-chain issue, not specific to attention
sparsity. If you hit it and don't need that extension, stub it out before importing
anything from `quark`, at the top of your script:

```python
import sys
import types

_ext_stub = types.ModuleType("quark.torch.kernel.hw_emulation.extensions")
_ext_stub.kernel_ext = None
sys.modules["quark.torch.kernel.hw_emulation.extensions"] = _ext_stub

def _unavailable(*args, **kwargs):
    raise RuntimeError("mx hip kernel unavailable (stubbed)")

_hip_stub = types.ModuleType("quark.torch.kernel.mx.hip")
_hip_stub.dq_mxfp4_hip = _unavailable
_hip_stub.qdq_mxfp4_hip = _unavailable
sys.modules["quark.torch.kernel.mx.hip"] = _hip_stub
```

## Worked examples

Multi-bin (recommended for general use -- calibrated across the full context range a
deployment is likely to see, rather than one specific length):

| model | context lengths | block_size | query_block_size | a | b | r_squared |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 4096-131072 (6 bins) | 64 | 128 | 1.331873 | 11.245634 | 0.913 |
| Qwen3-32B | 4096-131072 (6 bins) | 64 | 128 | 0.642108 | 11.779660 | 0.912 |
| Llama-3.1-8B | 4096-131072 (6 bins) | 64 | 128 | 0.209495 | 9.693106 | 0.918 |

All three are base (non-instruct) checkpoints, 8 RULER samples per bin (4096, 8192, 16384,
32768, 65536, 131072), round-robined across its 13 tasks independently at each bin,
`target_sparse_ratio={"prefill": 0.5, "decode": 0.0}`, `fit_logspace=True`. Qwen3-32B used
`chunk_size=512` (vs. 2048 for the other two) to fit its larger per-layer score tensor (more
attention heads) in memory at 131072 context; chunk size does not bias the measurement (see
`quark/torch/sparsity/attention_sparsity/README.md`'s chunked-prefill discussion), only
memory use. These coefficients are specific to the model, the sampled prompts, and the tile
shape used; recalibrate for any other model or kernel tiling rather than reusing them
directly.

## EXPERIMENTAL: GQA head-packing (this branch only, not in the main PR)

`calibrate_model_head_packed.py` is the same driver, but models a real tiled kernel's GQA
head-packing (a GQA group's query heads share one physical tile, so a key block is only
skippable if every row across every head in the group agrees -- not just within one head's
own tile, which is all `calibrate_model.py` models). Same RULER data, same always-on
raw-point capture and plot:

```bash
python calibrate_model_head_packed.py \
    --model Qwen/Qwen3-32B \
    --ruler_dir ./ruler_data/qwen3_32b \
    --context_lengths 32713 \
    --group_size 8 \
    --output qwen3_32b_head_packed_calibration.json
```

`--group_size` must match the model's real GQA group size (`num_attention_heads //
num_key_value_heads`, from its HF config; 8 for Qwen3-32B).
