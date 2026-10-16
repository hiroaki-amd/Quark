"""BLASST (skip-softmax) attention-sparsity calibration driver.

Uses only the public API documented in this repo's
`quark/torch/sparsity/attention_sparsity/README.md` Quick Start section:
`ModelSparseAttentionCalibrator.calibrate_model`. Feeds it RULER prompts (raw
`input + answer_prefix` text, no chat template -- appropriate for calibration
regardless of whether the checkpoint is a base or instruct model), round-robined
across RULER's 13 tasks independently at each context length in `--context_lengths`,
with the *same* sample count at every one of them (never skew toward -- or away
from -- any particular context length: an earlier version of this calibration
undersampled long context specifically, which is exactly where getting the
threshold right matters most).

Example:
    python calibrate_model.py \\
        --model Qwen/Qwen3-32B \\
        --ruler_dir /path/to/ruler_data/qwen3_32b \\
        --context_lengths 4096,8192,16384,32768,65536,131072 \\
        --block_size 64 --query_block_size 128 \\
        --output qwen3_32b_calibration.json

`--ruler_dir` must contain a `ctx_<N>/` subdirectory (each with RULER's 13 task
subdirectories) for every `N` in `--context_lengths` -- generate each with
`generate_ruler_data.sh` in this directory before running this.

Always saves the raw (threshold, seq_len, sparsity) observations behind the fit (next to
`--output`, suffixed `_raw_points.json`) and a sampled-points-vs-fitted-curve plot (suffixed
`.png`) -- a fit this cheap to produce is also cheap to sanity-check, and a plot catches a bad
fit (an unrepresentative point cloud, a curve dominated by outliers) that the `(a, b)` numbers
alone would hide. See `plot_calibration_fit.py` to regenerate just the plot later.
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import quark.torch.sparsity.attention_sparsity.calibration.calibrator as calibrator_module
from quark.torch.sparsity.attention_sparsity.api import ModelSparseAttentionCalibrator
from quark.torch.sparsity.attention_sparsity.calibration.calibrator import DEFAULT_MAX_SPARSITY, DEFAULT_MIN_SPARSITY
from quark.torch.sparsity.attention_sparsity.config import CalibrationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from plot_calibration_fit import plot_fit

RAW_POINTS: list[tuple[float, int, float]] = []
_original_add_observation = calibrator_module.DynamicThresholdCalibrator.add_observation


def _patched_add_observation(self, threshold: float, seq_len: int, sparsity: float) -> None:
    RAW_POINTS.append((threshold, seq_len, sparsity))
    return _original_add_observation(self, threshold, seq_len, sparsity)


calibrator_module.DynamicThresholdCalibrator.add_observation = _patched_add_observation


def r_squared_logspace(raw_points: list[tuple[float, int, float]], a: float, b: float) -> float:
    """Recomputes the log-space r_squared `DynamicThresholdCalibrator.fit` would have reported,
    from the raw points this module's monkeypatch captured (the public API does not expose it)."""
    points = np.array(raw_points, dtype=np.float64)
    threshold, seq_len, sparsity = points[:, 0], points[:, 1], points[:, 2]
    mask = (sparsity > DEFAULT_MIN_SPARSITY) & (sparsity < DEFAULT_MAX_SPARSITY)
    log_scale_factor = np.log(threshold[mask] * seq_len[mask])
    predicted = math.log(a) + b * sparsity[mask]
    ss_res = float(np.sum((log_scale_factor - predicted) ** 2))
    ss_tot = float(np.sum((log_scale_factor - log_scale_factor.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def load_ruler_samples_for_one_context_length(
    ctx_dir: str, tokenizer, context_length: int, num_samples: int
) -> list[torch.Tensor]:
    """Round-robin-across-tasks sampling of RULER's 13 task subdirectories under `ctx_dir`,
    each truncated to `context_length` tokens."""
    jsonl_files = sorted(glob.glob(os.path.join(ctx_dir, "*/validation.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No RULER JSONL files found in {ctx_dir} -- generate it first with generate_ruler_data.sh"
        )

    inputs_by_task = []
    for jf in jsonl_files:
        task_inputs = []
        with open(jf) as f:
            for line in f:
                entry = json.loads(line)
                task_inputs.append(entry["input"] + entry["answer_prefix"])
        inputs_by_task.append(task_inputs)

    n = min(num_samples, sum(len(t) for t in inputs_by_task))
    print(
        f"  ctx={context_length}: {sum(len(t) for t in inputs_by_task)} samples available across "
        f"{len(inputs_by_task)} tasks, using {n}"
    )

    selected = []
    task_idx = 0
    offsets = [0] * len(inputs_by_task)
    while len(selected) < n:
        if offsets[task_idx] < len(inputs_by_task[task_idx]):
            selected.append(inputs_by_task[task_idx][offsets[task_idx]])
            offsets[task_idx] += 1
        task_idx = (task_idx + 1) % len(inputs_by_task)

    chunks = []
    for text in selected:
        ids = tokenizer.encode(text)
        if len(ids) > context_length:
            ids = ids[:context_length]
        chunks.append(torch.tensor(ids).unsqueeze(0).cuda())
    return chunks


def load_ruler_samples(
    ruler_dir: str, tokenizer, context_lengths: list[int], num_samples_per_bin: int
) -> list[torch.Tensor]:
    """Loads `num_samples_per_bin` RULER samples at *each* context length in `context_lengths`
    (the same count at every one of them -- see the module docstring for why) and concatenates
    them, in `context_lengths` order, into a single list."""
    all_chunks = []
    for context_length in context_lengths:
        ctx_dir = os.path.join(ruler_dir, f"ctx_{context_length}")
        all_chunks.extend(
            load_ruler_samples_for_one_context_length(ctx_dir, tokenizer, context_length, num_samples_per_bin)
        )
    return all_chunks


class _ProgressList(list):
    """A plain list that prints flushed per-item timing/ETA as it is iterated -- calibration
    samples at long context can each take a while, so this gives visibility into progress on a
    multi-sample, potentially multi-hour run."""

    def __iter__(self):
        total = len(self)
        start = time.time()
        for i, item in enumerate(super().__iter__()):
            elapsed = time.time() - start
            eta = (elapsed / i * (total - i)) if i else float("nan")
            print(f"[{elapsed:7.1f}s] sample {i + 1}/{total} (seq_len={item.shape[-1]}), ETA {eta:.1f}s", flush=True)
            yield item
        print(f"[{time.time() - start:7.1f}s] all {total} samples done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF repo id or local path of the model to calibrate")
    parser.add_argument(
        "--ruler_dir", required=True, help="Directory containing a ctx_<N>/ subdir for each --context_lengths entry"
    )
    parser.add_argument(
        "--context_lengths", required=True,
        help="Comma-separated context lengths to calibrate at, e.g. 4096,8192,16384,32768,65536,131072",
    )
    parser.add_argument(
        "--num_samples", type=int, default=8,
        help="Number of RULER samples to use PER context length (the same count at every one of them)",
    )
    parser.add_argument("--block_size", type=int, default=64, help="Key-block skip-decision granularity")
    parser.add_argument("--query_block_size", type=int, default=128, help="Query-tile skip-decision granularity")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Eager-mode chunked-prefill chunk size")
    parser.add_argument("--target_sparsity", type=float, default=0.5, help="Target prefill sparsity to calibrate for")
    parser.add_argument("--output", required=True, help="Path to write the fitted (a, b) config JSON to")
    args = parser.parse_args()
    context_lengths = [int(c) for c in args.context_lengths.split(",")]

    t0 = time.time()
    print(f"[{time.time() - t0:7.1f}s] Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"[{time.time() - t0:7.1f}s] Model loaded.", flush=True)

    print(f"[{time.time() - t0:7.1f}s] Loading RULER samples...", flush=True)
    input_ids_list = _ProgressList(
        load_ruler_samples(args.ruler_dir, tokenizer, context_lengths, args.num_samples)
    )
    print(f"[{time.time() - t0:7.1f}s] Total samples: {len(input_ids_list)}", flush=True)

    config = CalibrationConfig(
        target_sparse_ratio={"prefill": args.target_sparsity, "decode": 0.0},
        block_size=args.block_size,
        query_block_size=args.query_block_size,
        chunk_size=args.chunk_size,
        fit_logspace=True,
    )
    print(
        f"[{time.time() - t0:7.1f}s] Calibrating (block_size={args.block_size}, "
        f"query_block_size={args.query_block_size}, chunk_size={args.chunk_size})...",
        flush=True,
    )
    calibrator = ModelSparseAttentionCalibrator(config)
    group = calibrator.calibrate_model(model, input_ids_list)
    fit_result = group.threshold_scale_factor["prefill"]
    r_squared = r_squared_logspace(RAW_POINTS, fit_result.a, fit_result.b)

    print(f"\n[{time.time() - t0:7.1f}s] === Calibration result ({args.model}) ===", flush=True)
    print(f"prefill: a={fit_result.a:.6f}, b={fit_result.b:.6f}, r_squared={r_squared:.4f}", flush=True)

    with open(args.output, "w") as f:
        json.dump(
            {
                "model": args.model,
                "block_size": args.block_size,
                "query_block_size": args.query_block_size,
                "chunk_size": args.chunk_size,
                "context_lengths": context_lengths,
                "num_samples_per_bin": args.num_samples,
                "num_samples": len(input_ids_list),
                "threshold_scale_factor": {"prefill": {"a": fit_result.a, "b": fit_result.b}},
                "r_squared": r_squared,
                "num_raw_points": len(RAW_POINTS),
                "target_sparsity": {"prefill": args.target_sparsity, "decode": 0.0},
            },
            f,
            indent=2,
        )
    print(f"Saved to {args.output}")

    raw_points_path = args.output.rsplit(".", 1)[0] + "_raw_points.json"
    with open(raw_points_path, "w") as f:
        json.dump({"columns": ["threshold", "seq_len", "sparsity"], "points": RAW_POINTS}, f)
    print(f"Saved {len(RAW_POINTS)} raw points to {raw_points_path}")

    plot_path = args.output.rsplit(".", 1)[0] + ".png"
    stats = plot_fit(
        raw_points_path, fit_result.a, fit_result.b, r_squared, plot_path,
        title=f"{args.model} BLASST calibration: sampled points vs. fitted curve\n"
        f"(ctx={context_lengths}, block_size={args.block_size}, query_block_size={args.query_block_size})",
    )
    print(f"Saved plot to {plot_path} (n={stats['n_total']}, in fit range={stats['n_in_range']})")


if __name__ == "__main__":
    main()
