"""EXPERIMENTAL BLASST calibration driver modeling AITER's real GQA head-packing skip
granularity -- NOT part of the public calibration API, NOT used by
`ModelSparseAttentionCalibrator`. See `compute_skip_sparsity_multi_threshold_head_packed`
in `quark/torch/sparsity/attention_sparsity/methods/flash_skip_softmax.py` for why: AITER's
unified_2d kernel packs a GQA group's query heads into a single physical ``block_m``-row
tile, so a key block is only skipped if *every* row across *all* heads in the group agrees,
not just within one head's own query tile (which is all the standard calibration models).

This script otherwise mirrors `calibrate_model.py` (same RULER sampling, same always-on
raw-point capture and plot), but reimplements the calibration driver loop directly instead
of calling `ModelSparseAttentionCalibrator.calibrate_model`, since the head-packed function
is not wired into that public API.

Example (Qwen3-32B, group_size=8 since num_attention_heads=64 and num_key_value_heads=8):
    python calibrate_model_head_packed.py \\
        --model Qwen/Qwen3-32B \\
        --ruler_dir /path/to/ruler_data/qwen3_32b \\
        --context_lengths 32713 \\
        --group_size 8 \\
        --output qwen3_32b_head_packed_calibration.json

`--group_size` must match the model's real GQA group size
(``num_attention_heads // num_key_value_heads``, from its HF config).
"""

import argparse
import contextlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from quark.torch.sparsity.attention_sparsity.calibration.calibrator import (
    DEFAULT_MAX_SPARSITY,
    DEFAULT_MIN_SPARSITY,
    DynamicThresholdCalibrator,
)
from quark.torch.sparsity.attention_sparsity.config import DEFAULT_THRESHOLD_TRIALS
from quark.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import (
    compute_skip_sparsity_multi_threshold_head_packed,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibrate_model import _ProgressList, load_ruler_samples
from plot_calibration_fit import plot_fit


def r_squared_logspace(raw_points: list[tuple[float, float]], a: float, b: float) -> float:
    """Recomputes the log-space r_squared `DynamicThresholdCalibrator.fit` would have
    reported, from the pooled (scale_factor, sparsity) points this script records."""
    points = np.array(raw_points, dtype=np.float64)
    scale_factor, sparsity = points[:, 0], points[:, 1]
    mask = (sparsity > DEFAULT_MIN_SPARSITY) & (sparsity < DEFAULT_MAX_SPARSITY)
    log_scale_factor = np.log(scale_factor[mask])
    predicted = math.log(a) + b * sparsity[mask]
    ss_res = float(np.sum((log_scale_factor - predicted) ** 2))
    ss_tot = float(np.sum((log_scale_factor - log_scale_factor.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


@contextlib.contextmanager
def _record_head_packed_sparsity(calibrator, threshold_trials, block_size, block_m, group_size):
    """Mirrors `record_softmax_sparsity` (quark/torch/sparsity/attention_sparsity/calibration/
    context.py), but calls `compute_skip_sparsity_multi_threshold_head_packed` instead of the
    standard (non-head-packing-aware) function. Averages sparsity across every call
    intercepted within this context (all layers of one forward pass, sharing one KV length)
    and records a single observation per threshold at exit -- same per-forward-pass (not
    per-layer) aggregation as the standard driver, for the same reason: per-layer sparsity
    variance would otherwise pollute the fit."""
    original_softmax = F.softmax
    accumulated = [[] for _ in threshold_trials]
    seq_lens_seen = []

    def sparse_softmax(input, dim=None, *args, **kwargs):
        if input.ndim == 4:
            seq_lens_seen.append(input.shape[-1])
            sparsities = compute_skip_sparsity_multi_threshold_head_packed(
                input, threshold_trials, block_size=block_size, block_m=block_m, group_size=group_size
            )
            for values, sparsity in zip(accumulated, sparsities):
                values.append(sparsity)
        return original_softmax(input, dim, *args, **kwargs)

    F.softmax = sparse_softmax
    try:
        yield
    finally:
        F.softmax = original_softmax
        if seq_lens_seen:
            seq_len = seq_lens_seen[0]
            averaged = [sum(values) / len(values) for values in accumulated]
            calibrator.add_observations(thresholds=threshold_trials, seq_len=seq_len, sparsities=averaged)


def run_chunked_prefill_with_head_packed_observation(
    model, input_ids, calibrator, threshold_trials, block_size, block_m, group_size, chunk_size
):
    """Mirrors `ModelSparseAttentionCalibrator._run_chunked_prefill`: one
    `_record_head_packed_sparsity` context per forward pass (one prefill chunk), so each
    chunk's averaged-across-layers observation is recorded at that chunk's own (growing) KV
    length."""
    seq_len = input_ids.shape[-1]
    if chunk_size <= 0 or seq_len <= chunk_size:
        with _record_head_packed_sparsity(calibrator, threshold_trials, block_size, block_m, group_size):
            model(input_ids, use_cache=True)
        return
    past_key_values = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        with _record_head_packed_sparsity(calibrator, threshold_trials, block_size, block_m, group_size):
            outputs = model(input_ids[:, start:end], past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF repo id or local path of the model to calibrate")
    parser.add_argument(
        "--ruler_dir", required=True, help="Directory containing a ctx_<N>/ subdir for each --context_lengths entry"
    )
    parser.add_argument(
        "--context_lengths", required=True,
        help="Comma-separated context lengths to calibrate at, e.g. 32713 or 4096,8192,16384,32768,65536,131072",
    )
    parser.add_argument(
        "--num_samples", type=int, default=24,
        help="Number of RULER samples to use PER context length (the same count at every one of them)",
    )
    parser.add_argument("--block_size", type=int, default=64, help="Key-block skip-decision granularity")
    parser.add_argument(
        "--block_m", type=int, default=128,
        help="Real physical query-tile size covering all heads in a GQA group together",
    )
    parser.add_argument(
        "--group_size", type=int, required=True,
        help="GQA group size (num_attention_heads // num_key_value_heads, from the model's HF config)",
    )
    parser.add_argument("--chunk_size", type=int, default=2048, help="Eager-mode chunked-prefill chunk size")
    parser.add_argument("--output", required=True, help="Path to write the fitted (a, b) config JSON to")
    args = parser.parse_args()
    context_lengths = [int(c) for c in args.context_lengths.split(",")]

    t0 = time.time()
    print(f"[{time.time() - t0:7.1f}s] Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config._attn_implementation = "eager"
    model.eval()
    print(f"[{time.time() - t0:7.1f}s] Model loaded.", flush=True)

    print(f"[{time.time() - t0:7.1f}s] Loading RULER samples...", flush=True)
    input_ids_list = _ProgressList(
        load_ruler_samples(args.ruler_dir, tokenizer, context_lengths, args.num_samples)
    )
    print(f"[{time.time() - t0:7.1f}s] Total samples: {len(input_ids_list)}", flush=True)

    calibrator = DynamicThresholdCalibrator()
    print(
        f"[{time.time() - t0:7.1f}s] Calibrating with head-packing (block_size={args.block_size}, "
        f"block_m={args.block_m}, group_size={args.group_size}, chunk_size={args.chunk_size})...",
        flush=True,
    )
    with torch.no_grad():
        for input_ids in input_ids_list:
            run_chunked_prefill_with_head_packed_observation(
                model, input_ids, calibrator, DEFAULT_THRESHOLD_TRIALS,
                args.block_size, args.block_m, args.group_size, args.chunk_size,
            )

    fit_result = calibrator.fit(fit_logspace=True)
    a, b = fit_result.a, fit_result.b
    raw_points = list(zip(calibrator._scale_factors, calibrator._sparsities))
    r_squared = r_squared_logspace(raw_points, a, b)

    print(f"\n[{time.time() - t0:7.1f}s] === Head-packed calibration result ({args.model}) ===", flush=True)
    print(f"prefill: a={a:.6f}, b={b:.6f}, r_squared={r_squared:.4f}, n={len(raw_points)}", flush=True)

    with open(args.output, "w") as f:
        json.dump(
            {
                "model": args.model,
                "block_size": args.block_size,
                "block_m": args.block_m,
                "group_size": args.group_size,
                "chunk_size": args.chunk_size,
                "context_lengths": context_lengths,
                "num_samples_per_bin": args.num_samples,
                "num_samples": len(input_ids_list),
                "threshold_scale_factor": {"prefill": {"a": a, "b": b}},
                "r_squared": r_squared,
                "num_raw_points": len(raw_points),
            },
            f,
            indent=2,
        )
    print(f"Saved to {args.output}")

    raw_points_path = args.output.rsplit(".", 1)[0] + "_raw_points.json"
    with open(raw_points_path, "w") as f:
        json.dump({"columns": ["scale_factor", "sparsity"], "points": raw_points}, f)
    print(f"Saved {len(raw_points)} raw points to {raw_points_path}")

    plot_path = args.output.rsplit(".", 1)[0] + ".png"
    stats = plot_fit(
        raw_points_path, a, b, r_squared, plot_path,
        title=f"{args.model} BLASST calibration, head-packing EXPERIMENT\n"
        f"(ctx={context_lengths}, block_size={args.block_size}, block_m={args.block_m}, group_size={args.group_size})",
    )
    print(f"Saved plot to {plot_path} (n={stats['n_total']}, in fit range={stats['n_in_range']})")


if __name__ == "__main__":
    main()
