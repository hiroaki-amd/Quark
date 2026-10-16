"""Plot a calibration's raw (sparsity, scale_factor) sampling points alongside its fitted
curve, as a sanity check that the fit is a reasonable summary of the underlying data (not,
say, dominated by a handful of outliers, or fit over a oddly-shaped point cloud).

`calibrate_model.py` always saves the raw points this reads and calls this automatically; run
it directly only to replot an existing raw-points file (e.g. with a different `--real_*`
reference point).
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIN_SPARSITY, MAX_SPARSITY = 0.10, 0.90  # DEFAULT_MIN/MAX_SPARSITY in calibrator.py

# Brand-neutral placeholder palette (dataviz skill references/palette.md): sequential blue for
# the sampled points, a warm accent for the fitted curve, a status-red for an optional
# real-kernel reference point.
COLOR_POINTS = "#4C72B0"
COLOR_FIT = "#DD8452"
COLOR_REAL = "#C44E52"
COLOR_MUTED = "#8C8C8C"


def plot_fit(raw_points_path, a, b, r_squared, output_path, title, real_point=None):
    """
    :param str raw_points_path: Path to a JSON file with ``{"columns": [...], "points": [...]}``,
        where each row is either ``(threshold, seq_len, sparsity)`` or ``(scale_factor,
        sparsity)`` (``scale_factor = threshold * seq_len``).
    :param float a: Fitted ``scale_factor = a * exp(b * sparsity)`` coefficient.
    :param float b: Same.
    :param float r_squared: Fit quality, shown in the legend.
    :param str output_path: Where to save the PNG.
    :param str title: Plot title.
    :param tuple[float, float] | None real_point: Optional ``(sparsity, scale_factor)`` to mark
        as a real-kernel reference point (e.g. a measured elision at a specific threshold).
    """
    with open(raw_points_path) as f:
        raw = json.load(f)
    points = np.array(raw["points"], dtype=np.float64)
    if len(raw["columns"]) == 3:
        threshold, seq_len, sparsity = points[:, 0], points[:, 1], points[:, 2]
        scale_factor = threshold * seq_len
    else:
        scale_factor, sparsity = points[:, 0], points[:, 1]

    in_range = (sparsity > MIN_SPARSITY) & (sparsity < MAX_SPARSITY)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    ax.scatter(
        sparsity[~in_range], scale_factor[~in_range],
        s=8, alpha=0.15, color=COLOR_MUTED, linewidths=0,
        label="excluded (outside 0.1-0.9 fit range)",
    )
    ax.scatter(
        sparsity[in_range], scale_factor[in_range],
        s=12, alpha=0.4, color=COLOR_POINTS, linewidths=0,
        label=f"sampled points (n={int(in_range.sum())}, used for fit)",
    )

    xs = np.linspace(0.001, 0.999, 200)
    ax.plot(xs, a * np.exp(b * xs), color=COLOR_FIT, linewidth=2.5,
            label=f"fit: scale_factor = {a:.3f} * exp({b:.3f} * sparsity), r2={r_squared:.3f}")

    if real_point is not None:
        real_sparsity, real_scale_factor = real_point
        ax.scatter(
            [real_sparsity], [real_scale_factor],
            s=140, marker="D", color=COLOR_REAL, zorder=5, edgecolors="white", linewidths=1,
            label=f"real kernel reference (sparsity={real_sparsity:.1%})",
        )

    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("sparsity (fraction of key blocks skipped)")
    ax.set_ylabel("scale_factor = threshold x seq_len (log scale)")
    ax.set_title(title)
    ax.axvline(0.5, color=COLOR_MUTED, linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate("target\nsparsity=0.5", xy=(0.5, 0.02), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=8, color=COLOR_MUTED)

    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="both", alpha=0.15)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return {"n_total": len(points), "n_in_range": int(in_range.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_points", required=True, help="Path to the raw points JSON")
    parser.add_argument("--fit", required=True, help="Path to the calibrate_model.py output JSON")
    parser.add_argument("--output", required=True, help="Path to save the plot PNG to")
    parser.add_argument("--real_sparsity", type=float, default=None, help="Optional real-kernel reference sparsity")
    parser.add_argument("--real_threshold", type=float, default=None, help="Optional real-kernel reference threshold")
    parser.add_argument("--real_seq_len", type=int, default=None, help="Optional real-kernel reference seq_len")
    args = parser.parse_args()

    with open(args.fit) as f:
        fit = json.load(f)
    a = fit["threshold_scale_factor"]["prefill"]["a"]
    b = fit["threshold_scale_factor"]["prefill"]["b"]
    r_squared = fit.get("r_squared", float("nan"))

    real_point = None
    if args.real_sparsity is not None and args.real_threshold is not None and args.real_seq_len is not None:
        real_point = (args.real_sparsity, args.real_threshold * args.real_seq_len)

    title = f"{fit.get('model', 'model')} BLASST calibration: sampled points vs. fitted curve"
    stats = plot_fit(args.raw_points, a, b, r_squared, args.output, title, real_point=real_point)
    print(f"Saved plot to {args.output}")
    print(f"Total points: {stats['n_total']}, in fit range: {stats['n_in_range']}")


if __name__ == "__main__":
    main()
