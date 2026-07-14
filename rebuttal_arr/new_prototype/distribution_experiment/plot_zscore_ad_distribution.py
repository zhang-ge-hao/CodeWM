#!/usr/bin/env python3
"""Plot distributions of per-case mean z-score and Anderson-Darling statistic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    ROOT
    / "data"
    / "distribution_consistency"
    / "rw100-z4-all1620-random-seeds-v1"
)
DEFAULT_INPUT = DEFAULT_RUN_ROOT / "results.jsonl"
DEFAULT_OUTPUT = DEFAULT_RUN_ROOT / "zscore_ad_distributions.png"

SCHEME_ORDER = ("sweet", "wllm", "synthid")
SCHEME_LABELS = {"sweet": "SWEET", "wllm": "WLLM", "synthid": "SynthID"}
SCHEME_COLORS = {
    "sweet": "#D55E00",
    "wllm": "#0072B2",
    "synthid": "#009E73",
}
OVERALL_COLOR = "#262626"
HISTOGRAM_COLOR = "#B8BDC5"
GRID_COLOR = "#D8DADD"
CRITICAL_15_COLOR = "#7A5195"
CRITICAL_5_COLOR = "#C23B32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--bins", type=int, default=80)
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            for field in ("sample_mean", "paper_ad_statistic", "watermark"):
                if field not in row:
                    raise ValueError(f"{path}:{line_number} lacks {field}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no results in {path}")
    return rows


def gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """A small NumPy-only Gaussian KDE using Silverman's bandwidth rule."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.zeros_like(grid)
    standard_deviation = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    robust_scale = min(standard_deviation, iqr / 1.349) if iqr > 0 else standard_deviation
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = max(abs(float(np.mean(values))) * 1e-3, 1e-6)
    bandwidth = max(0.9 * robust_scale * values.size ** (-1 / 5), 1e-6)
    offsets = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * offsets**2).mean(axis=1) / (
        bandwidth * np.sqrt(2 * np.pi)
    )


def padded_range(values: np.ndarray, fraction: float = 0.045) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    padding = max((maximum - minimum) * fraction, 1e-6)
    return minimum - padding, maximum + padding


def empirical_cdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(tuple(values), dtype=float))
    probabilities = np.arange(1, ordered.size + 1) / ordered.size
    return ordered, probabilities


def style_axis(axis: object) -> None:
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.65, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A8A8A")
    axis.spines["bottom"].set_color("#8A8A8A")
    axis.tick_params(labelsize=9, colors="#333333")


def add_distribution_panel(
    axis: object,
    values: np.ndarray,
    values_by_scheme: dict[str, np.ndarray],
    *,
    bins: int,
    xlabel: str,
    panel_label: str,
    critical_values: tuple[tuple[float, str, str], ...] = (),
) -> None:
    x_min, x_max = padded_range(values)
    grid = np.linspace(x_min, x_max, 700)
    bin_edges = np.linspace(float(np.min(values)), float(np.max(values)), bins + 1)
    axis.hist(
        values,
        bins=bin_edges,
        density=True,
        color=HISTOGRAM_COLOR,
        alpha=0.58,
        edgecolor="white",
        linewidth=0.35,
        label=f"All cases (n={values.size:,})",
    )
    axis.plot(
        grid,
        gaussian_kde(values, grid),
        color=OVERALL_COLOR,
        linewidth=2.0,
        label="All-case KDE",
        zorder=5,
    )
    for scheme in SCHEME_ORDER:
        scheme_values = values_by_scheme[scheme]
        axis.plot(
            grid,
            gaussian_kde(scheme_values, grid),
            color=SCHEME_COLORS[scheme],
            linewidth=1.55,
            alpha=0.95,
            label=f"{SCHEME_LABELS[scheme]} (n={scheme_values.size:,})",
        )

    mean = float(np.mean(values))
    median = float(np.median(values))
    axis.axvline(mean, color="#555555", linestyle="--", linewidth=1.1)
    axis.axvline(median, color="#555555", linestyle=":", linewidth=1.2)
    for value, label, color in critical_values:
        axis.axvline(value, color=color, linestyle="--", linewidth=1.35, zorder=6)
        axis.text(
            value,
            0.985,
            label,
            transform=axis.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8.2,
            color=color,
        )

    q05, q25, q75, q95 = np.quantile(values, [0.05, 0.25, 0.75, 0.95])
    summary = (
        f"mean {mean:.3f}   median {median:.3f}\n"
        f"IQR [{q25:.3f}, {q75:.3f}]   5–95% [{q05:.3f}, {q95:.3f}]"
    )
    axis.text(
        0.985,
        0.94,
        summary,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        color="#303030",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#C8C8C8"},
    )
    axis.set_xlim(x_min, x_max)
    axis.set_xlabel(xlabel, fontsize=10.5)
    axis.set_ylabel("Probability density", fontsize=10.5)
    axis.set_title(panel_label, loc="left", fontsize=11.5, fontweight="bold")
    style_axis(axis)


def add_ecdf_panel(
    axis: object,
    values_by_scheme: dict[str, np.ndarray],
    *,
    xlabel: str,
    panel_label: str,
    overall_values: np.ndarray,
    critical_values: tuple[tuple[float, str, str], ...] = (),
) -> None:
    overall_x, overall_y = empirical_cdf(overall_values)
    axis.step(
        overall_x,
        overall_y,
        where="post",
        color=OVERALL_COLOR,
        linewidth=2.0,
        label="All cases",
    )
    for scheme in SCHEME_ORDER:
        x_values, y_values = empirical_cdf(values_by_scheme[scheme])
        axis.step(
            x_values,
            y_values,
            where="post",
            color=SCHEME_COLORS[scheme],
            linewidth=1.55,
            alpha=0.95,
            label=SCHEME_LABELS[scheme],
        )
    for value, label, color in critical_values:
        axis.axvline(value, color=color, linestyle="--", linewidth=1.35)
        accepted = int(np.sum(overall_values <= value))
        axis.text(
            value,
            0.035,
            f"{label}: {accepted:,}/{overall_values.size:,}",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8.2,
            color=color,
        )
    axis.set_xlim(*padded_range(overall_values))
    axis.set_ylim(0, 1.005)
    axis.set_xlabel(xlabel, fontsize=10.5)
    axis.set_ylabel("Empirical cumulative probability", fontsize=10.5)
    axis.set_title(panel_label, loc="left", fontsize=11.5, fontweight="bold")
    style_axis(axis)


def plot(rows: list[dict[str, object]], output: Path, *, bins: int, dpi: int) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "codewm-matplotlib")
    )
    import matplotlib.pyplot as plt

    means = np.asarray([float(row["sample_mean"]) for row in rows])
    statistics = np.asarray([float(row["paper_ad_statistic"]) for row in rows])
    mean_by_scheme = {
        scheme: np.asarray(
            [float(row["sample_mean"]) for row in rows if row["watermark"] == scheme]
        )
        for scheme in SCHEME_ORDER
    }
    statistic_by_scheme = {
        scheme: np.asarray(
            [
                float(row["paper_ad_statistic"])
                for row in rows
                if row["watermark"] == scheme
            ]
        )
        for scheme in SCHEME_ORDER
    }
    if any(values.size == 0 for values in mean_by_scheme.values()):
        raise ValueError("one or more watermark schemes have no results")

    critical_values = (
        (0.521, "15% critical value", CRITICAL_15_COLOR),
        (0.712, "5% critical value", CRITICAL_5_COLOR),
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#202020",
            "axes.titlecolor": "#202020",
            "text.color": "#202020",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.6, 9.3))
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.835,
        wspace=0.14,
        hspace=0.30,
    )
    add_distribution_panel(
        axes[0, 0],
        means,
        mean_by_scheme,
        bins=bins,
        xlabel="Mean endpoint z-score across 30 random walks",
        panel_label="A  Mean z-score distribution",
    )
    add_distribution_panel(
        axes[0, 1],
        statistics,
        statistic_by_scheme,
        bins=bins,
        xlabel="Anderson–Darling statistic",
        panel_label="B  Anderson–Darling statistic distribution",
        critical_values=critical_values,
    )
    add_ecdf_panel(
        axes[1, 0],
        mean_by_scheme,
        xlabel="Mean endpoint z-score across 30 random walks",
        panel_label="C  Mean z-score empirical CDF",
        overall_values=means,
    )
    add_ecdf_panel(
        axes[1, 1],
        statistic_by_scheme,
        xlabel="Anderson–Darling statistic",
        panel_label="D  Anderson–Darling statistic empirical CDF",
        overall_values=statistics,
        critical_values=critical_values,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        fontsize=9.2,
        handlelength=2.4,
        columnspacing=1.5,
    )
    figure.suptitle(
        "Equivalent-space endpoint distributions across 1,620 watermarked programs",
        y=0.982,
        fontsize=14.5,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.bins < 10:
        raise ValueError("--bins must be at least 10")
    plot(load_results(args.input), args.output, bins=args.bins, dpi=args.dpi)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
