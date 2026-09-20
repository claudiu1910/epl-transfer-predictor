"""Matplotlib / seaborn figures for regression diagnostics.

Every function returns a :class:`matplotlib.figure.Figure` rather than calling
``plt.show()``, so the same code renders inside Streamlit (``st.pyplot(fig)``)
and writes PNGs from ``train.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend: safe under Streamlit and in CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)

# A small, consistent palette so the dashboard reads as one design.
ACCENT = "#2f7d6f"
ACCENT_LIGHT = "#79bfae"
NEGATIVE = "#c2543d"
GRID = "#d9d9d9"
INK = "#1f2933"

DEFAULT_FIGSIZE = (7.0, 5.0)
REPORTS_DIR = Path("reports")


def apply_style() -> None:
    """Apply the shared seaborn/matplotlib look."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
        }
    )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def plot_actual_vs_predicted(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Actual vs predicted transfer value",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Scatter of held-out predictions against reality, with the y = x line.

    Points on the dashed line are perfect predictions; the vertical gap to it
    is the error on that player.
    """
    apply_style()
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        actual,
        predicted,
        s=42,
        alpha=0.65,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.6,
        label="Players (held-out set)",
    )

    # y = x reference line across the combined range of both axes.
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    padding = 0.05 * (high - low or 1.0)
    limits = (low - padding, high + padding)
    ax.plot(limits, limits, linestyle="--", linewidth=1.6, color=NEGATIVE, label="Perfect prediction (y = x)")

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual value (EUR millions)")
    ax.set_ylabel("Predicted value (EUR millions)")
    ax.set_title(title)
    ax.legend(loc="upper left")

    # Corner annotation keeps the headline metric attached to the picture.
    if len(actual) > 1:
        residuals = actual - predicted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(residuals)))
        ax.annotate(
            f"$R^2$ = {r2:.3f}\nMAE = EUR {mae:.2f}M",
            xy=(0.97, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f4f7f6", "edgecolor": GRID},
        )

    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Residuals vs predicted value",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Residual plot: structure here means the linear form is leaving signal behind."""
    apply_style()
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residuals = actual - predicted

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, linestyle="--", linewidth=1.4, color=NEGATIVE)
    ax.scatter(predicted, residuals, s=40, alpha=0.65, color=ACCENT, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Predicted value (EUR millions)")
    ax.set_ylabel("Residual (actual - predicted, EUR millions)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_residual_distribution(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Residual distribution",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Histogram + KDE of the errors; a centred bell means unbiased predictions."""
    apply_style()
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(residuals, kde=len(residuals) > 5, color=ACCENT, edgecolor="white", ax=ax)
    ax.axvline(0.0, linestyle="--", linewidth=1.4, color=NEGATIVE)
    ax.set_xlabel("Residual (EUR millions)")
    ax.set_ylabel("Players")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_coefficients(
    coefficients: pd.DataFrame,
    top_n: int | None = None,
    title: str = "What drives predicted value",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Horizontal bar chart of standardised regression coefficients.

    Args:
        coefficients: Output of :func:`src.model.get_coefficients`
            (``label`` / ``coefficient`` columns, sorted by absolute size).
        top_n: Keep only the ``n`` strongest effects.
    """
    apply_style()
    frame = coefficients.copy()
    if top_n:
        frame = frame.head(top_n)
    # Largest bar on top once the axis is inverted.
    frame = frame.sort_values("coefficient")

    colors = [ACCENT if value >= 0 else NEGATIVE for value in frame["coefficient"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(frame["label"], frame["coefficient"], color=colors, edgecolor="white")
    ax.axvline(0.0, color=INK, linewidth=1.0)
    ax.set_xlabel("Coefficient (EUR millions per standard deviation)")
    ax.set_ylabel("")
    ax.set_title(title)

    # Value labels sit just outside each bar.
    span = float(frame["coefficient"].abs().max() or 1.0)
    for bar, value in zip(bars, frame["coefficient"]):
        offset = 0.02 * span * (1 if value >= 0 else -1)
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    ax.set_xlim(-1.35 * span, 1.35 * span)
    fig.tight_layout()
    return fig


def plot_value_distribution(
    values: Sequence[float],
    title: str = "Transfer value distribution",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Distribution of the target - context for how skewed the market is."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(np.asarray(values, dtype=float), kde=True, color=ACCENT_LIGHT, edgecolor="white", ax=ax)
    ax.set_xlabel("Transfer value (EUR millions)")
    ax.set_ylabel("Players")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Batch export
# --------------------------------------------------------------------------- #
def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Write a figure to disk and close it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved figure to %s", destination)
    return destination


def save_diagnostic_figures(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    coefficients: pd.DataFrame,
    values: Sequence[float] | None = None,
    output_dir: str | Path = REPORTS_DIR,
) -> list[Path]:
    """Render the full diagnostic set as PNGs (called at the end of training)."""
    output_dir = Path(output_dir)
    written = [
        save_figure(plot_actual_vs_predicted(y_true, y_pred), output_dir / "actual_vs_predicted.png"),
        save_figure(plot_residuals(y_true, y_pred), output_dir / "residuals.png"),
        save_figure(plot_residual_distribution(y_true, y_pred), output_dir / "residual_distribution.png"),
        save_figure(plot_coefficients(coefficients), output_dir / "feature_impact.png"),
    ]
    if values is not None:
        written.append(save_figure(plot_value_distribution(values), output_dir / "value_distribution.png"))
    return written
