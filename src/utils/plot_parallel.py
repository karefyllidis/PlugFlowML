#!/usr/bin/env python3
"""
Parallel-axes plot utilities (matplotlib-only).

Two complementary techniques for multidimensional EDA:

* ``plot_parallel_coordinates`` — Inselberg-style polylines across continuous
  axes. One polyline per row, optionally colored by a continuous variable.
  Best for inspecting space-filling of a DOE / training set and seeing
  per-sample dependencies between dimensions.

* ``plot_parallel_sets`` — Kosara-style categorical ribbons. Continuous columns
  are binned into ``n_bins`` categories per axis; ribbon width between two
  adjacent axes is proportional to joint count, optionally colored by the
  mean of an outcome variable. Best for showing how many runs land in each
  operating regime and how regimes co-occur.

Both functions follow the project's matplotlib conventions (compatible with
``setup_matplotlib()``) and return figure/axes for caller-side saving.

Author: Nikolas Karefyllidis, PhD
"""

from __future__ import annotations

from typing import Sequence, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_columns(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise min-max normalize to [0, 1]. Returns (norm, lo, hi).

    Constant columns are mapped to 0.5 to avoid division by zero.
    """
    lo = np.nanmin(data, axis=0)
    hi = np.nanmax(data, axis=0)
    span = hi - lo
    safe_span = np.where(span > 0, span, 1.0)
    norm = (data - lo) / safe_span
    # constant columns -> midline
    const_cols = span <= 0
    if np.any(const_cols):
        norm[:, const_cols] = 0.5
    return norm, lo, hi


def _format_tick(value: float) -> str:
    """Compact numeric formatter that auto-switches between fixed and scientific."""
    if not np.isfinite(value):
        return ""
    a = abs(value)
    if a == 0:
        return "0"
    if a < 1e-3 or a >= 1e4:
        return f"{value:.2e}"
    if a < 1:
        return f"{value:.3g}"
    return f"{value:.4g}"


def _draw_axis_columns(
    ax: plt.Axes,
    n_dims: int,
    axis_labels: Sequence[str],
    lo: np.ndarray,
    hi: np.ndarray,
    n_ticks: int = 5,
    label_fontsize: int = 9,
    tick_fontsize: int = 8,
) -> None:
    """Draw a vertical axis at each x = i with ticks/labels in real units."""
    for i in range(n_dims):
        ax.axvline(i, color="0.25", lw=0.9, zorder=1)
        if hi[i] > lo[i]:
            tick_values = np.linspace(lo[i], hi[i], n_ticks)
            tick_norm = np.linspace(0.0, 1.0, n_ticks)
        else:
            tick_values = np.array([lo[i]])
            tick_norm = np.array([0.5])
        for t_norm, t_val in zip(tick_norm, tick_values):
            ax.plot([i - 0.018, i + 0.018], [t_norm, t_norm],
                    color="0.25", lw=0.8, zorder=2)
            ax.text(
                i + 0.025, t_norm, _format_tick(t_val),
                fontsize=tick_fontsize, va="center", ha="left", color="0.15",
            )
        ax.text(
            i, 1.045, axis_labels[i],
            fontsize=label_fontsize, ha="center", va="bottom",
            color="black",
        )


# ---------------------------------------------------------------------------
# Parallel Coordinates (continuous)
# ---------------------------------------------------------------------------

def plot_parallel_coordinates(
    df: pd.DataFrame,
    dims: Sequence[str],
    color_by: Optional[str] = None,
    *,
    axis_labels: Optional[Sequence[str]] = None,
    color_label: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "magma",
    alpha: float = 0.35,
    linewidth: float = 0.8,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12.0, 4.5),
    sort_by_color: bool = True,
    max_lines: Optional[int] = None,
    sample_seed: int = 0,
    stratify_by: Optional[str] = None,
    n_strata: int = 4,
    strata_cmap: str = "viridis",
    strata_label: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Inselberg parallel coordinates plot for continuous multidimensional data.

    Each row becomes a polyline crossing ``len(dims)`` parallel vertical axes;
    each axis is independently min-max normalized so all values share a single
    [0, 1] display range.

    Parameters
    ----------
    df : DataFrame
        Source data; must contain all ``dims`` (and ``color_by`` if given).
    dims : sequence of str
        Column names in the order they should appear on the x-axis.
    color_by : str, optional
        Continuous column whose value colors each polyline (via ``cmap``).
        If None, all polylines are drawn in a neutral gray.
    axis_labels : sequence of str, optional
        Display labels for each axis. Defaults to ``dims``.
    color_label : str, optional
        Colorbar label. Defaults to ``color_by``.
    title : str, optional
        Figure suptitle.
    cmap : str
        Matplotlib colormap name.
    alpha : float
        Polyline transparency. Lower for dense datasets.
    linewidth : float
        Polyline width.
    ax : Axes, optional
        Existing axes to draw into. Otherwise a new figure is created.
    figsize : (float, float)
        Figure size when ``ax`` is None.
    sort_by_color : bool
        If True and ``color_by`` is given, draw rows from low to high color
        value so high-value lines appear on top.
    max_lines : int, optional
        If set and the dataset has more than ``max_lines`` rows, draw only a
        random sample of ``max_lines`` rows. The per-axis normalization range
        is still taken from the full dataset, so the axes stay stable. Useful
        for dense designs where spaghetti makes structure hard to read.
        When ``stratify_by`` is also set, sampling is **stratified** — each
        stratum gets ``max_lines // n_strata`` rows so all colors stay
        visible even at high data volumes.
    sample_seed : int
        RNG seed used when ``max_lines`` triggers subsampling (default 0).
    stratify_by : str, optional
        Column in ``df`` whose **quantile bins** are used to color each line
        categorically. Lines are grouped into ``n_strata`` bins (equal count
        per bin) and each bin is drawn with a distinct color from
        ``strata_cmap``. A legend is added with the value range per bin.
        Use this to reveal whether the other dimensions are independently
        sampled within each regime of ``stratify_by``, or systematically
        correlated. Mutually exclusive with ``color_by``.
    n_strata : int
        Number of quantile bins used when ``stratify_by`` is set (default 4).
    strata_cmap : str
        Matplotlib colormap sampled at ``n_strata`` evenly-spaced points to
        produce the categorical palette. Use something different from
        ``cmap`` to avoid clashing with paired parallel-sets plots
        (default ``"viridis"``).
    strata_label : str, optional
        Legend title; defaults to ``stratify_by``.

    Returns
    -------
    (fig, ax)
    """
    if color_by is not None and stratify_by is not None:
        raise ValueError(
            "Pass either color_by (continuous colormap) or stratify_by "
            "(categorical quantile bins), not both."
        )

    use_cols = list(dims)
    if color_by:
        use_cols.append(color_by)
    if stratify_by:
        use_cols.append(stratify_by)
    # Deduplicate while preserving order (e.g. stratify_by may already be in dims).
    use_cols = list(dict.fromkeys(use_cols))
    work_full = df.loc[:, use_cols].dropna().copy()
    if len(work_full) == 0:
        raise ValueError("No rows remain after dropping NaNs for the requested columns.")

    n_full = len(work_full)

    # Normalize against the FULL set (pre-subsampling) so the axis ranges do
    # not depend on which rows happened to survive sampling.
    data_full = work_full[list(dims)].to_numpy(dtype=float)
    _, lo, hi = _normalize_columns(data_full)
    span = hi - lo
    safe_span = np.where(span > 0, span, 1.0)

    def _normalize(arr: np.ndarray) -> np.ndarray:
        n = (arr - lo) / safe_span
        n[:, span <= 0] = 0.5
        return n

    n_dims = len(dims)
    if n_dims < 2:
        raise ValueError("plot_parallel_coordinates requires at least 2 dimensions.")
    xs = np.arange(n_dims, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ── Branch 3: stratify_by (categorical color, quantile bins) ─────────────
    if stratify_by is not None:
        s_vals = work_full[stratify_by].to_numpy(dtype=float)
        # Quantile edges; collapse duplicates (e.g. heavily-tied columns).
        q = np.linspace(0.0, 1.0, n_strata + 1)
        edges = np.unique(np.quantile(s_vals, q))
        if len(edges) < 2:
            edges = np.array([float(s_vals.min()), float(s_vals.max() + 1e-9)])
        n_eff = len(edges) - 1
        bin_idx_full = np.clip(
            np.searchsorted(edges, s_vals, side="right") - 1, 0, n_eff - 1
        )

        # Categorical palette: sample strata_cmap at bin midpoints.
        cmap_obj = plt.get_cmap(strata_cmap)
        palette = [cmap_obj((i + 0.5) / max(n_eff, 1)) for i in range(n_eff)]

        # Stratified subsample so every bin contributes (otherwise small bins
        # might disappear in a uniform sample).
        per_bin_cap = (
            max(1, int(max_lines) // max(n_eff, 1))
            if max_lines is not None else None
        )
        rng = np.random.default_rng(int(sample_seed))

        for b in range(n_eff):
            mask = bin_idx_full == b
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            if per_bin_cap is not None and idx.size > per_bin_cap:
                idx = rng.choice(idx, size=per_bin_cap, replace=False)
            data_b = work_full[list(dims)].to_numpy(dtype=float)[idx]
            norm_b = _normalize(data_b)
            segments = np.stack(
                [np.repeat(xs[None, :], len(idx), axis=0), norm_b], axis=-1
            )
            lc = LineCollection(
                segments, colors=[palette[b]], alpha=alpha,
                linewidths=linewidth, zorder=3,
                label=f"{_format_tick(edges[b])}–{_format_tick(edges[b+1])}",
            )
            ax.add_collection(lc)

        leg = ax.legend(
            title=strata_label if strata_label else stratify_by,
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            fontsize=8, title_fontsize=9, frameon=False,
            handlelength=2.2, borderaxespad=0.0,
        )
        # Make legend handles fully opaque so the swatches read as the true
        # palette (the lines themselves can stay translucent).
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)

    # ── Branch 2: color_by (continuous colormap) ────────────────────────────
    elif color_by is not None:
        if max_lines is not None and n_full > max_lines:
            work = work_full.sample(n=int(max_lines), random_state=int(sample_seed))
        else:
            work = work_full
        data = work[list(dims)].to_numpy(dtype=float)
        n_rows = len(work)
        norm = _normalize(data)

        c_values = work[color_by].to_numpy(dtype=float)
        if sort_by_color:
            order = np.argsort(c_values, kind="stable")
            norm_sorted = norm[order]
            c_sorted = c_values[order]
        else:
            norm_sorted = norm
            c_sorted = c_values
        segments = np.stack(
            [np.repeat(xs[None, :], n_rows, axis=0), norm_sorted], axis=-1
        )
        lc = LineCollection(
            segments, cmap=cmap, alpha=alpha, linewidths=linewidth, zorder=3
        )
        lc.set_array(c_sorted)
        ax.add_collection(lc)
        # Colorbar from a separate ScalarMappable so it shows the full,
        # opaque colormap (matches plot_parallel_sets and avoids inheriting
        # the polyline alpha).
        finite = c_sorted[np.isfinite(c_sorted)]
        if finite.size > 0:
            vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-9
        sm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=plt.get_cmap(cmap),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035, aspect=22)
        cbar.set_label(color_label if color_label else color_by, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # ── Branch 1: neutral lines ─────────────────────────────────────────────
    else:
        if max_lines is not None and n_full > max_lines:
            work = work_full.sample(n=int(max_lines), random_state=int(sample_seed))
        else:
            work = work_full
        data = work[list(dims)].to_numpy(dtype=float)
        n_rows = len(work)
        norm = _normalize(data)
        segments = np.stack(
            [np.repeat(xs[None, :], n_rows, axis=0), norm], axis=-1
        )
        lc = LineCollection(
            segments, colors="0.35", alpha=alpha, linewidths=linewidth, zorder=3
        )
        ax.add_collection(lc)

    labels = list(axis_labels) if axis_labels is not None else list(dims)
    _draw_axis_columns(ax, n_dims, labels, lo, hi)

    ax.set_xlim(-0.25, n_dims - 1 + 0.25)
    ax.set_ylim(-0.04, 1.10)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=18)

    return fig, ax


# ---------------------------------------------------------------------------
# Parallel Sets (categorical, Kosara-style)
# ---------------------------------------------------------------------------

def _binned_labels(
    series: pd.Series, n_bins: int, strategy: str = "equal_width",
) -> Tuple[pd.Series, List[str], np.ndarray]:
    """Bin a continuous series into integer bin indices [0, n_eff-1].

    Returns (bin_idx, bin_labels, bin_edges). ``n_eff`` may be smaller than
    ``n_bins`` if duplicates collapse (e.g. constant column).
    """
    values = series.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Column '{series.name}' has no finite values.")
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        edges = np.array([lo, lo + max(abs(lo) * 1e-6, 1e-9)])
    elif strategy == "equal_width":
        edges = np.linspace(lo, hi, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(finite, qs))
        if edges.size < 2:
            edges = np.array([lo, hi])
    else:
        raise ValueError(f"Unknown binning strategy: {strategy!r}")

    bin_idx = np.clip(
        np.searchsorted(edges, values, side="right") - 1,
        0, len(edges) - 2,
    )
    labels = [f"{_format_tick(edges[i])}–{_format_tick(edges[i+1])}"
              for i in range(len(edges) - 1)]
    return pd.Series(bin_idx, index=series.index), labels, edges


def _bezier_ribbon(
    x0: float, x1: float,
    y0_bot: float, y0_top: float,
    y1_bot: float, y1_top: float,
) -> MplPath:
    """Build a closed cubic-Bezier ribbon connecting two vertical segments."""
    xm = 0.5 * (x0 + x1)
    verts = [
        (x0, y0_bot),
        (xm, y0_bot), (xm, y1_bot), (x1, y1_bot),
        (x1, y1_top),
        (xm, y1_top), (xm, y0_top), (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def plot_parallel_sets(
    df: pd.DataFrame,
    dims: Sequence[str],
    color_by: Optional[str] = None,
    *,
    n_bins: int = 5,
    bin_strategy: str = "equal_width",
    axis_labels: Optional[Sequence[str]] = None,
    color_label: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "magma",
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    base_color: str = "#4477AA",
    ribbon_alpha: float = 0.85,
    gap_frac: float = 0.015,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (13.0, 5.5),
    bin_label_fontsize: int = 7,
    show_bin_labels: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Kosara-style Parallel Sets for binned multidimensional data.

    Each continuous column in ``dims`` is binned into ``n_bins`` categories.
    For each adjacent pair of axes, ribbons are drawn whose width is
    proportional to the joint count of (left-bin, right-bin) and whose color
    is the mean of ``color_by`` over rows in that joint bin (or a single base
    color if ``color_by`` is None).

    Parameters
    ----------
    df : DataFrame
        Source data; must contain all ``dims`` (and ``color_by`` if given).
    dims : sequence of str
        Column names in the order they should appear on the x-axis.
    color_by : str, optional
        Continuous outcome column used to color ribbons by their joint-bin
        mean. If None, all ribbons share ``base_color``.
    n_bins : int
        Number of bins per axis (categories).
    bin_strategy : {'equal_width', 'quantile'}
        How to bin continuous columns.
    axis_labels : sequence of str, optional
        Display labels for each axis. Defaults to ``dims``.
    color_label : str, optional
        Colorbar label. Defaults to ``color_by``.
    title : str, optional
        Figure title.
    cmap : str
        Matplotlib colormap name (used when ``color_by`` is given).
    color_vmin, color_vmax : float, optional
        Manual color-scale bounds. By default they are derived from the
        **actual range of joint-bin means** that get painted onto the
        ribbons (not the raw ``color_by`` range), so the colorbar matches
        the colors visible in the plot. Pass values explicitly to override —
        e.g. ``color_vmin=0, color_vmax=100`` to force the colorbar onto a
        physical 0–100 % scale shared with another figure.
    base_color : str
        Ribbon color when ``color_by`` is None.
    ribbon_alpha : float
        Ribbon transparency.
    gap_frac : float
        Gap between bin segments on each axis, as a fraction of axis height.
    ax : Axes, optional
        Existing axes to draw into. Otherwise a new figure is created.
    figsize : (float, float)
        Figure size when ``ax`` is None.
    bin_label_fontsize : int
        Font size for per-bin range labels written on each axis.
    show_bin_labels : bool
        If True, annotate each bin segment with its value range.

    Returns
    -------
    (fig, ax)
    """
    use_cols = list(dims) + ([color_by] if color_by else [])
    work = df.loc[:, use_cols].dropna().copy()
    if len(work) == 0:
        raise ValueError("No rows remain after dropping NaNs for the requested columns.")

    n_dims = len(dims)
    if n_dims < 2:
        raise ValueError("plot_parallel_sets requires at least 2 dimensions.")

    # Bin every dimension.
    bin_idx_cols: Dict[str, pd.Series] = {}
    bin_labels: Dict[str, List[str]] = {}
    n_eff: Dict[str, int] = {}
    for d in dims:
        bi, bl, _ = _binned_labels(work[d], n_bins, bin_strategy)
        bin_idx_cols[d] = bi
        bin_labels[d] = bl
        n_eff[d] = len(bl)

    n_total = len(work)
    color_values = (work[color_by].to_numpy(dtype=float)
                    if color_by is not None else None)

    # Marginal bin counts per axis.
    bin_counts: Dict[str, np.ndarray] = {
        d: np.bincount(bin_idx_cols[d].to_numpy(), minlength=n_eff[d])
        for d in dims
    }

    # Y-extents on each axis (with gaps).
    gap = gap_frac
    bin_extents: Dict[str, List[Tuple[float, float]]] = {}
    for d in dims:
        counts = bin_counts[d]
        nb = n_eff[d]
        total_gap = gap * (nb - 1) if nb > 1 else 0.0
        usable = max(0.0, 1.0 - total_gap)
        heights = counts / n_total * usable
        extents = []
        y = 0.0
        for h in heights:
            extents.append((y, y + h))
            y += h + gap
        bin_extents[d] = extents

    # Precompute joint counts (and sums) for every axis pair so we can derive
    # the color normalization from the actual bin-mean range that will be
    # painted onto the ribbons. Using the raw color_by min/max here would
    # leave the colorbar extending into regions of the colormap that no
    # ribbon actually uses (e.g. a 16–100 % data range when bin means only
    # span 35–98 %).
    joint_counts: List[np.ndarray] = []
    joint_sums: List[np.ndarray] = []
    for i in range(n_dims - 1):
        d_l, d_r = dims[i], dims[i + 1]
        bi_l = bin_idx_cols[d_l].to_numpy()
        bi_r = bin_idx_cols[d_r].to_numpy()
        nl, nr = n_eff[d_l], n_eff[d_r]
        j = np.zeros((nl, nr), dtype=np.int64)
        np.add.at(j, (bi_l, bi_r), 1)
        joint_counts.append(j)
        if color_values is not None:
            js = np.zeros((nl, nr), dtype=np.float64)
            np.add.at(js, (bi_l, bi_r), color_values)
            joint_sums.append(js)

    # Color normalization for ribbons — based on the bin-mean range, not the
    # raw data range. Users can still override via color_vmin / color_vmax.
    if color_values is not None:
        all_means: List[np.ndarray] = []
        for jc, js in zip(joint_counts, joint_sums):
            nonempty = jc > 0
            if np.any(nonempty):
                all_means.append(js[nonempty] / jc[nonempty])
        if all_means:
            stacked = np.concatenate(all_means)
            auto_vmin = float(np.nanmin(stacked))
            auto_vmax = float(np.nanmax(stacked))
        else:
            auto_vmin, auto_vmax = 0.0, 1.0
        vmin = auto_vmin if color_vmin is None else float(color_vmin)
        vmax = auto_vmax if color_vmax is None else float(color_vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-9
        norm_color = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
    else:
        norm_color = None
        cmap_obj = None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Draw ribbons for each adjacent axis pair (joint/joint_sum precomputed).
    for i in range(n_dims - 1):
        d_l, d_r = dims[i], dims[i + 1]
        nl, nr = n_eff[d_l], n_eff[d_r]
        joint = joint_counts[i]
        joint_sum = joint_sums[i] if color_values is not None else None

        # Left-side stacking: within each left bin, stack outgoing ribbons
        # by ascending right-bin index.
        left_cursor = np.array([ext[0] for ext in bin_extents[d_l]], dtype=float)
        # Right-side stacking: within each right bin, stack incoming ribbons
        # by ascending left-bin index.
        right_cursor = np.array([ext[0] for ext in bin_extents[d_r]], dtype=float)

        left_total_h = np.array(
            [ext[1] - ext[0] for ext in bin_extents[d_l]], dtype=float)
        right_total_h = np.array(
            [ext[1] - ext[0] for ext in bin_extents[d_r]], dtype=float)
        left_count = bin_counts[d_l].astype(float)
        right_count = bin_counts[d_r].astype(float)

        for b_l in range(nl):
            for b_r in range(nr):
                c = joint[b_l, b_r]
                if c == 0:
                    continue
                # Left side ribbon height = fraction of left bin
                if left_count[b_l] > 0:
                    h_l = left_total_h[b_l] * c / left_count[b_l]
                else:
                    h_l = 0.0
                if right_count[b_r] > 0:
                    h_r = right_total_h[b_r] * c / right_count[b_r]
                else:
                    h_r = 0.0

                y0_bot = left_cursor[b_l]
                y0_top = y0_bot + h_l
                y1_bot = right_cursor[b_r]
                y1_top = y1_bot + h_r

                left_cursor[b_l] = y0_top
                right_cursor[b_r] = y1_top

                if color_values is not None:
                    mean_c = joint_sum[b_l, b_r] / c
                    facecolor = cmap_obj(norm_color(mean_c))
                else:
                    facecolor = base_color

                path = _bezier_ribbon(
                    i, i + 1,
                    y0_bot, y0_top,
                    y1_bot, y1_top,
                )
                patch = PathPatch(
                    path,
                    facecolor=facecolor,
                    edgecolor="none",
                    alpha=ribbon_alpha,
                    zorder=2,
                )
                ax.add_patch(patch)

    # Draw bin "boxes" on each axis on top of ribbons.
    bar_half_width = 0.012
    for i, d in enumerate(dims):
        for b, (y0, y1) in enumerate(bin_extents[d]):
            if y1 - y0 <= 0:
                continue
            ax.add_patch(plt.Rectangle(
                (i - bar_half_width, y0),
                2 * bar_half_width, y1 - y0,
                facecolor="0.15", edgecolor="white", linewidth=0.6,
                zorder=4,
            ))
            if show_bin_labels:
                ax.text(
                    i + bar_half_width + 0.012,
                    0.5 * (y0 + y1),
                    bin_labels[d][b],
                    fontsize=bin_label_fontsize, va="center", ha="left",
                    color="0.15",
                )

    # Axis labels at the top.
    labels = list(axis_labels) if axis_labels is not None else list(dims)
    for i, lab in enumerate(labels):
        ax.text(
            i, 1.045, lab,
            fontsize=9, ha="center", va="bottom",
            color="black",
        )

    # Colorbar.
    if color_values is not None:
        sm = mpl.cm.ScalarMappable(norm=norm_color, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035, aspect=22)
        cbar.set_label(color_label if color_label else color_by, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    ax.set_xlim(-0.30, n_dims - 1 + 0.30)
    ax.set_ylim(-0.03, 1.10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=18)

    return fig, ax


__all__ = [
    "plot_parallel_coordinates",
    "plot_parallel_sets",
]
