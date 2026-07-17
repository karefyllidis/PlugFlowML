"""Physics-aware evaluation helpers for full-axial-profile PFR surrogates.

Adapted from the axial-station / exit-plane diagnostics in the NESP MLP reference
(``main_MLP_PFR_profiles.py``) to PlugFlowML's ``features_targets_*.pkl`` schema, where
the axial coordinate is ``relative_position`` (x/L in [0, 1]) rather than
``residence_time_s``. Shared by Main_6 (SimpleNN) and reusable by Main_7 (PINN).

All functions operate on physical-unit numpy arrays plus column-name lists; they do
not touch global matplotlib rcParams (the caller runs ``setup_matplotlib()`` once).

N. Karefyllidis 2026
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.utils.plot_style import (
    AXIAL_STATION_LABELS,
    AXIAL_STATION_STYLES,
    COLOR_CANTERA,
    COLOR_MODEL,
)


# ── metrics ──────────────────────────────────────────────────────────────────
def column_spans(y_true: np.ndarray) -> np.ndarray:
    """Per-column value range (max - min) of the ground truth, for NRMSE scaling."""
    y_true = np.atleast_2d(np.asarray(y_true, dtype=float))
    return y_true.max(axis=0) - y_true.min(axis=0)


def nrmse_pct(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spans: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-column normalised RMSE as a percentage: ``100 * RMSE / span``.

    ``span`` defaults to the range of ``y_true`` for each column. Columns with a
    zero span (constant truth) return NaN rather than dividing by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    spans = column_spans(y_true) if spans is None else np.asarray(spans, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * rmse / spans
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out


def _nearest_station_rows(
    run_ids: np.ndarray,
    relative_position: np.ndarray,
    stations: Sequence[float],
) -> dict[float, np.ndarray]:
    """Positional row indices nearest to each x/L station, one per run."""
    run_ids = np.asarray(run_ids)
    rel = np.asarray(relative_position, dtype=float)
    picks: dict[float, list[int]] = {float(s): [] for s in stations}
    for rid in np.unique(run_ids):
        loc = np.flatnonzero(run_ids == rid)
        x = rel[loc]
        if x.size == 0:
            continue
        for s in stations:
            picks[float(s)].append(int(loc[int(np.argmin(np.abs(x - s)))]))
    return {s: np.asarray(idx, dtype=int) for s, idx in picks.items()}


def collect_station_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_ids: np.ndarray,
    relative_position: np.ndarray,
    stations: Sequence[float],
    target_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-station error table across runs (nearest axial point to each x/L).

    For every station the nearest axial row of each run is gathered, then RMSE /
    MAE / NRMSE% are computed per target across those rows. NRMSE% is normalised by
    the *global* per-column span (over all supplied rows) so stations are comparable.

    Returns a tidy DataFrame: ``station, target, nrmse_pct, rmse, mae, n_runs``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if target_cols is None:
        target_cols = [f"target_{i}" for i in range(y_true.shape[1])]
    spans = column_spans(y_true)
    station_idx = _nearest_station_rows(run_ids, relative_position, stations)

    rows = []
    for s in stations:
        idx = station_idx[float(s)]
        if idx.size == 0:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        nr = nrmse_pct(yt, yp, spans=spans)
        rmse = np.sqrt(np.mean((yt - yp) ** 2, axis=0))
        mae = np.mean(np.abs(yt - yp), axis=0)
        for k, tgt in enumerate(target_cols):
            rows.append(
                {
                    "station": float(s),
                    "target": tgt,
                    "nrmse_pct": float(nr[k]),
                    "rmse": float(rmse[k]),
                    "mae": float(mae[k]),
                    "n_runs": int(idx.size),
                }
            )
    return pd.DataFrame(rows)


def species_sum_conservation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_cols: Sequence[str],
    target_cols: Sequence[str],
    run_ids: Optional[np.ndarray] = None,
    relative_position: Optional[np.ndarray] = None,
    exit_station: float = 1.0,
) -> dict:
    """Mass-fraction closure check: mean of ΣY over species, Cantera vs NN.

    The lumped mass fractions should sum to ~1 at every axial point. This reports the
    mean sum (and absolute deviation) overall and, when run/position arrays are given,
    at the reactor exit (nearest row to ``exit_station`` per run).
    """
    idx = [target_cols.index(c) for c in species_cols if c in target_cols]
    yt = np.asarray(y_true, dtype=float)[:, idx]
    yp = np.asarray(y_pred, dtype=float)[:, idx]
    sum_true = yt.sum(axis=1)
    sum_pred = yp.sum(axis=1)
    stats = {
        "n_species": len(idx),
        "mean_sum_true_overall": float(np.mean(sum_true)),
        "mean_sum_pred_overall": float(np.mean(sum_pred)),
        "mean_abs_sum_error_overall": float(np.mean(np.abs(sum_pred - sum_true))),
        "max_abs_sum_error_overall": float(np.max(np.abs(sum_pred - sum_true))),
    }
    if run_ids is not None and relative_position is not None:
        exit_idx = _nearest_station_rows(run_ids, relative_position, [exit_station])[
            float(exit_station)
        ]
        if exit_idx.size:
            stats.update(
                {
                    "exit_station": float(exit_station),
                    "mean_sum_true_exit": float(np.mean(sum_true[exit_idx])),
                    "mean_sum_pred_exit": float(np.mean(sum_pred[exit_idx])),
                    "mean_abs_sum_error_exit": float(
                        np.mean(np.abs(sum_pred[exit_idx] - sum_true[exit_idx]))
                    ),
                }
            )
    return stats


# ── plots ────────────────────────────────────────────────────────────────────
def plot_axial_station_nrmse(
    df_station: pd.DataFrame,
    target_cols: Sequence[str],
    *,
    good_pct: float = 5.0,
    acceptable_pct: float = 15.0,
    stations: Optional[Sequence[float]] = None,
    label_map: Optional[dict] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Grouped NRMSE% bars per target, one bar per x/L station, with target bands.

    Matches the NESP reference look (``AXIAL_STATION_STYLES``): white-fill bars with
    a coloured edge + hatch per station (magenta/blue/red/black for 0.25/0.5/0.75/1.0),
    plus palegreen/yellow acceptance guides. Falls back to a plain colour cycle for
    stations outside the standard four. Returns the matplotlib Figure.
    """
    cols = [c for c in target_cols if c in set(df_station["target"])]
    if stations is None:
        stations = sorted(df_station["station"].unique())
    stations = sorted(float(s) for s in stations)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(cols)), 4.2))
    else:
        fig = ax.figure

    style_by_station = {float(s): style for s, style in AXIAL_STATION_STYLES}
    label_by_station = {
        float(s): lab for (s, _), lab in zip(AXIAL_STATION_STYLES, AXIAL_STATION_LABELS)
    }
    _fallback_edges = ["magenta", "blue", "red", "black", "darkorange", "teal"]

    n_st = max(1, len(stations))
    width = 0.18
    offsets = np.linspace(-(n_st - 1) * width / 2, (n_st - 1) * width / 2, n_st)
    x = np.arange(len(cols), dtype=float)

    lut = {(r.target, r.station): r.nrmse_pct for r in df_station.itertuples(index=False)}
    for i, s in enumerate(stations):
        heights = [lut.get((c, s), np.nan) for c in cols]
        style = style_by_station.get(
            s, {"facecolor": "white", "edgecolor": _fallback_edges[i % len(_fallback_edges)], "linewidth": 1.0}
        )
        hatch = style.get("hatch")
        bar_kw = {k: v for k, v in style.items() if k != "hatch"}
        bars = ax.bar(
            x + offsets[i],
            heights,
            width,
            label=label_by_station.get(s, f"x/L = {s:g}"),
            zorder=3,
            **bar_kw,
        )
        if hatch:
            for b in bars:
                b.set_hatch(hatch)

    labels = [(label_map or {}).get(c, c) for c in cols]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
    ax.set_ylabel("NRMSE (%)", fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)

    finite = [h for s in stations for h in [lut.get((c, s), np.nan) for c in cols] if np.isfinite(h)]
    ymax = max(finite) * 1.15 if finite else 1.0
    ymax = min(100.0, ymax)
    ax.set_ylim(0, ymax)

    ax.axhspan(0.0, min(good_pct, ymax), color="palegreen", alpha=0.1, zorder=0)
    ax.axhspan(good_pct, min(acceptable_pct, ymax), color="yellow", alpha=0.05, zorder=0)
    ax.axhline(good_pct, color="palegreen", linestyle="-", linewidth=1.2, zorder=2, label="Target (good) < 5%")
    ax.axhline(acceptable_pct, color="yellow", linestyle="-", linewidth=1.2, zorder=2, label="Acceptable <= 15%")

    if title:
        ax.set_title(title, fontsize=9)
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles, leg_labels, fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def plot_exit_joint_panels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols: Sequence[str],
    *,
    target_cols: Optional[Sequence[str]] = None,
    ncols: int = 3,
    label_map: Optional[dict] = None,
    scale_map: Optional[dict] = None,
    title: Optional[str] = None,
):
    """Grid of exit-plane joint plots: Cantera (x) vs NN (y) scatter + marginals.

    Matches the NESP reference look: a Cantera-blue scatter, a model-red 1:1 line
    (5% padded), and top/right marginal density histograms — both blue, since they
    describe the same Cantera-vs-NN pair, not two different series. ``y_true``/
    ``y_pred`` should already be restricted to the exit rows. ``scale_map`` optionally
    divides a column (e.g. ``{"pressure_Pa": 1e5}`` to display bar). Returns the
    matplotlib Figure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if target_cols is None:
        target_cols = list(cols)
    idx_of = {c: list(target_cols).index(c) for c in cols if c in target_cols}
    cols = [c for c in cols if c in idx_of]

    n = len(cols)
    if n == 0:
        raise ValueError("plot_exit_joint_panels: no valid columns to plot")
    ncols = int(min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    label_map = label_map or {}
    scale_map = scale_map or {}

    fig = plt.figure(figsize=(4.5 * ncols, 4.2 * nrows))
    for k, col in enumerate(cols):
        j = idx_of[col]
        div = float(scale_map.get(col, 1.0))
        xt = y_true[:, j] / div
        yp = y_pred[:, j] / div
        qty_label = label_map.get(col, col.replace("_", " "))

        ax = fig.add_subplot(nrows, ncols, k + 1)
        ax.scatter(xt, yp, color=COLOR_CANTERA, alpha=0.65, s=40, edgecolors="none", label=qty_label)

        lo = float(np.nanmin([xt.min(), yp.min()]))
        hi = float(np.nanmax([xt.max(), yp.max()]))
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLOR_MODEL, linestyle="-", linewidth=1.0)
        ax.set_xlabel("Cantera", fontsize=7)
        ax.set_ylabel("NN", fontsize=7)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="22%", pad=0.12, sharex=ax)
        ax_right = divider.append_axes("right", size="22%", pad=0.12, sharey=ax)
        bins = max(3, min(12, len(xt)))
        ax_top.hist(xt, bins=bins, color=COLOR_CANTERA, alpha=0.5, density=True)
        ax_right.hist(yp, bins=bins, orientation="horizontal", color=COLOR_CANTERA, alpha=0.5, density=True)
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

    for k in range(n, nrows * ncols):
        fig.add_subplot(nrows, ncols, k + 1).set_visible(False)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
