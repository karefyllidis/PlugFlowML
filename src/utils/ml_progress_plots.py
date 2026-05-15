"""Matplotlib helpers for external NN training / Optuna progress viewers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_progress(
    df: pd.DataFrame,
    *,
    cur_epoch: int | None = None,
    figsize: tuple[float, float] = (9.0, 3.0),
) -> plt.Figure:
    """Two-panel train MSE + checkpoint R² (matches former in-notebook live plot)."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    if df.empty:
        ax0.set_title("Waiting for training progress…")
        ax1.set_title("Checkpoint R²")
        return fig

    epochs = df["epoch"].astype(float).values
    train_mse = df["train_mse"].astype(float).values
    ax0.semilogy(epochs, train_mse, color="#1f77b4", lw=1.2)
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Train MSE (standardised)")
    title_ep = cur_epoch if cur_epoch is not None else int(epochs[-1])
    ax0.set_title(f"Train loss (epoch {title_ep})")
    ax0.grid(True, which="both", alpha=0.25)

    ck = df[df["is_checkpoint"].astype(int) == 1].copy()
    if not ck.empty:
        ep_ck = ck["epoch"].astype(float).values
        ax1.plot(
            ep_ck,
            ck["train_r2"].astype(float).values,
            marker="o",
            ms=3,
            lw=1.0,
            label="train R²",
        )
        ax1.plot(
            ep_ck,
            ck["test_r2"].astype(float).values,
            marker="s",
            ms=3,
            lw=1.0,
            label="test R²",
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("R² (physical, uniform avg)")
    ax1.set_title("Checkpoint R²")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    return fig


def plot_optuna_progress(
    snap: dict[str, Any],
    *,
    figsize: tuple[float, float] = (9.0, 6.5),
) -> plt.Figure:
    """Stacked optimisation history + parallel coordinates from §6b snapshot JSON."""
    trials = snap.get("trials", [])
    best_val_r2 = float(snap.get("best_val_r2", float("nan")))
    n_trials_complete = int(snap.get("n_trials_complete", len(trials)))

    fig, (ax_hist, ax_pc) = plt.subplots(
        2,
        1,
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.05, 1.35]},
    )

    if not trials:
        ax_hist.set_title("Optuna progress (waiting for trials…)")
        ax_pc.set_title("Parallel coordinates")
        return fig

    trial_nums = [t["number"] for t in trials]
    trial_vals = [float(t["value"]) for t in trials]
    best_so_far = np.maximum.accumulate(trial_vals)

    ax_hist.plot(trial_nums, trial_vals, "o", alpha=0.55, color="b", label="Trial val R²")
    ax_hist.plot(trial_nums, best_so_far, "-", lw=1.6, color="r", label="Best so far")
    ax_hist.set_xlabel("Trial")
    ax_hist.set_ylabel("Validation R² (uniform avg)")
    ax_hist.set_title(
        f"Optuna progress  ·  {n_trials_complete} trials  ·  best R²={best_val_r2:.4f}"
    )
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(loc="lower right", fontsize=9)

    _need = ("h1", "h2", "h3", "dropout", "learning_rate", "batch_size")
    rows_pc: list[list[float]] = []
    for t in trials:
        p = t.get("params")
        if not isinstance(p, dict) or not p:
            continue
        if not all(k in p for k in _need):
            continue
        rows_pc.append([float(p[k]) for k in _need] + [float(t["value"])])

    if len(rows_pc) >= 1:
        Xp = np.asarray(rows_pc, dtype=float)
        col_labels = ["h1", "h2", "h3", "dropout", "learning_rate", "batch_size", "val R²"]
        n_dim = Xp.shape[1]
        Xn = np.zeros_like(Xp)
        lr_i = 4
        for j in range(n_dim):
            col = Xp[:, j].copy()
            if j == lr_i:
                col = np.log10(np.maximum(col, 1e-12))
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                Xn[:, j] = 0.5
            else:
                Xn[:, j] = (col - lo) / (hi - lo)
        ix = np.arange(n_dim)
        obj = Xp[:, -1]
        vmin, vmax = float(np.nanmin(obj)), float(np.nanmax(obj))
        norm_pc = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_pc = plt.cm.turbo
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            norm_pc = plt.Normalize(0.0, 1.0)
        ibest = int(np.nanargmax(obj))
        for i in range(len(Xn)):
            if i == ibest:
                continue
            ax_pc.plot(
                ix,
                Xn[i],
                color=cmap_pc(norm_pc(obj[i])),
                alpha=0.38,
                lw=1.1,
                zorder=1,
            )
        ax_pc.plot(ix, Xn[ibest], color="0.15", lw=1.5, zorder=3, label="Best trial")
        ax_pc.set_xticks(ix)
        ax_pc.set_xticklabels(col_labels, rotation=90, ha="right")
        ax_pc.set_ylabel("Normalized (min–max per axis; log for learning rate)")
        ax_pc.set_ylim(-0.06, 1.06)
        ax_pc.set_title("Parallel coordinates — tested hyperparameters vs validation R²")
        ax_pc.grid(True, axis="y", alpha=0.25)
        ax_pc.legend(loc="lower right", fontsize=8)
        sm_pc = plt.cm.ScalarMappable(norm=norm_pc, cmap=cmap_pc)
        sm_pc.set_array([])
        fig.colorbar(
            sm_pc,
            ax=ax_pc,
            fraction=0.046,
            pad=0.04,
            label="Validation R² (uniform avg)",
        )
    else:
        ax_pc.set_title("Parallel coordinates (waiting for per-trial params)")
        ax_pc.text(
            0.5,
            0.5,
            "No param rows yet",
            ha="center",
            va="center",
            transform=ax_pc.transAxes,
        )

    return fig
