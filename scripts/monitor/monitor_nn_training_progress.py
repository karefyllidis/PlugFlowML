#!/usr/bin/env python3
"""Plot Main_6 / Main_7 training or Optuna progress (edit flags below; no CLI).

MAIN_6 / MAIN_7 — exactly one True (notebook + log paths).
OPTUNA — True during §6b (Optuna JSON); False during §8 (training CSV: train/test R² + gap).
FOLLOW — True: refresh until log idle, then final plot and exit; False: one-shot.

Run from repo root:  python scripts/monitor/monitor_nn_training_progress.py

See docs/ML_CONFIG_GUIDE.md (external monitor; Main_7 train/val/test splits).
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# === set these (exactly one of MAIN_6 / MAIN_7 must be True) ===
# ============================================================
MAIN_6 = False
MAIN_7 = True

OPTUNA = True   # True = Optuna JSON (§6b), False = training CSV (§8)
FOLLOW = False    # True = live refresh; stops when log is idle and shows final plot

_INTERVAL_S = 1.0
_STALE_POLLS = 5  # unchanged polls → run treated as finished

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.utils.plot_style import setup_matplotlib

setup_matplotlib()

if MAIN_6 and MAIN_7:
    raise SystemExit("Only one of MAIN_6 or MAIN_7 can be True.")
if not MAIN_6 and not MAIN_7:
    raise SystemExit("Set MAIN_6 or MAIN_7 to True.")

if MAIN_6:
    RUN_LABEL = "Main_6"
    LOG = _REPO / "outputs/reports/Main_6__train_evaluate_SimpleNN_IO_training_progress.csv"
    OPTUNA_LOG = _REPO / "outputs/figures/Main_6__train_evaluate_SimpleNN_IO/optuna_tuning_plot_data.json"
else:
    RUN_LABEL = "Main_7"
    LOG = _REPO / "outputs/reports/Main_7_train_evaluate_SimpleNN_full_profile_training_progress.csv"
    OPTUNA_LOG = _REPO / "outputs/figures/Main_7_train_evaluate_SimpleNN_full_profile/optuna_tuning_plot_data.json"

WATCH = OPTUNA_LOG if OPTUNA else LOG


def _plot_training(block, *, final=False):
    ep, train_mse = [], []
    ck_ep, test_mse = [], []
    ck_r2, train_r2, test_r2 = [], [], []

    if LOG.is_file():
        lines = LOG.read_text(encoding="utf-8").strip().splitlines()
        for line in lines[1:]:
            c = line.split(",")
            if len(c) < 7:
                continue
            ep.append(float(c[0]))
            train_mse.append(float(c[1]))
            if c[6].strip() != "1":
                continue
            e = float(c[0])
            ck_ep.append(e)
            if c[2]:
                test_mse.append(float(c[2]))
            if c[3] and c[4]:
                ck_r2.append(e)
                train_r2.append(float(c[3]))
                test_r2.append(float(c[4]))

    fig, (ax_loss, ax_r2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for _ax in (ax_loss, ax_r2):
        setup_matplotlib(_ax)

    if ep:
        ax_loss.plot(ep, train_mse, color="b", lw=1.6, label="Train (per epoch)")
        if ck_ep and test_mse:
            ax_loss.plot(ck_ep[: len(test_mse)], test_mse, color="r", lw=1.6, ls="-", ms=4, label="Test (checkpoints)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE (standardised targets)")
        ax_loss.set_yscale("log")
        ax_loss.set_title("Convergence — MSE loss", fontweight="normal")
        ax_loss.grid(True, which="both", alpha=0.35)
        ax_loss.legend(loc="upper right", frameon=True, fontsize=9)

        if ck_r2:
            ax_r2.plot(ck_r2, train_r2, color="b", lw=1.6, label="Train R²")
            ax_r2.plot(ck_r2, test_r2, color="r", lw=1.6, ls="-", label="Test R²")
            r2_min = min(min(train_r2), min(test_r2))
            ax_r2.set_ylim(bottom=min(-0.05, r2_min - 0.05), top=1.02)

            gaps = [tr - te for tr, te in zip(train_r2, test_r2)]
            ax_gap = ax_r2.twinx()
            setup_matplotlib(ax_gap)
            ax_gap.plot(ck_r2, gaps, color="m", lw=1.2, ls="--", label="Train − test gap")
            gap_lo = min(gaps)
            gap_hi = max(gaps)
            pad = max(0.02, 0.15 * (gap_hi - gap_lo) if gap_hi > gap_lo else 0.05)
            ax_gap.set_ylim(bottom=min(-0.02, gap_lo - pad), top=gap_hi + pad)
            ax_gap.set_ylabel("Overfit gap (train − test R²)", fontsize=9)

            h1, l1 = ax_r2.get_legend_handles_labels()
            h2, l2 = ax_gap.get_legend_handles_labels()
            ax_r2.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True, fontsize=8)

            last_gap = gaps[-1]
            ax_r2.text(
                0.02,
                0.98,
                f"Latest (epoch {int(ck_r2[-1])}):  "
                f"train R²={train_r2[-1]:.4f},  test R²={test_r2[-1]:.4f},  gap={last_gap:+.4f}",
                transform=ax_r2.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                fontweight="normal",
            )
        else:
            ax_r2.legend(loc="lower right", frameon=True, fontsize=9)
        ax_r2.axhline(1.0, color="k", lw=0.8, alpha=0.4)
        ax_r2.axhline(0.0, color="k", lw=0.8, alpha=0.2)
        ax_r2.set_xlabel("Epoch")
        ax_r2.set_ylabel("R² (uniform average over targets)")
        ax_r2.set_title("Convergence — R² (physical units)", fontweight="normal")
        ax_r2.grid(True, alpha=0.35)
    else:
        ax_loss.set_title("Waiting for log…", fontweight="normal")

    tag = "final" if final else "live"
    fig.suptitle(f"{RUN_LABEL} — training ({tag})", fontweight="normal", y=0.98)
    fig.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.05)


def _parallel_coords_on_ax(ax, trials):
    """Parallel coordinates (vertical axes) — same layout as notebook §6b-ii."""
    need = ("h1", "h2", "h3", "dropout", "learning_rate", "batch_size")
    rows = []
    for t in trials:
        p = t.get("params")
        if not isinstance(p, dict) or not all(k in p for k in need):
            continue
        rows.append([float(p[k]) for k in need] + [float(t["value"])])

    if not rows:
        ax.set_title("Parallel coordinates (waiting for trial params)", fontweight="normal")
        return

    xp = np.asarray(rows, dtype=float)
    labels = ["h1", "h2", "h3", "dropout", "learning_rate", "batch_size", "val R²"]
    xn = np.zeros_like(xp)
    lr_col = 4
    for j in range(xp.shape[1]):
        col = xp[:, j].copy()
        if j == lr_col:
            col = np.log10(np.maximum(col, 1e-12))
        lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            xn[:, j] = 0.5
        else:
            xn[:, j] = (col - lo) / (hi - lo)

    ix = np.arange(xp.shape[1])
    obj = xp[:, -1]
    vmin, vmax = float(np.nanmin(obj)), float(np.nanmax(obj))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        norm = plt.Normalize(0.0, 1.0)
    cmap = plt.cm.turbo
    ibest = int(np.nanargmax(obj))
    for i in range(len(xn)):
        if i == ibest:
            continue
        ax.plot(ix, xn[i], color=cmap(norm(obj[i])), alpha=0.38, lw=1.1, zorder=1)
    ax.plot(ix, xn[ibest], color="r", lw=1.5, linestyle="--", zorder=3, label="Best trial")
    ax.set_xticks(ix)
    ax.set_xticklabels(labels, rotation=90, ha="right")
    ax.set_ylabel("Normalized (min–max per axis; log for learning rate)")
    ax.set_ylim(-0.06, 1.06)
    ax.set_title("Parallel coordinates — hyperparameters vs validation R²", fontweight="normal")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Validation R² (uniform avg)")


def _plot_optuna(block, *, final=False):
    trials = []
    best_r2 = float("nan")
    n_done = 0
    if OPTUNA_LOG.is_file():
        snap = json.loads(OPTUNA_LOG.read_text(encoding="utf-8"))
        trials = snap.get("trials", [])
        best_r2 = float(snap.get("best_val_r2", float("nan")))
        n_done = int(snap.get("n_trials_complete", len(trials)))

    tag = "final" if final else "live"
    # Same panel heights as before (4.5 + 5.5 in), one figure, 2 rows × 1 col; width +1 cm
    _optuna_w_in = 10 + 1 / 2.54
    fig, (ax_hist, ax_pc) = plt.subplots(
        2,
        1,
        figsize=(_optuna_w_in, 10),
        gridspec_kw={"height_ratios": [4.5, 5.5]},
    )
    for _ax in (ax_hist, ax_pc):
        setup_matplotlib(_ax)

    if trials:
        nums = [float(t["number"]) for t in trials]
        vals = [float(t["value"]) for t in trials]
        best_so_far = np.maximum.accumulate(vals)
        ax_hist.plot(nums, vals, "o", color="b", alpha=0.55, label="Trial val R²")
        ax_hist.plot(nums, best_so_far, "--", color="r", lw=1.6, label="Best so far")
        ax_hist.set_xlabel("Trial")
        ax_hist.set_ylabel("Validation R² (uniform avg)")
        ax_hist.set_title(
            f"Optuna convergence  ·  {n_done} trials  ·  best R²={best_r2:.4f}",
            fontweight="normal",
        )
        ax_hist.grid(True, alpha=0.35)
        ax_hist.legend(loc="lower right", frameon=True, fontsize=9)
    else:
        ax_hist.set_title("Waiting for Optuna log…", fontweight="normal")

    _parallel_coords_on_ax(ax_pc, trials)

    fig.suptitle(f"{RUN_LABEL} — Optuna ({tag})", fontweight="normal", y=0.98)
    fig.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.05)


def _plot(block, *, final=False):
    if OPTUNA:
        _plot_optuna(block, final=final)
    else:
        _plot_training(block, final=final)


if __name__ == "__main__":
    mode = "optuna" if OPTUNA else "training"
    print(f"Monitor: {WATCH}")
    print(f"{RUN_LABEL}  MAIN_6={MAIN_6}  MAIN_7={MAIN_7}  OPTUNA={OPTUNA}  FOLLOW={FOLLOW}  ({mode})")
    if not FOLLOW:
        _plot(True, final=True)
    else:
        plt.ion()
        last_size = -1
        stale = 0
        got_data = False
        try:
            while True:
                size = WATCH.stat().st_size if WATCH.is_file() else -1
                if size != last_size:
                    plt.close("all")
                    _plot(False, final=False)
                    last_size = size
                    stale = 0
                    got_data = size > 0
                else:
                    stale += 1
                    if got_data and stale >= _STALE_POLLS:
                        plt.close("all")
                        _plot(True, final=True)
                        print("Run finished (log idle). Close the figure window to exit.")
                        break
                time.sleep(_INTERVAL_S)
        except KeyboardInterrupt:
            print("Stopped.")
