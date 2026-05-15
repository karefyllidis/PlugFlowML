#!/usr/bin/env python3
"""Plot Main_6 / Main_7 progress from ``data/logs/`` (one command).

Edit at most two flags, then from repo root:

    python scripts/monitor/monitor_nn_training_progress.py

Picks **whichever log was updated most recently** (Optuna JSON during §6b, training
CSV during §8). ``LIVE=True`` refreshes until the log stops changing (~90s idle).
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# === settings (Main_7 default) ===
MAIN_6 = False
MAIN_7 = True

LIVE = False  # True = refresh while the notebook runs; False = plot once and exit

_INTERVAL_S = 1.0
_IDLE_S = 90.0  # LIVE: stop after this many seconds with no log changes

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.utils.plot_style import setup_matplotlib
from src.utils.training_progress_log import (
    MAIN_6_STEM,
    MAIN_7_STEM,
    optuna_snapshot_path,
    training_progress_log_path,
)

setup_matplotlib()

if MAIN_6 and MAIN_7:
    raise SystemExit("Only one of MAIN_6 or MAIN_7 can be True.")
if not MAIN_6 and not MAIN_7:
    raise SystemExit("Set MAIN_6 or MAIN_7 to True.")

if MAIN_6:
    RUN_LABEL = "Main_6"
    _STEM = MAIN_6_STEM
else:
    RUN_LABEL = "Main_7"
    _STEM = MAIN_7_STEM

TRAIN_LOG = training_progress_log_path(_REPO, _STEM)
OPTUNA_LOG = optuna_snapshot_path(_REPO, _STEM)


def load_training_progress(log_path: Path) -> dict[str, list[float]]:
    """Parse §8 CSV (same schema as ``append_training_progress``)."""
    ep, train_mse = [], []
    ck_ep, test_mse = [], []
    ck_r2, train_r2, test_r2 = [], [], []

    if not log_path.is_file():
        return {
            "ep": ep,
            "train_mse": train_mse,
            "ck_ep": ck_ep,
            "test_mse": test_mse,
            "ck_r2": ck_r2,
            "train_r2": train_r2,
            "test_r2": test_r2,
        }

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
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

    return {
        "ep": ep,
        "train_mse": train_mse,
        "ck_ep": ck_ep,
        "test_mse": test_mse,
        "ck_r2": ck_r2,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }


def load_optuna_snapshot(log_path: Path) -> dict:
    """Parse §6b JSON (same schema as ``write_optuna_snapshot``)."""
    if not log_path.is_file():
        return {
            "trials": [],
            "best_val_r2": float("nan"),
            "n_trials_complete": 0,
            "started_at": None,
            "study_name": None,
        }
    snap = json.loads(log_path.read_text(encoding="utf-8"))
    trials = snap.get("trials", [])
    best = snap.get("best_val_r2")
    return {
        "trials": trials,
        "best_val_r2": float("nan") if best is None else float(best),
        "n_trials_complete": int(snap.get("n_trials_complete", len(trials))),
        "started_at": snap.get("started_at"),
        "study_name": snap.get("study_name"),
    }


def _pick_active_log(csv_path: Path, json_path: Path) -> tuple[Path, str]:
    """Newer mtime wins; if only one exists, use it."""
    csv_m = json_m = -1.0
    if csv_path.is_file():
        csv_m = csv_path.stat().st_mtime
    if json_path.is_file():
        json_m = json_path.stat().st_mtime
    if csv_m < 0 and json_m < 0:
        return csv_path, "training"
    if json_m >= csv_m:
        return json_path, "optuna"
    return csv_path, "training"


def _plot_training(log_path: Path, block, *, final=False):
    d = load_training_progress(log_path)
    ep, train_mse = d["ep"], d["train_mse"]
    ck_ep, test_mse = d["ck_ep"], d["test_mse"]
    ck_r2, train_r2, test_r2 = d["ck_r2"], d["train_r2"], d["test_r2"]

    fig, (ax_loss, ax_r2, ax_gap) = plt.subplots(1, 3, figsize=(16, 4.5))
    for _ax in (ax_loss, ax_r2, ax_gap):
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
            gaps = [tr - te for tr, te in zip(train_r2, test_r2)]

            ax_r2.plot(ck_r2, train_r2, color="b", lw=1.6, label="Train R²")
            ax_r2.plot(ck_r2, test_r2, color="r", lw=1.6, ls="-", label="Test R²")
            r2_min = min(min(train_r2), min(test_r2))
            ax_r2.set_ylim(bottom=min(-0.05, r2_min - 0.05), top=1.02)
            ax_r2.axhline(1.0, color="k", lw=0.8, alpha=0.4)
            ax_r2.axhline(0.0, color="k", lw=0.8, alpha=0.2)
            ax_r2.set_xlabel("Epoch")
            ax_r2.set_ylabel("R² (uniform average over targets)")
            ax_r2.set_title("Convergence — R² (physical units)", fontweight="normal")
            ax_r2.grid(True, alpha=0.35)
            ax_r2.legend(loc="lower right", frameon=True, fontsize=9)

            ax_gap.plot(ck_r2, gaps, color="lime", lw=1.6, ls="-", label="Train − test")
            ax_gap.axhline(0.0, color="k", lw=0.8, alpha=0.35)
            gap_lo, gap_hi = min(gaps), max(gaps)
            pad = max(0.02, 0.15 * (gap_hi - gap_lo) if gap_hi > gap_lo else 0.05)
            ax_gap.set_ylim(bottom=min(-0.02, gap_lo - pad), top=gap_hi + pad)
            ax_gap.set_xlabel("Epoch")
            ax_gap.set_ylabel("Gap (train − test R²)")
            ax_gap.set_title("Overfit gap", fontweight="normal")
            ax_gap.grid(True, alpha=0.35)
            ax_gap.legend(loc="upper right", frameon=True, fontsize=9)

            last_gap = gaps[-1]
            ax_gap.text(
                0.02,
                0.98,
                f"Latest (epoch {int(ck_r2[-1])}):  gap={last_gap:+.4f}",
                transform=ax_gap.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                fontweight="normal",
            )
        else:
            ax_r2.legend(loc="lower right", frameon=True, fontsize=9)
            ax_gap.set_title("Overfit gap (waiting for checkpoints)", fontweight="normal")
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


def _plot_optuna(log_path: Path, block, *, final=False):
    snap = load_optuna_snapshot(log_path)
    trials = snap["trials"]
    best_r2 = snap["best_val_r2"]
    n_done = snap["n_trials_complete"]
    started_at = snap.get("started_at")
    run_lbl = f"  ·  run {started_at}" if started_at else ""

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

    fig.suptitle(f"{RUN_LABEL} — Optuna ({tag}){run_lbl}", fontweight="normal", y=0.98)
    fig.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.05)


def _plot(csv_path: Path, json_path: Path, block, *, final=False) -> tuple[Path, str]:
    path, kind = _pick_active_log(csv_path, json_path)
    if kind == "optuna":
        _plot_optuna(path, block, final=final)
    else:
        _plot_training(path, block, final=final)
    return path, kind


def _watch_signature(path: Path) -> tuple[int, float] | None:
    """File size + mtime; None if missing."""
    if not path.is_file():
        return None
    st = path.stat()
    return (st.st_size, st.st_mtime)


def _pair_signature(csv_path: Path, json_path: Path) -> tuple:
    return (_watch_signature(csv_path), _watch_signature(json_path))


if __name__ == "__main__":
    print(f"{RUN_LABEL}  |  logs: data/logs/  |  LIVE={LIVE}")
    if not TRAIN_LOG.is_file() and not OPTUNA_LOG.is_file():
        print("[INFO] No logs yet — run Main_7 §6b or §8 (writes under data/logs/).")

    if not LIVE:
        path, kind = _plot(TRAIN_LOG, OPTUNA_LOG, block=True, final=True)
        print(f"Plotted: {path.name} ({kind})")
    else:
        plt.ion()
        last_pair = None
        idle_polls = 0
        stale_need = max(1, int(_IDLE_S / _INTERVAL_S))
        try:
            while True:
                pair = _pair_signature(TRAIN_LOG, OPTUNA_LOG)
                if pair != last_pair:
                    plt.close("all")
                    path, kind = _plot(TRAIN_LOG, OPTUNA_LOG, block=False, final=False)
                    print(f"[update] {path.name} ({kind})")
                    last_pair = pair
                    idle_polls = 0
                else:
                    idle_polls += 1
                    if last_pair is not None and any(s is not None for s in last_pair):
                        if idle_polls >= stale_need:
                            plt.close("all")
                            path, kind = _plot(TRAIN_LOG, OPTUNA_LOG, block=False, final=True)
                            print(f"Done ({kind} idle). Ctrl+C to exit.")
                            try:
                                while plt.get_fignums():
                                    plt.pause(1.0)
                            except KeyboardInterrupt:
                                pass
                            break
                time.sleep(_INTERVAL_S)
        except KeyboardInterrupt:
            print("Stopped.")
