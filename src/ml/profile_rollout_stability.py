"""Rollout stability helpers for Main_8 (eval-only clamps and diagnostics)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def clamp_state_physical(
    y_phys: np.ndarray,
    state_cols: list[str],
    species_cols: list[str],
    *,
    min_primary: dict[str, float] | None = None,
) -> np.ndarray:
    """Clamp primary variables positive; species non-negative and renormalized.

    Used only after ``inverse_transform`` during evaluation — not in the scaled rollout loop.
    """
    out = np.asarray(y_phys, dtype=float).copy()
    defaults = {
        "temperature_K": 200.0,
        "pressure_Pa": 1.0,
        "density_kgm3": 1e-6,
        "velocity_ms": 1e-6,
    }
    mins = {**defaults, **(min_primary or {})}
    for col, lo in mins.items():
        if col in state_cols:
            j = state_cols.index(col)
            out[:, j] = np.maximum(out[:, j], lo)

    if species_cols:
        sp_idx = [state_cols.index(c) for c in species_cols]
        out[:, sp_idx] = np.maximum(out[:, sp_idx], 0.0)
        sums = out[:, sp_idx].sum(axis=1, keepdims=True)
        mask = sums.squeeze(-1) > 1e-12
        if np.any(mask):
            sp_block = out[:, sp_idx]
            sp_block[mask] = sp_block[mask] / sums[mask]
            out[:, sp_idx] = sp_block

    return out


def rollout_minmax_table(
    rows: list[dict],
    state_cols: list[str],
    *,
    label: str = "rollout",
) -> pd.DataFrame:
    """Per-state min/max for true vs predicted rollout profiles."""
    if not rows:
        return pd.DataFrame()
    y_true = np.concatenate([r["true"] for r in rows], axis=0)
    y_pred = np.concatenate([r["pred"] for r in rows], axis=0)
    records = []
    for j, col in enumerate(state_cols):
        records.append({
            "variable": col,
            "true_min": float(np.min(y_true[:, j])),
            "true_max": float(np.max(y_true[:, j])),
            "pred_min": float(np.nanmin(y_pred[:, j])),
            "pred_max": float(np.nanmax(y_pred[:, j])),
            "pred_min_over_true_max": float(
                np.nanmax(y_pred[:, j]) / max(np.max(y_true[:, j]), 1e-12)
            ),
            "mode": label,
        })
    return pd.DataFrame(records)


def first_divergence_report(
    rows: list[dict],
    state_cols: list[str],
    relative_position: np.ndarray,
    *,
    rel_threshold: float = 0.25,
    abs_threshold: float | None = None,
) -> pd.DataFrame:
    """First axial index where |pred-true| exceeds threshold (per variable, per run)."""
    records = []
    offset = 0
    for r in rows:
        n = r["true"].shape[0]
        x_rel = relative_position[offset : offset + n]
        offset += n
        for j, col in enumerate(state_cols):
            err = np.abs(r["pred"][:, j] - r["true"][:, j])
            scale = np.maximum(np.abs(r["true"][:, j]), 1e-12)
            rel_err = err / scale
            hit = np.where(rel_err > rel_threshold)[0]
            if abs_threshold is not None:
                hit = np.where((err > abs_threshold) | (rel_err > rel_threshold))[0]
            first_i = int(hit[0]) if len(hit) else -1
            records.append({
                "run_id": r["run_id"],
                "variable": col,
                "first_index": first_i,
                "relative_position": float(x_rel[first_i]) if first_i >= 0 else np.nan,
                "rel_err_at_first": float(rel_err[first_i]) if first_i >= 0 else np.nan,
            })
    return pd.DataFrame(records)


def first_nonphysical_divergence_report(
    rows: list[dict],
    state_cols: list[str],
    species_cols: list[str],
    relative_position: np.ndarray,
    *,
    rel_threshold: float = 0.25,
) -> pd.DataFrame:
    """First axial index where prediction is nonphysical or rel error exceeds threshold."""
    primary = ["temperature_K", "pressure_Pa", "density_kgm3", "velocity_ms"]
    records = []
    offset = 0
    for r in rows:
        n = r["true"].shape[0]
        x_rel = relative_position[offset : offset + n]
        offset += n
        true_phys = r["true"]
        pred_phys = r["pred"]
        for j, col in enumerate(state_cols):
            pred_v = pred_phys[:, j]
            true_v = true_phys[:, j]
            nonphys = np.zeros(n, dtype=bool)
            if col in primary:
                if col == "temperature_K":
                    nonphys |= pred_v < 200.0
                elif col == "pressure_Pa":
                    nonphys |= pred_v < 1.0
                else:
                    nonphys |= pred_v < 1e-6
            if col in species_cols:
                nonphys |= pred_v < 0.0
            scale = np.maximum(np.abs(true_v), 1e-12)
            rel_err = np.abs(pred_v - true_v) / scale
            bad = nonphys | (rel_err > rel_threshold)
            first_i = int(np.argmax(bad)) if np.any(bad) else -1
            reason = ""
            if first_i >= 0:
                if nonphys[first_i]:
                    reason = "nonphysical"
                elif rel_err[first_i] > rel_threshold:
                    reason = "rel_err"
            records.append({
                "run_id": r["run_id"],
                "variable": col,
                "first_index": first_i,
                "relative_position": float(x_rel[first_i]) if first_i >= 0 else np.nan,
                "rel_err_at_first": float(rel_err[first_i]) if first_i >= 0 else np.nan,
                "reason": reason,
            })
    return pd.DataFrame(records)


def envelope_sanity_summary(
    rows: list[dict],
    state_cols: list[str],
    species_cols: list[str],
    *,
    rel_margin: float = 0.25,
) -> dict[str, bool]:
    """True if rollout pred max/min stay within rel_margin of truth envelope per variable."""
    table = rollout_minmax_table(rows, state_cols)
    ok: dict[str, bool] = {}
    for _, row in table.iterrows():
        col = row["variable"]
        tmin, tmax = row["true_min"], row["true_max"]
        span = max(tmax - tmin, 1e-12)
        lo = tmin - rel_margin * span
        hi = tmax + rel_margin * span
        if col in species_cols:
            lo, hi = max(0.0, tmin - 0.05), min(1.0, tmax + 0.05)
        ok[col] = bool(
            np.isfinite(row["pred_min"])
            and np.isfinite(row["pred_max"])
            and row["pred_min"] >= lo
            and row["pred_max"] <= hi
        )
    return ok


def plot_y_limits_from_true(
    rows: list[dict],
    state_cols: list[str],
    primary_cols: Sequence[str] | None = None,
    margin: float = 0.08,
) -> dict[str, tuple[float, float]]:
    """Per-variable y-limits from true profiles with relative margin."""
    if primary_cols is None:
        primary_cols = ["temperature_K", "pressure_Pa", "density_kgm3", "velocity_ms"]
    y_true = np.concatenate([r["true"] for r in rows], axis=0)
    limits: dict[str, tuple[float, float]] = {}
    for col in state_cols:
        j = state_cols.index(col)
        vmin = float(np.nanmin(y_true[:, j]))
        vmax = float(np.nanmax(y_true[:, j]))
        span = max(vmax - vmin, 1e-12)
        if col in primary_cols:
            limits[col] = (vmin - margin * span, vmax + margin * span)
        else:
            limits[col] = (max(0.0, vmin - 0.05 * span), min(1.0, vmax + 0.05 * span))
    return limits
