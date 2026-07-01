"""Helpers for full-axial profile surrogates (Main_5 / Main_6)."""

from __future__ import annotations

import numpy as np


def anchor_inlet_profile_predictions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    run_ids: np.ndarray,
    relative_position: np.ndarray,
    *,
    x_tol: float = 1e-12,
    copy: bool = False,
) -> tuple[np.ndarray, int]:
    """Set predictions at each run's inlet (minimum ``relative_position``) to match truth.

    Design inputs (T0, P0, geometry, …) are already identical row-by-row in the
  axial overlay plots; mismatches at the left of the curve are model error at the
    first axial station. Anchoring enforces the simulated inlet state as a hard BC
    before metrics and Cantera-vs-ML profile figures — standard for profile surrogates.

    Returns ``(y_pred, n_rows_anchored)``. Updates ``y_pred`` in place unless
    ``copy=True``.
    """
    out = np.asarray(y_pred, dtype=float)
    if copy:
        out = out.copy()
    y_true = np.asarray(y_true, dtype=float)
    if out.shape != y_true.shape:
        raise ValueError(f"y_pred shape {out.shape} != y_true shape {y_true.shape}")

    run_ids = np.asarray(run_ids)
    x_over_L = np.asarray(relative_position, dtype=float)
    if out.shape[0] != len(run_ids) or out.shape[0] != len(x_over_L):
        raise ValueError("run_ids and relative_position must align with y_pred rows")

    n_anchored = 0
    for rid in np.unique(run_ids):
        mask = run_ids == rid
        if not np.any(mask):
            continue
        loc = np.flatnonzero(mask)
        x_run = x_over_L[mask]
        x_min = float(np.min(x_run))
        inlet = loc[x_run <= x_min + x_tol]
        if inlet.size == 0:
            continue
        out[inlet] = y_true[inlet]
        n_anchored += int(inlet.size)

    if not copy:
        y_pred[:] = out
    return out, n_anchored
