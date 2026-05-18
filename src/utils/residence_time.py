"""Local residence time along steady PFR axial profiles."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_residence_time_for_run(
    z_m: np.ndarray,
    u_ms: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Cumulative residence time τ(z) for one sorted profile.

    Uses midpoint velocity: dt[i] = dz[i] / u_mid[i], τ[0] = 0.
    """
    z = np.asarray(z_m, dtype=float)
    u = np.asarray(u_ms, dtype=float)
    if z.size < 2:
        raise ValueError("Need at least two axial points to compute residence time.")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z_position_m must be strictly increasing within a run.")
    if not np.all(u > 0):
        raise ValueError("velocity_ms must be positive for residence-time mapping.")

    dz = np.diff(z)
    u_mid = 0.5 * (u[:-1] + u[1:])
    dt = dz / u_mid
    tau = np.concatenate([[0.0], np.cumsum(dt)])
    if not np.all(np.diff(tau) > 0):
        raise ValueError("tau must be strictly increasing (check velocity and grid).")
    tau_total = float(tau[-1])
    relative_tau = tau / tau_total if tau_total > 0 else np.zeros_like(tau)
    return {
        "tau_s": tau,
        "dt_s": np.concatenate([dt, [np.nan]]),
        "log_dt_s": np.concatenate([np.log(np.maximum(dt, 1e-30)), [np.nan]]),
        "tau_total_s": tau_total,
        "relative_tau": relative_tau,
    }


def add_residence_time_columns(
    df: pd.DataFrame,
    *,
    run_cols: list[str],
    z_col: str = "z_position_m",
    u_col: str = "velocity_ms",
    on_error: str = "warn",
) -> pd.DataFrame:
    """Add τ, dt, log_dt, relative_tau per row (grouped by simulation run).

    Rows in runs that fail validation are left as NaN for τ columns when
    ``on_error='warn'``; use ``on_error='raise'`` to fail fast.
    """
    if on_error not in ("warn", "raise"):
        raise ValueError("on_error must be 'warn' or 'raise'")

    out = df.copy()
    for col in ("tau_s", "dt_s", "log_dt_s", "tau_total_s", "relative_tau"):
        if col not in out.columns:
            out[col] = np.nan

    group_keys = [c for c in run_cols if c in out.columns]
    if not group_keys:
        raise ValueError("run_cols must identify unique simulation runs.")

    missing = [c for c in (z_col, u_col) if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns for residence time: {missing}")

    failed_runs: list[str] = []
    for key, idx in out.groupby(group_keys, sort=False).groups.items():
        g = out.loc[idx].sort_values(z_col)
        try:
            rt = compute_residence_time_for_run(
                g[z_col].to_numpy(),
                g[u_col].to_numpy(),
            )
        except ValueError as exc:
            label = key if isinstance(key, str) else repr(key)
            failed_runs.append(f"{label}: {exc}")
            if on_error == "raise":
                raise ValueError(f"Residence time failed for run {label}") from exc
            continue
        out.loc[g.index, "tau_s"] = rt["tau_s"]
        out.loc[g.index, "dt_s"] = rt["dt_s"]
        out.loc[g.index, "log_dt_s"] = rt["log_dt_s"]
        out.loc[g.index, "relative_tau"] = rt["relative_tau"]
        out.loc[g.index, "tau_total_s"] = rt["tau_total_s"]

    if failed_runs:
        msg = (
            f"Residence time skipped for {len(failed_runs)} run(s). "
            f"First: {failed_runs[0]}"
        )
        warnings.warn(msg, stacklevel=2)
        logger.warning(msg)

    return out


def validate_residence_time_on_df(
    df: pd.DataFrame,
    *,
    run_cols: list[str],
    z_col: str = "z_position_m",
    u_col: str = "velocity_ms",
) -> list[str]:
    """Return human-readable errors for runs with invalid τ columns (empty if OK)."""
    errors: list[str] = []
    group_keys = [c for c in run_cols if c in df.columns]
    for key, idx in df.groupby(group_keys, sort=False).groups.items():
        g = df.loc[idx].sort_values(z_col)
        label = key if isinstance(key, str) else repr(key)
        if g[u_col].isna().any() or (g[u_col] <= 0).any():
            errors.append(f"{label}: non-positive or NaN {u_col}")
            continue
        if g["tau_s"].isna().any():
            errors.append(f"{label}: tau_s not computed")
            continue
        tau = g["tau_s"].to_numpy(dtype=float)
        if not np.all(np.diff(tau) > 0):
            errors.append(f"{label}: tau_s not strictly increasing")
        if not np.isfinite(g["tau_total_s"].iloc[0]) or g["tau_total_s"].iloc[0] <= 0:
            errors.append(f"{label}: invalid tau_total_s")
    return errors
