"""Main_8 smoke / small-dataset helpers (pipeline validation, not benchmarking)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_DEFAULT_STRATIFY_COLS = (
    "initial_temperature_K",
    "initial_pressure_Pa",
    "reactor_length_m",
    "reactor_diameter_m",
    "mass_flow_rate_kgps",
    "heat_flux_Wm2",
)


@dataclass(frozen=True)
class DatasetScaleReport:
    n_runs_total: int
    n_train_runs: int
    n_val_runs: int
    n_test_runs: int
    smoke_mode: bool
    warnings: tuple[str, ...]


def assess_run_level_scale(
    n_runs_total: int,
    n_train_runs: int,
    n_val_runs: int,
    n_test_runs: int,
    *,
    min_runs_stable_metrics: int = 20,
    min_runs_per_split: int = 3,
) -> DatasetScaleReport:
    """Classify dataset size and collect user-facing warnings."""
    msgs: list[str] = []
    smoke = n_runs_total < min_runs_stable_metrics

    if smoke:
        msgs.append(
            f"[SMOKE] Only {n_runs_total} Cantera run(s) — Main_8 validates data flow, "
            "scaling, sequences, and rollout mechanics; do not treat RMSE/R² as benchmarks."
        )
    if n_train_runs < min_runs_per_split:
        msgs.append(
            f"[WARN] Train split has {n_train_runs} run(s) "
            f"(recommend ≥{min_runs_per_split} for stable fitting)."
        )
    if n_val_runs < min_runs_per_split:
        msgs.append(
            f"[WARN] Val split has {n_val_runs} run(s) — val rollout-MSE may be NaN or noisy; "
            "use train loss + rollout diagnostics instead."
        )
    if n_test_runs < min_runs_per_split:
        msgs.append(
            f"[WARN] Test split has {n_test_runs} run(s) — test metrics are indicative only."
        )
    if n_runs_total < 5:
        msgs.append(
            "[WARN] Very few runs: prefer hidden_size 16–32, lr 1e-4–3e-4, grad clipping, "
            "and short epoch budgets until a production Cantera set is available."
        )

    return DatasetScaleReport(
        n_runs_total=n_runs_total,
        n_train_runs=n_train_runs,
        n_val_runs=n_val_runs,
        n_test_runs=n_test_runs,
        smoke_mode=smoke,
        warnings=tuple(msgs),
    )


def _quantile_stratum(values: np.ndarray, n_strata: int) -> np.ndarray:
    """Integer stratum labels 0..K-1 from quantile bins (handles ties)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.zeros(0, dtype=int)
    q = np.linspace(0.0, 1.0, max(2, n_strata + 1))
    edges = np.unique(np.quantile(v, q))
    if len(edges) < 2:
        return np.zeros(len(v), dtype=int)
    n_eff = len(edges) - 1
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_eff - 1)


def subset_sequence_dict(
    sequences: dict[int, dict],
    max_runs: int | None,
    *,
    seed: int = 42,
) -> dict[int, dict]:
    """Cap run count for faster val/test (deterministic subsample)."""
    if max_runs is None or len(sequences) <= max_runs:
        return sequences
    keys = sorted(sequences.keys())
    pick = np.random.default_rng(seed).choice(keys, size=max_runs, replace=False)
    return {int(k): sequences[int(k)] for k in pick}


def subsample_runs_representative(
    df: pd.DataFrame,
    run_id: pd.Series,
    run_cols: list[str],
    n_runs: int,
    *,
    random_state: int = 42,
    n_strata: int = 4,
    stratify_cols: list[str] | None = None,
    include_profile_length: bool = True,
) -> np.ndarray:
    """Pick ``n_runs`` Cantera runs stratified over design space (and axial length).

    One row per run is taken from ``run_cols``; each continuous column is binned
    into quantile strata, then a composite stratum key drives proportional sampling
    so rare design corners still appear in the dev set.
    """
    n_runs = int(n_runs)
    rid = run_id.to_numpy()
    unique = np.unique(rid)
    if n_runs <= 0 or n_runs >= len(unique):
        return unique

    cols = [c for c in (stratify_cols or _DEFAULT_STRATIFY_COLS) if c in run_cols and c in df.columns]
    if not cols:
        cols = [c for c in run_cols if c in df.columns]
    per_run = df.groupby(rid, sort=False)[cols].first()

    parts: list[pd.Series] = []
    for col in cols:
        vals = per_run[col].to_numpy()
        if per_run[col].dtype == object or per_run[col].dtype.name == "category":
            codes, _ = pd.factorize(per_run[col].astype(str))
            parts.append(pd.Series(codes, index=per_run.index, name=col))
        else:
            parts.append(
                pd.Series(_quantile_stratum(vals, n_strata), index=per_run.index, name=col)
            )

    if include_profile_length:
        n_ax = df.groupby(rid, sort=False).size().reindex(per_run.index)
        parts.append(
            pd.Series(_quantile_stratum(n_ax.to_numpy(dtype=float), min(3, n_strata)), index=per_run.index, name="n_axial")
        )

    if parts:
        stratum = parts[0].astype(str)
        for s in parts[1:]:
            stratum = stratum + "|" + s.astype(str)
    else:
        stratum = pd.Series("0", index=per_run.index)

    rng = np.random.default_rng(random_state)
    chosen: list = []
    groups = stratum.groupby(stratum)
    n_strata_eff = len(groups)

    strata = [(key, idx.index.to_numpy()) for key, idx in groups]
    sizes = np.array([len(ids) for _, ids in strata], dtype=float)
    weights = sizes / sizes.sum()
    targets = np.floor(weights * n_runs).astype(int)
    if n_runs >= n_strata_eff:
        targets = np.maximum(1, targets)
    while targets.sum() > n_runs:
        j = int(np.argmax(targets))
        targets[j] -= 1
    while targets.sum() < n_runs:
        j = int(rng.integers(0, len(targets)))
        if targets[j] < len(strata[j][1]):
            targets[j] += 1

    for (key, ids), k in zip(strata, targets):
        k = min(int(k), len(ids))
        if k <= 0:
            continue
        pick = rng.choice(ids, size=k, replace=False)
        chosen.extend(pick.tolist())

    if len(chosen) < n_runs:
        pool = np.setdiff1d(unique, np.array(chosen, dtype=rid.dtype))
        extra = min(n_runs - len(chosen), len(pool))
        if extra:
            chosen.extend(rng.choice(pool, size=extra, replace=False).tolist())
    elif len(chosen) > n_runs:
        chosen = rng.choice(np.array(chosen), size=n_runs, replace=False).tolist()

    return np.asarray(chosen, dtype=rid.dtype)


def print_scale_report(report: DatasetScaleReport) -> None:
    for line in report.warnings:
        print(line)
    if not report.warnings:
        print(
            f"Dataset scale OK ({report.n_runs_total} runs): metrics suitable for comparison."
        )


def metrics_disclaimer(smoke_mode: bool) -> str:
    if smoke_mode:
        return (
            "[SMOKE] Metrics below are for pipeline checks only — not model quality. "
            "Inspect rollout min/max and first-divergence tables for drift."
        )
    return "[INFO] Metrics intended for model comparison (sufficient run count)."


def format_metric_line(name: str, rmse: float, r2: float | None, smoke_mode: bool) -> str:
    if smoke_mode:
        return f"  {name}: RMSE={rmse:.4g} (R² omitted in smoke mode)"
    if r2 is None:
        return f"  {name}: RMSE={rmse:.4f}"
    return f"  {name}: RMSE={rmse:.4f}  R²={r2:.4f}"


def print_safeguards_checklist(
    *,
    grad_clip_norm: float,
    learning_rate: float,
    hidden_size: int,
    batch_size: int,
    split_by_run: bool = True,
    scalers_train_only: bool = True,
    rollout_scaled_space: bool = True,
    early_stopping_metric: str = "validation_rollout_mse",
) -> None:
    """Print Main_8 safeguard checklist before training."""
    items = [
        ("1. Gradient clipping", f"enabled (norm={grad_clip_norm})" if grad_clip_norm > 0 else "OFF"),
        ("2. Learning rate", f"{learning_rate:g} (target ~1e-4–3e-4)"),
        ("3. Hidden size", f"{hidden_size} (initial cap 32–64)"),
        ("4. Split by run (not axial rows)", "yes" if split_by_run else "NO"),
        ("5. Scalers fit on train runs only", "yes" if scalers_train_only else "NO"),
        ("6. Rollout in scaled space", "yes" if rollout_scaled_space else "NO"),
        ("7. Divergence diagnostics", "first_nonphysical_divergence_report + min/max tables"),
        ("8. Checkpoint criterion", early_stopping_metric),
        ("9. Metrics", "teacher-forced / rollout / outlet reported separately"),
        ("10. Plots", "y-limits from true envelope (exploded rollouts capped)"),
    ]
    print("Main_8 safeguards checklist:")
    for label, status in items:
        print(f"  [{status}] {label}")
    print()


def _safe_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / R² on finite pairs only; NaN if no valid points."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    n_nan_pred = int(np.sum(~np.isfinite(y_pred)))
    if ok.sum() < 2:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "n_finite": int(ok.sum()),
            "n_nan_pred": n_nan_pred,
        }
    yt, yp = y_true[ok], y_pred[ok]
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "n_finite": int(ok.sum()),
        "n_nan_pred": n_nan_pred,
    }


def count_nonfinite_predictions(rows: list[dict]) -> dict[str, int]:
    """Count NaN/Inf in physical pred arrays per mode."""
    pred = np.concatenate([r["pred"] for r in rows], axis=0)
    bad = ~np.isfinite(pred)
    return {
        "n_points": int(pred.size),
        "n_nonfinite": int(bad.sum()),
        "n_runs_any": int(sum(not np.isfinite(r["pred"]).all() for r in rows)),
    }


def per_variable_profile_metrics(
    rows: list[dict],
    state_cols: list[str],
    species_cols: list[str],
) -> pd.DataFrame:
    """Per-variable RMSE, MAE, R², outlet error, and envelope ratios (physical units)."""
    records = []
    for j, col in enumerate(state_cols):
        y_true = np.concatenate([r["true"][:, j] for r in rows])
        y_pred = np.concatenate([r["pred"][:, j] for r in rows])
        outlet_true = np.array([r["true"][-1, j] for r in rows])
        outlet_pred = np.array([r["pred"][-1, j] for r in rows])
        m = _safe_regression_metrics(y_true, y_pred)
        m_out = _safe_regression_metrics(outlet_true, outlet_pred)
        tmin, tmax = float(np.nanmin(y_true)), float(np.nanmax(y_true))
        pmin, pmax = float(np.nanmin(y_pred)), float(np.nanmax(y_pred))
        records.append({
            "variable": col,
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "outlet_rmse": m_out["rmse"],
            "n_nan_pred": m["n_nan_pred"],
            "true_min": tmin,
            "true_max": tmax,
            "pred_min": pmin,
            "pred_max": pmax,
            "pred_max_over_true_max": pmax / max(tmax, 1e-12) if np.isfinite(pmax) else np.nan,
            "pred_min_under_true_min": pmin / max(tmin, 1e-12) if tmin > 0 and np.isfinite(pmin) else np.nan,
        })
    if species_cols:
        sp_idx = [state_cols.index(c) for c in species_cols]
        sums_true = np.concatenate([r["true"][:, sp_idx].sum(1) for r in rows])
        sums_pred = np.concatenate([r["pred"][:, sp_idx].sum(1) for r in rows])
        m = _safe_regression_metrics(sums_true, sums_pred)
        outlet_true = np.array([r["true"][-1, sp_idx].sum() for r in rows])
        outlet_pred = np.array([r["pred"][-1, sp_idx].sum() for r in rows])
        m_out = _safe_regression_metrics(outlet_true, outlet_pred)
        records.append({
            "variable": "|sum_Y-1| (species)",
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "outlet_rmse": m_out["rmse"],
            "n_nan_pred": m["n_nan_pred"],
            "true_min": float(np.nanmin(sums_true)),
            "true_max": float(np.nanmax(sums_true)),
            "pred_min": float(np.nanmin(sums_pred)),
            "pred_max": float(np.nanmax(sums_pred)),
            "pred_max_over_true_max": np.nan,
            "pred_min_under_true_min": np.nan,
        })
    return pd.DataFrame(records)


def print_per_variable_metrics(
    tf_rows: list[dict],
    ro_rows: list[dict],
    state_cols: list[str],
    species_cols: list[str],
    *,
    title_tf: str = "Teacher-forced (one-step)",
    title_ro: str = "Autoregressive rollout",
) -> None:
    """Print separate metric tables — no pooled cross-unit R²."""
    for label, rows in ((title_tf, tf_rows), (title_ro, ro_rows)):
        nf = count_nonfinite_predictions(rows)
        if nf["n_nonfinite"] > 0:
            print(
                f"\n[WARN] {label}: {nf['n_nonfinite']:,} non-finite pred values "
                f"({nf['n_runs_any']} runs) — metrics use finite points only."
            )
        print(f"\n=== {label} ===")
        print(per_variable_profile_metrics(rows, state_cols, species_cols).to_string(index=False))


def rollout_divergence_summary(div_df: pd.DataFrame, *, rel_threshold: float = 0.25) -> str:
    """One-line summary of earliest divergence across variables."""
    if div_df is None or div_df.empty:
        return "Rollout divergence: no rows."
    hit = div_df[div_df["first_index"] >= 0]
    if hit.empty:
        return f"Rollout divergence: no step exceeded {rel_threshold:.0%} rel error."
    earliest = hit.sort_values(["relative_position", "first_index"]).iloc[0]
    return (
        f"Rollout divergence earliest: run_id={earliest['run_id']}, "
        f"{earliest['variable']} at rel_pos={earliest['relative_position']:.3f} "
        f"(rel err={earliest['rel_err_at_first']:.3g})."
    )
