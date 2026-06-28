"""Append-only training progress CSV and incremental Optuna JSON snapshots.

Log paths are shared with ``scripts/monitor/monitor_nn_training_progress.py``.
CPU / Optuna ``n_jobs`` settings in Main_7 do not change formats or paths.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# Notebook figure/report stems (keep in sync with monitor MAIN_6 / MAIN_7 / MAIN_8 flags).
MAIN_6_STEM = "Main_6__train_evaluate_SimpleNN_IO"
MAIN_7_STEM = "Main_7_train_evaluate_SimpleNN_full_profile"
MAIN_8_STEM = "Main_8_PINN_PFR"

# Main_6 / Main_7 §8 CSV + §6b Optuna JSON (monitor reads via path helpers below).
DATA_LOGS_DIRNAME = "data/logs"


def data_logs_dir(repo_root: str | Path) -> Path:
    return Path(repo_root) / DATA_LOGS_DIRNAME


def training_progress_log_path(repo_root: str | Path, notebook_stem: str) -> Path:
    """§8 CSV path consumed by the external training monitor."""
    return data_logs_dir(repo_root) / f"{notebook_stem}_training_progress.csv"


def optuna_snapshot_path(repo_root: str | Path, notebook_stem: str) -> Path:
    """§6b incremental JSON path consumed by the external Optuna monitor."""
    return data_logs_dir(repo_root) / f"{notebook_stem}_optuna_tuning_plot_data.json"


_TRAINING_HEADER = (
    "epoch",
    "train_mse",
    "test_mse",
    "train_r2",
    "test_r2",
    "lr",
    "is_checkpoint",
)


def init_training_progress_log(path: str | Path) -> Path:
    """Create or truncate the training progress CSV and write the header row."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)  # data/logs/
    with p.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_TRAINING_HEADER)
    return p


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def append_training_progress(
    path: str | Path,
    epoch: int,
    train_mse: float,
    *,
    test_mse: float | None = None,
    train_r2: float | None = None,
    test_r2: float | None = None,
    lr: float | None = None,
    is_checkpoint: bool = False,
) -> None:
    """Append one training-progress row and flush for external readers."""
    p = Path(path)
    with p.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                int(epoch),
                f"{float(train_mse):.10g}",
                _fmt_optional(test_mse),
                _fmt_optional(train_r2),
                _fmt_optional(test_r2),
                _fmt_optional(lr),
                1 if is_checkpoint else 0,
            ]
        )
        f.flush()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, snap: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_optuna_meta(path: Path) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    started = old.get("started_at")
    name = old.get("study_name")
    return (str(started) if started else None, str(name) if name else None)


def init_optuna_snapshot(
    path: str | Path,
    *,
    study_name: str = "",
) -> Path:
    """Clear Optuna JSON at the start of a new §6b study (monitor shows empty until trials finish)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    snap: dict[str, Any] = {
        "started_at": _utc_now_iso(),
        "study_name": study_name or None,
        "trials": [],
        "best_val_r2": None,
        "n_trials_complete": 0,
        "importances": None,
    }
    _write_json_atomic(p, snap)
    return p


def optuna_snapshot_from_study(
    study: Any,
    *,
    importances: Mapping[str, float] | None = None,
    started_at: str | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Build the JSON snapshot dict used by Main_6 / Main_7 §6b-ii."""
    completed = [t for t in study.trials if t.value is not None]
    vals = [float(t.value) for t in completed]
    snap: dict[str, Any] = {
        "started_at": started_at or _utc_now_iso(),
        "study_name": study_name,
        "trials": [
            {
                "number": t.number,
                "value": float(t.value),
                "params": dict(t.params),
            }
            for t in completed
        ],
        "best_val_r2": max(vals) if vals else None,
        "n_trials_complete": len(completed),
        "importances": dict(importances) if importances is not None else None,
    }
    return snap


def write_optuna_snapshot(
    path: str | Path,
    study: Any,
    *,
    importances: Mapping[str, float] | None = None,
) -> Path:
    """Write Optuna study progress to JSON (safe for incremental polling)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    started_at, study_name = _read_optuna_meta(p)
    snap = optuna_snapshot_from_study(
        study,
        importances=importances,
        started_at=started_at,
        study_name=study_name,
    )
    _write_json_atomic(p, snap)
    return p
