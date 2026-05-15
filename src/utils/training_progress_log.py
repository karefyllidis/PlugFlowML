"""Append-only training progress CSV and incremental Optuna JSON snapshots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

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
    p.parent.mkdir(parents=True, exist_ok=True)
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


def optuna_snapshot_from_study(
    study: Any,
    *,
    importances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the JSON snapshot dict used by Main_6 / Main_7 §6b-ii."""
    completed = [t for t in study.trials if t.value is not None]
    vals = [float(t.value) for t in completed]
    snap: dict[str, Any] = {
        "trials": [
            {
                "number": t.number,
                "value": float(t.value),
                "params": dict(t.params),
            }
            for t in completed
        ],
        "best_val_r2": max(vals) if vals else float("nan"),
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
    snap = optuna_snapshot_from_study(study, importances=importances)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p
