"""Mirror notebook stdout/stderr to a single .txt file under outputs/reports/.

Usage (first code cell after imports/paths):

    from src.utils.run_log import start_run_log
    start_run_log('Main_6_train_evaluate_SimpleNN_full_profile')

Each run overwrites ``outputs/reports/<notebook_name>.txt`` so the latest run is
always at a stable path. Calling ``start_run_log`` again closes any active tee
and opens a fresh file (e.g. re-running the setup cell restarts logging).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


class _Tee:
    """Forward writes to two text streams (e.g. original stdout + log file)."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        try:
            self._secondary.write(data)
            self._secondary.flush()
        except Exception:
            pass
        return n

    def flush(self) -> None:
        try:
            self._primary.flush()
        except Exception:
            pass
        try:
            self._secondary.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()


_ACTIVE_LOG_PATH: Path | None = None
_ACTIVE_LOG_FILE: TextIO | None = None
_ORIGINAL_STDOUT: TextIO | None = None
_ORIGINAL_STDERR: TextIO | None = None


def start_run_log(
    notebook_name: str,
    reports_dir: str | Path = "outputs/reports",
) -> Path:
    """Tee stdout/stderr to ``<reports_dir>/<safe_notebook_name>.txt`` (overwrite).

    If logging is already active, it is stopped first so this call always
    starts a fresh log file at the same stable path.

    Returns the Path to the log file.
    """
    global _ACTIVE_LOG_PATH, _ACTIVE_LOG_FILE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR

    if _ACTIVE_LOG_FILE is not None:
        stop_run_log()

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in notebook_name
    ).strip("_") or "notebook"
    log_path = reports / f"{safe_name}.txt"

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    t0 = datetime.now().isoformat(timespec="seconds")
    py = sys.version.split()[0]
    log_file.write(f"# HydrAI | {notebook_name} | {t0} | Python {py}\n")
    log_file.flush()

    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    _ACTIVE_LOG_FILE = log_file
    _ACTIVE_LOG_PATH = log_path

    print(f"[run_log] {log_path} (overwrite)")
    return log_path


def stop_run_log() -> None:
    """Restore original stdout/stderr and close the log file (optional)."""
    global _ACTIVE_LOG_PATH, _ACTIVE_LOG_FILE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR

    if _ACTIVE_LOG_FILE is None:
        return
    try:
        if _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        if _ORIGINAL_STDERR is not None:
            sys.stderr = _ORIGINAL_STDERR
        _ACTIVE_LOG_FILE.write(
            f"\n# end {datetime.now().isoformat(timespec='seconds')}\n"
        )
        _ACTIVE_LOG_FILE.flush()
        _ACTIVE_LOG_FILE.close()
    finally:
        _ACTIVE_LOG_FILE = None
        _ACTIVE_LOG_PATH = None
        _ORIGINAL_STDOUT = None
        _ORIGINAL_STDERR = None
