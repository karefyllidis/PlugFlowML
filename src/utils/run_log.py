"""Capture notebook terminal output to a .txt file under outputs/reports/.

Usage in a notebook (first code cell, after imports/paths are set):

    from src.utils.run_log import start_run_log
    log_path = start_run_log('Main_3')        # writes outputs/reports/Main_3_run_YYYYMMDD_HHMMSS.txt
    print('Hello')                            # appears in notebook and in the .txt file

The helper mirrors stdout/stderr to a file while still echoing to the notebook
display. A timestamp suffix prevents overwriting earlier runs.
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
        return getattr(self._primary, 'isatty', lambda: False)()


_ACTIVE_LOG_PATH: Path | None = None
_ACTIVE_LOG_FILE: TextIO | None = None
_ORIGINAL_STDOUT: TextIO | None = None
_ORIGINAL_STDERR: TextIO | None = None


def start_run_log(notebook_name: str,
                  reports_dir: str | Path = 'outputs/reports') -> Path:
    """Start mirroring stdout/stderr to a timestamped .txt under reports_dir.

    Calling start_run_log twice in the same kernel reuses the same file for
    that kernel session (subsequent calls just print the path) so re-running
    the setup cell does not spawn dozens of partial logs.

    Returns the Path to the active log file.
    """
    global _ACTIVE_LOG_PATH, _ACTIVE_LOG_FILE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR

    if _ACTIVE_LOG_FILE is not None and _ACTIVE_LOG_PATH is not None:
        print(f'[run_log] Already logging to {_ACTIVE_LOG_PATH}')
        return _ACTIVE_LOG_PATH

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_'
                        for ch in notebook_name).strip('_') or 'notebook'
    log_path = reports / f'{safe_name}_run_{timestamp}.txt'

    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
    header = (
        f'# HydrAI run log\n'
        f'# Notebook:  {notebook_name}\n'
        f'# Started:   {datetime.now().isoformat(timespec="seconds")}\n'
        f'# Python:    {sys.version.split()[0]}\n'
        f'# Log file:  {log_path}\n'
        f'{"-" * 72}\n'
    )
    log_file.write(header)
    log_file.flush()

    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    _ACTIVE_LOG_FILE = log_file
    _ACTIVE_LOG_PATH = log_path

    print(f'[run_log] Capturing notebook output to {log_path}')
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
            f'\n{"-" * 72}\n# Stopped: {datetime.now().isoformat(timespec="seconds")}\n'
        )
        _ACTIVE_LOG_FILE.flush()
        _ACTIVE_LOG_FILE.close()
    finally:
        _ACTIVE_LOG_FILE = None
        _ACTIVE_LOG_PATH = None
        _ORIGINAL_STDOUT = None
        _ORIGINAL_STDERR = None
