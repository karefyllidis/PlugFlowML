#!/usr/bin/env python3
"""
Clean and sort completed Main_2 run artifacts.

Default behavior is DRY-RUN (no file moves). Use --apply to execute.

What it does:
1) Scans task artifacts across logs/, temp/, and data/training/task_*
2) Detects completed tasks using either:
   - "[TASK N] Done." in log, or
   - completed_runs_task_N.txt last line shows m/m
3) Archives completed-task artifacts into:
   logs/archive/completed_runs_<timestamp>/task_<N>/

Run from project root:
  python scripts/dev/clean_completed_runs.py
  python scripts/dev/clean_completed_runs.py --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / "logs"
TEMP_DIR = ROOT / "temp"
TRAINING_DIR = ROOT / "data" / "training"


@dataclass
class TaskStatus:
    task_id: int
    has_done_log: bool = False
    completed_num: int | None = None
    completed_den: int | None = None
    n_output_files: int = 0

    @property
    def is_complete(self) -> bool:
        by_done = self.has_done_log
        by_ratio = (
            self.completed_num is not None
            and self.completed_den is not None
            and self.completed_den > 0
            and self.completed_num == self.completed_den
        )
        return by_done or by_ratio

    @property
    def progress_text(self) -> str:
        if self.completed_num is None or self.completed_den is None:
            return "n/a"
        return f"{self.completed_num}/{self.completed_den}"


def extract_task_id(name: str) -> int | None:
    m = re.search(r"(?:task_|task)(\d+)", name)
    return int(m.group(1)) if m else None


def parse_completed_ratio(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    lines = [ln.strip() for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    if not lines:
        return None, None
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", lines[-1])
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def gather_task_statuses() -> dict[int, TaskStatus]:
    statuses: dict[int, TaskStatus] = {}

    def get(task_id: int) -> TaskStatus:
        if task_id not in statuses:
            statuses[task_id] = TaskStatus(task_id=task_id)
        return statuses[task_id]

    # Logs: main2_task_N.log and progress json
    if LOG_DIR.exists():
        for log in LOG_DIR.glob("main2_task_*.log"):
            tid = extract_task_id(log.name)
            if tid is None:
                continue
            s = get(tid)
            text = log.read_text(errors="replace")
            if re.search(r"\[TASK\s+\d+\]\s+Done\.", text):
                s.has_done_log = True

        for progress in LOG_DIR.glob("data_generation_progress_task_*.json"):
            tid = extract_task_id(progress.name)
            if tid is not None:
                get(tid)

    # Temp: completed and conditions files
    if TEMP_DIR.exists():
        for comp in TEMP_DIR.glob("completed_runs_task_*.txt"):
            tid = extract_task_id(comp.name)
            if tid is None:
                continue
            s = get(tid)
            s.completed_num, s.completed_den = parse_completed_ratio(comp)

        for cond in TEMP_DIR.glob("conditions_run_task_*.csv"):
            tid = extract_task_id(cond.name)
            if tid is not None:
                get(tid)

    # Data outputs
    if TRAINING_DIR.exists():
        for task_dir in TRAINING_DIR.glob("task_*"):
            if not task_dir.is_dir():
                continue
            tid = extract_task_id(task_dir.name)
            if tid is None:
                continue
            s = get(tid)
            n = 0
            n += len(list(task_dir.glob("*.pkl")))
            n += len(list(task_dir.glob("*.pickle")))
            n += len(list(task_dir.glob("*.csv")))
            s.n_output_files = n

    return dict(sorted(statuses.items(), key=lambda kv: kv[0]))


def collect_files_for_task(task_id: int) -> list[Path]:
    files: list[Path] = []
    patterns = [
        (LOG_DIR, f"main2_task_{task_id}.log"),
        (LOG_DIR, f"data_generation_progress_task_{task_id}.json"),
        (TEMP_DIR, f"completed_runs_task_{task_id}.txt"),
        (TEMP_DIR, f"conditions_run_task_{task_id}.csv"),
    ]
    for base, pat in patterns:
        p = base / pat
        if p.exists():
            files.append(p)
    return files


def archive_completed(statuses: dict[int, TaskStatus], apply: bool) -> int:
    completed_ids = [tid for tid, st in statuses.items() if st.is_complete]
    if not completed_ids:
        print("\nNo completed tasks found. Nothing to archive.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = LOG_DIR / "archive" / f"completed_runs_{stamp}"
    print(f"\nArchive destination: {archive_root}")

    moved = 0
    for tid in completed_ids:
        task_files = collect_files_for_task(tid)
        if not task_files:
            continue
        target_dir = archive_root / f"task_{tid}"
        for src in task_files:
            dst = target_dir / src.name
            print(f"  {'MOVE' if apply else 'WOULD MOVE'} {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            if apply:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            moved += 1

    print(f"\n{'Moved' if apply else 'Would move'} {moved} file(s) from completed tasks.")
    return moved


def print_summary(statuses: dict[int, TaskStatus]) -> None:
    if not statuses:
        print("No task artifacts found under logs/, temp/, or data/training/task_*.")
        return

    print("\nTask summary:")
    print("task | complete | progress | outputs | done-log")
    print("-----+----------+----------+---------+---------")
    for tid, st in statuses.items():
        complete = "yes" if st.is_complete else "no"
        done_log = "yes" if st.has_done_log else "no"
        print(
            f"{tid:>4} | {complete:^8} | {st.progress_text:^8} | {st.n_output_files:^7} | {done_log:^7}"
        )

    total = len(statuses)
    complete_n = sum(1 for s in statuses.values() if s.is_complete)
    print(f"\nTasks discovered: {total}")
    print(f"Completed tasks: {complete_n}")
    print(f"Incomplete tasks: {total - complete_n}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean/sort completed run artifacts by archiving completed task files."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute file moves. Default is dry-run.",
    )
    args = parser.parse_args()

    print("HydrAI completed-run cleaner")
    print(f"Project root: {ROOT}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    statuses = gather_task_statuses()
    print_summary(statuses)
    archive_completed(statuses, apply=args.apply)

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to execute moves.")


if __name__ == "__main__":
    main()
