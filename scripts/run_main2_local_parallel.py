#!/usr/bin/env python3
"""
Launch several ``run_main2_slurm_chunk.py`` workers on one machine.

Same chunking contract as SLURM (``TASK_ID``, ``NTASKS``), but uses subprocess
so it runs on **Windows**, macOS, and Linux without bash or ``srun``.

Usage (from project root)::

    python scripts/run_main2_local_parallel.py --ntasks 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--ntasks",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4). Each gets TASK_ID 0..ntasks-1.",
    )
    args = p.parse_args()
    if args.ntasks < 1:
        sys.exit("[ERROR] --ntasks must be >= 1")

    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "run_main2_slurm_chunk.py"
    if not script.exists():
        print(f"[ERROR] Missing {script}", file=sys.stderr)
        sys.exit(1)

    procs: list[subprocess.Popen] = []
    for i in range(args.ntasks):
        env = os.environ.copy()
        env["TASK_ID"] = str(i)
        env["NTASKS"] = str(args.ntasks)
        procs.append(
            subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(root),
                env=env,
            )
        )

    failed = 0
    for proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed += 1
            print(f"[WARN] Worker exited with code {rc}", file=sys.stderr)

    if failed:
        sys.exit(1)
    print(f"[OK] All {args.ntasks} workers finished successfully.")


if __name__ == "__main__":
    main()
