#!/usr/bin/env python3
"""Consolidate per-task training data into a single dataset for ML pipeline.

Usage:
    python scripts/dev/consolidate_training_data.py            # merge + save (default)
    python scripts/dev/consolidate_training_data.py --dry-run  # preview only (no writes)

After a parallel SLURM run, each task writes to data/training/task_<N>/.
This script merges all training_data_complete_*.pkl files into a single
dataset at data/training/training_data_complete_<timestamp>.pkl, which is
the format expected by notebooks/Main_3_data_exploration_feature_engineering.ipynb.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Resolve project root (script lives in scripts/dev/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.dataframe_pickle import load_dataframe_pickle, save_dataframe_pickle


def discover_task_outputs(base_dir: Path) -> dict[int, dict]:
    """Scan task_* directories for training outputs.
    
    Returns dict: task_id -> {'pkl': Path, 'meta': Path or None, 'rows': int}
    """
    tasks = {}
    for task_dir in sorted(base_dir.glob("task_*")):
        if not task_dir.is_dir():
            continue
        try:
            task_id = int(task_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        pkl_files = sorted(task_dir.glob("training_data_complete_*.pkl"), reverse=True)
        meta_files = sorted(task_dir.glob("metadata_*.json"), reverse=True)

        if not pkl_files:
            continue  # no data for this task

        pkl_path = pkl_files[0]  # latest
        meta_path = meta_files[0] if meta_files else None

        # Quick row count from metadata if available
        rows = 0
        if meta_path and meta_path.exists():
            try:
                with open(meta_path) as f:
                    m = json.load(f)
                    rows = m.get("total_rows", 0)
            except Exception:
                pass

        tasks[task_id] = {"pkl": pkl_path, "meta": meta_path, "rows": rows}

    return tasks


def load_and_merge(tasks: dict[int, dict], verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load all task pickles and merge into one DataFrame + combined metadata."""
    dfs = []
    combined_meta = {
        "source_tasks": [],
        "total_simulations": 0,
        "successful_simulations": 0,
        "failed_simulations": 0,
        "reactants": set(),
        "parameter_ranges": {},
    }

    for task_id in sorted(tasks.keys()):
        info = tasks[task_id]
        if verbose:
            print(f"  Loading task {task_id}: {info['pkl'].name} ...")
        df = load_dataframe_pickle(info["pkl"])
        dfs.append(df)
        combined_meta["source_tasks"].append(
            {"task_id": task_id, "file": str(info["pkl"]), "rows": len(df)}
        )

        # Merge metadata
        if info["meta"] and info["meta"].exists():
            with open(info["meta"]) as f:
                m = json.load(f)
            combined_meta["total_simulations"] += m.get("total_simulations", 0)
            combined_meta["successful_simulations"] += m.get("successful_simulations", 0)
            combined_meta["failed_simulations"] += m.get("failed_simulations", 0)
            for r in m.get("reactants", []):
                combined_meta["reactants"].add(r)
            # Merge parameter ranges (take union of min/max)
            for k, v in m.get("parameter_ranges", {}).items():
                if k not in combined_meta["parameter_ranges"]:
                    combined_meta["parameter_ranges"][k] = list(v)
                else:
                    combined_meta["parameter_ranges"][k][0] = min(
                        combined_meta["parameter_ranges"][k][0], v[0]
                    )
                    combined_meta["parameter_ranges"][k][1] = max(
                        combined_meta["parameter_ranges"][k][1], v[1]
                    )

    combined_meta["reactants"] = sorted(combined_meta["reactants"])

    if not dfs:
        return pd.DataFrame(), combined_meta

    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df, combined_meta


def cleanup_task_outputs(base_dir: Path, tasks: dict[int, dict], verbose: bool = True) -> dict[str, int]:
    """Delete old per-task merged artifacts after successful consolidation."""
    deleted_files = 0
    removed_dirs = 0

    for task_id in sorted(tasks.keys()):
        task_dir = base_dir / f"task_{task_id}"
        if not task_dir.exists():
            continue

        # Remove all old per-task final artifacts; consolidated file is now canonical.
        for pattern in ("training_data_complete_*.pkl", "metadata_*.json"):
            for path in task_dir.glob(pattern):
                try:
                    path.unlink()
                    deleted_files += 1
                    if verbose:
                        print(f"  Deleted: {path}")
                except Exception as exc:
                    print(f"  [WARN] Could not delete {path}: {exc}")

        # Remove task directory if it became empty.
        try:
            next(task_dir.iterdir())
        except StopIteration:
            try:
                task_dir.rmdir()
                removed_dirs += 1
                if verbose:
                    print(f"  Removed empty dir: {task_dir}")
            except Exception as exc:
                print(f"  [WARN] Could not remove directory {task_dir}: {exc}")
        except Exception:
            pass

    return {"deleted_files": deleted_files, "removed_dirs": removed_dirs}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview inputs only (do not save merged outputs)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "training"),
        help="Base directory containing task_* folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for merged file (default: same as base-dir)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep per-task files after successful merge",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir

    print("=" * 60)
    print("TRAINING DATA CONSOLIDATION")
    print("=" * 60)
    print(f"Base dir : {base_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Mode     : {'DRY-RUN (preview only)' if args.dry_run else 'APPLY (will save)'}")
    print(f"Cleanup  : {'disabled (--no-cleanup)' if args.no_cleanup else 'enabled (default)'}")
    print()

    # Discover tasks
    tasks = discover_task_outputs(base_dir)
    if not tasks:
        print("[WARN] No task_* directories with training_data_complete_*.pkl found.")
        print("       Run data generation first, or check the base directory.")
        sys.exit(0)

    print(f"Found {len(tasks)} task(s) with training data:")
    total_rows_est = 0
    for tid in sorted(tasks.keys()):
        info = tasks[tid]
        print(f"  task_{tid:02d}: {info['pkl'].name}  (~{info['rows']:,} rows)")
        total_rows_est += info["rows"]
    print(f"  Estimated total: ~{total_rows_est:,} rows")
    print()

    if args.dry_run:
        print("[DRY-RUN] No changes made. Re-run without --dry-run to merge and save.")
        sys.exit(0)

    # Load and merge
    print("Merging datasets ...")
    merged_df, combined_meta = load_and_merge(tasks, verbose=True)

    if merged_df.empty:
        print("[ERROR] Merged DataFrame is empty. Check source files.")
        sys.exit(1)

    # Save consolidated output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pkl = output_dir / f"training_data_complete_{timestamp}.pkl"
    output_meta = output_dir / f"metadata_{timestamp}.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe_pickle(merged_df, output_pkl)
    print(f"\n[OK] Saved merged dataset: {output_pkl}")
    print(f"     Rows: {len(merged_df):,}  Columns: {len(merged_df.columns)}")
    print(f"     Size: {os.path.getsize(output_pkl) / 1e6:.2f} MB")

    # Save combined metadata
    combined_meta["consolidation_date"] = datetime.now().isoformat()
    combined_meta["total_rows"] = len(merged_df)
    combined_meta["total_columns"] = len(merged_df.columns)
    combined_meta["success_rate"] = (
        100 * combined_meta["successful_simulations"] / combined_meta["total_simulations"]
        if combined_meta["total_simulations"] > 0
        else 0
    )

    with open(output_meta, "w") as f:
        json.dump(combined_meta, f, indent=2)
    print(f"[OK] Saved metadata: {output_meta}")

    cleanup_stats = {"deleted_files": 0, "removed_dirs": 0}
    if not args.no_cleanup:
        print("\nCleaning old per-task data ...")
        cleanup_stats = cleanup_task_outputs(base_dir, tasks, verbose=True)
        print(
            f"[OK] Cleanup complete: deleted_files={cleanup_stats['deleted_files']}, "
            f"removed_dirs={cleanup_stats['removed_dirs']}"
        )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Tasks merged       : {len(tasks)}")
    print(f"  Total rows         : {len(merged_df):,}")
    print(f"  Total simulations  : {combined_meta['total_simulations']:,}")
    print(f"  Successful         : {combined_meta['successful_simulations']:,}")
    print(f"  Failed             : {combined_meta['failed_simulations']:,}")
    print(f"  Success rate       : {combined_meta['success_rate']:.1f}%")
    print(f"  Reactants          : {combined_meta['reactants']}")
    print(
        f"  Cleanup            : deleted {cleanup_stats['deleted_files']} files, "
        f"removed {cleanup_stats['removed_dirs']} empty task dirs"
    )
    print()
    print("Next step: Open notebooks/Main_3_data_exploration_feature_engineering.ipynb")
    print(f"           It will auto-detect: {output_pkl.name}")


if __name__ == "__main__":
    main()
