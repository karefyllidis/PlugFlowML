#!/usr/bin/env python3
"""
Check training data: how many simulations ran, how many are needed, how long to finish.
Run from HydrAI project root:  python scripts/dev/check_complete_runs.py
"""

import csv
import json
import pickle
import re
from pathlib import Path
from datetime import datetime

# Paths relative to project root (script in scripts/dev/)
ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data" / "training"
CONFIG = ROOT / "configs" / "ml" / "ml_data_generation_config.json"

SLURM_CPUS_DEFAULT = 224
SLURM_WALL_H = 1.0  # hours from #SBATCH --time=01:00:00


def _load_run_root_values():
    """Read key=value pairs from logs/RUN_ROOT.txt when available."""
    run_root = ROOT / "logs" / "RUN_ROOT.txt"
    vals = {}
    if not run_root.exists():
        return vals
    for line in run_root.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip()
    return vals


def main():
    print("\n" + "=" * 55)
    print("  HydrAI Training Data Check")
    print("=" * 55)

    # ---- 1. Read config to know what was intended ----
    with open(CONFIG) as f:
        cfg = json.load(f)

    reactants = cfg.get("reactants", [])
    max_comb = cfg.get("max_combinations_per_reactant", 100)
    method = cfg.get("sampling_method", "")
    param_ranges = cfg.get("parameter_ranges", {})

    full_grid = 1
    for k, v in param_ranges.items():
        if k.startswith("_") or not isinstance(v, list) or len(v) != 3:
            continue
        full_grid *= int(v[2])

    intended = min(full_grid, max_comb) * len(reactants)

    run_root_vals = _load_run_root_values()
    slurm_cpus = int(
        run_root_vals.get("SLURM_CPUS_ON_NODE")
        or run_root_vals.get("SLURM_NTASKS")
        or SLURM_CPUS_DEFAULT
    )

    print(f"\n  Config:")
    print(f"    Reactants:       {reactants}")
    print(f"    Method:          {method}")
    print(f"    Full grid:       {full_grid:,} (all combos of {len(param_ranges)-1} params x points)")
    print(f"    max_combinations_per_reactant: {max_comb}")
    print(f"    Intended runs:   {intended}")
    print(f"    CPUs used (estimate): {slurm_cpus}")

    # ---- 2. Count files and unique simulations ----
    task_dirs = sorted(DATA_DIR.iterdir()) if DATA_DIR.is_dir() else []
    task_dirs = [d for d in task_dirs if d.is_dir() and d.name.startswith("task_")]

    total_files = 0
    sims_with_output = 0  # task dirs that have at least one pkl/csv
    sims_by_metadata_success = 0
    sims_by_metadata_total = 0
    metadata_tasks = 0

    for td in task_dirs:
        files = list(td.glob("*.pkl")) + list(td.glob("*.pickle")) + list(td.glob("*.csv"))
        total_files += len(files)
        if files:
            sims_with_output += 1
        metadata_files = sorted(td.glob("metadata_*.json"))
        if metadata_files:
            metadata_tasks += 1
            try:
                with open(metadata_files[-1], "r", encoding="utf-8") as f:
                    md = json.load(f)
                sims_by_metadata_success += int(md.get("successful_simulations", 0))
                sims_by_metadata_total += int(md.get("total_simulations", 0))
            except Exception:
                pass

    # Each task dir = one SLURM task. Tasks 0..(intended-1) got a simulation,
    # tasks intended..223 got nothing. So sims_with_output = tasks that produced data.

    print(f"\n  Output:")
    print(f"    Task dirs:       {len(task_dirs)}")
    print(f"    Task dirs with output: {sims_with_output}")
    print(f"    Total files:     {total_files}")
    if metadata_tasks > 0:
        print(f"    Metadata tasks:  {metadata_tasks}")
        print(f"    Sims from metadata: success={sims_by_metadata_success}, total={sims_by_metadata_total}")

    # ---- 3. Peek at one file to understand structure ----
    sample = None
    for td in task_dirs:
        files = list(td.glob("*.pkl")) + list(td.glob("*.pickle"))
        if files:
            sample = files[0]
            break

    n_sims = 0
    if sample:
        with open(sample, "rb") as f:
            obj = pickle.load(f)

        print(f"\n  Sample file: {sample.name}")
        print(f"    Type: {type(obj).__name__}")

        if hasattr(obj, "columns"):
            print(f"    Rows: {len(obj)},  Columns: {len(obj.columns)}")
            # Input conditions = columns constant within this file
            const_cols = [c for c in obj.columns if obj[c].nunique() == 1]
            varying_cols = [c for c in obj.columns if obj[c].nunique() > 1]
            print(f"    Constant cols (= input conditions): {const_cols}")
            print(f"    Varying cols (= reactor profile):   {len(varying_cols)} cols")
        elif isinstance(obj, dict):
            print(f"    Keys: {list(obj.keys())[:15]}{'...' if len(obj) > 15 else ''}")
            const_cols = []
            for k, v in obj.items():
                if hasattr(v, "__len__") and not isinstance(v, str):
                    flat = v.ravel().tolist() if hasattr(v, "ravel") else list(v)
                    if len(set(flat)) == 1:
                        const_cols.append(k)
                elif not hasattr(v, "__len__"):
                    const_cols.append(k)
            print(f"    Constant keys (= input conditions): {const_cols}")
        elif isinstance(obj, (list, tuple)):
            print(f"    Length: {len(obj)}, element type: {type(obj[0]).__name__ if obj else '?'}")
            const_cols = []
        else:
            const_cols = []

        # Count unique sims and build manifest (one row per file with its input conditions)
        if const_cols:
            unique_sigs = set()
            manifest_rows = []
            for td in task_dirs:
                task_id = td.name.replace("task_", "")
                for fp in sorted(td.glob("*.pkl")) + sorted(td.glob("*.pickle")):
                    try:
                        with open(fp, "rb") as f:
                            d = pickle.load(f)
                        vals = []
                        n_rows = 0
                        if hasattr(d, "columns"):
                            n_rows = len(d)
                            for c in const_cols:
                                if c in d.columns:
                                    vals.append(d[c].iloc[0])
                        elif isinstance(d, dict):
                            for c in const_cols:
                                if c in d:
                                    v = d[c]
                                    if hasattr(v, "__len__") and not isinstance(v, str):
                                        n_rows = n_rows or (len(v) if not hasattr(v, "shape") else v.shape[0])
                                        vals.append(v.ravel()[0] if hasattr(v, "ravel") else v[0])
                                    else:
                                        vals.append(v)
                        if vals:
                            sig = tuple(round(x, 8) if isinstance(x, float) else x for x in vals)
                            unique_sigs.add(sig)
                            row = {"task_id": task_id, "file": fp.name}
                            for c, val in zip(const_cols, vals):
                                row[c] = val
                            row["n_rows"] = n_rows
                            manifest_rows.append(row)
                    except Exception:
                        continue
            n_sims = len(unique_sigs)

            # Export manifest and unique conditions so you can see exactly what ran
            out_dir = ROOT / "data" / "training"
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = out_dir / "training_manifest.csv"
            unique_path = out_dir / "unique_conditions_run.csv"
            if manifest_rows:
                with open(manifest_path, "w", newline="") as f:
                    cols = ["task_id", "file"] + const_cols + ["n_rows"]
                    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(manifest_rows)
                print(f"\n  Exported: {manifest_path}  ({len(manifest_rows)} rows = one per output file)")

                unique_conds = sorted(unique_sigs)
                with open(unique_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(const_cols)
                    w.writerows(unique_conds)
                print(f"  Exported: {unique_path}  ({len(unique_conds)} rows = unique input conditions)")

    # ---- 4. Check logs for "Done" and timing ----
    logs = sorted(LOG_DIR.glob("main2_task_*.log")) if LOG_DIR.is_dir() else []
    n_done = 0
    for log in logs:
        text = log.read_text(errors="replace")
        if re.search(r"\[TASK\s+\d+\]\s+Done\.", text):
            n_done += 1

    # ---- 5. CLEAR ANSWERS ----
    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)

    print(f"\n  Q: How many simulations have been run?")
    if sims_by_metadata_success > 0:
        print(
            f"  A: {sims_by_metadata_success} successful simulations "
            f"(from per-task metadata; attempted={sims_by_metadata_total})"
        )
    elif n_sims > 0:
        print(f"  A: {n_sims} unique simulations (from {total_files} output files)")
    else:
        print(f"  A: {sims_with_output} task dirs have output ({total_files} files)")
        print(f"     (could not determine exact unique sim count)")

    print(f"\n  Q: How many are needed?")
    print(f"  A: {intended} (config max_combinations_per_reactant = {max_comb})")
    print(f"     Full grid would be {full_grid:,} (if you raise max_combinations)")

    done = (
        sims_by_metadata_success
        if sims_by_metadata_success > 0
        else (n_sims if n_sims > 0 else sims_with_output)
    )
    remaining = max(0, intended - done)
    print(f"\n  Q: How many are missing?")
    print(f"  A: {done} / {intended} done  →  {remaining} remaining")
    if n_done == 0 and done > 0:
        print(f"     (Note: 0 tasks printed 'Done' in logs — all were killed by")
        print(f"      the {SLURM_WALL_H}h time limit, but {done} produced partial/full output)")

    print(f"\n  Q: How long to complete?")
    if done > 0:
        cpu_h_used = slurm_cpus * SLURM_WALL_H
        h_per_sim = cpu_h_used / done
        print(f"  A: Based on this run: {done} sims in {SLURM_WALL_H}h using {slurm_cpus} CPUs")
        print(f"     → ~{h_per_sim * 60:.1f} CPU-min per simulation")
        if remaining > 0:
            cpu_h_remain = remaining * h_per_sim
            wall_h = cpu_h_remain / slurm_cpus
            print(f"     Remaining {remaining} sims need ~{cpu_h_remain:.0f} CPU-h")
            print(f"     With {slurm_cpus} CPUs: ~{wall_h:.1f}h wall time ({wall_h / SLURM_WALL_H:.0f} more jobs)")
        else:
            print(f"     All {intended} intended runs appear complete!")

        if full_grid > intended:
            cpu_h_full = full_grid * h_per_sim
            wall_h_full = cpu_h_full / slurm_cpus
            print(f"\n     For the FULL GRID ({full_grid:,} sims):")
            print(f"     → {cpu_h_full:,.0f} CPU-hours total")
            print(f"     → {wall_h_full:,.0f}h wall time with {slurm_cpus} CPUs ({wall_h_full / 24:,.0f} days)")
            for cpus in [500, 1000, 2000]:
                wh = cpu_h_full / cpus
                print(f"     → {wh:,.0f}h with {cpus} CPUs ({wh / 24:,.0f} days)")
    else:
        print(f"  A: Cannot estimate — no completed simulations detected.")

    print("\n" + "=" * 55)

    # ---- 6. Write report to file ----
    report_path = ROOT / "logs" / "check_complete_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Write summary to file
    with open(report_path, "w") as f:
        f.write(f"HydrAI Training Data Check - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"{'=' * 55}\n")
        f.write(f"Reactants: {reactants}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Full grid: {full_grid:,}\n")
        f.write(f"max_combinations: {max_comb}\n")
        f.write(f"CPUs used (estimate): {slurm_cpus}\n")
        f.write(f"Intended: {intended}\n")
        f.write(f"Task dirs with output: {sims_with_output}\n")
        f.write(f"Total files: {total_files}\n")
        if sims_by_metadata_success > 0:
            f.write(
                "Simulations (metadata): "
                f"successful={sims_by_metadata_success}, total={sims_by_metadata_total}\n"
            )
        else:
            f.write(f"Unique simulations: {n_sims if n_sims else '?'}\n")
        f.write(f"Log 'Done': {n_done}/{len(logs)}\n")
        f.write(f"Done: {done}/{intended}\n")
        f.write(f"Remaining: {remaining}\n")
        if done > 0:
            h_per_sim = slurm_cpus * SLURM_WALL_H / done
            f.write(f"Est. CPU-min/sim: {h_per_sim * 60:.1f}\n")
            if full_grid > intended:
                cpu_h_full = full_grid * h_per_sim
                f.write(f"Est. CPU-h for full grid: {cpu_h_full:,.0f}\n")
                f.write(f"Est. wall-h for full grid ({slurm_cpus} CPUs): {cpu_h_full / slurm_cpus:,.0f}\n")
        f.write(f"\nExports (what has run):\n")
        f.write(f"  data/training/training_manifest.csv   - one row per file, with input conditions\n")
        f.write(f"  data/training/unique_conditions_run.csv - one row per unique (T,p,L,d,mfr,q)\n")
    print(f"\n  Report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
