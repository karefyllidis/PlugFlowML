#!/usr/bin/env python3
"""
Run Main_2 (training data generation) for one SLURM task chunk.

Reads TASK_ID and NTASKS from environment. Partitions reactants so this process
only generates data for reactants where index % NTASKS == TASK_ID. Writes to
data/training/task_<TASK_ID> so parallel tasks do not overwrite each other.

- Creates temp/conditions_run_task_<TASK_ID>.csv and passes conditions_log_path,
  conditions_to_basename, and append_condition_log to generate_dataset(). The
  generator should: (1) save each run as <conditions_to_basename(conditions)>.pkl,
  (2) call append_condition_log(conditions_log_path, conditions) after each run.
- Creates temp/completed_runs_task_<TASK_ID>.txt and passes record_completed_run
  so the generator can append "m/n" (e.g. 1/23) after each run completes.

Usage (from SLURM script):
  export TASK_ID=0 NTASKS=11
  python scripts/cluster/run_main2_slurm_chunk.py

Optional:
  export PLUGFLOWML_ML_CONFIG=/path/to/smoke_or_custom.json
Progress file (updated after each simulation for this task):
  logs/data_generation_progress_task_<TASK_ID>.json
"""

import csv
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Project root: script lives in scripts/cluster/; project root is two levels up
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# Import after path is set; cantera must be imported before our src
import numpy as np
import cantera as ct
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("cantera").setLevel(logging.CRITICAL)
logging.getLogger("sundials").setLevel(logging.CRITICAL)

from src.ml.data_generation import TrainingDataGenerator

# Input parameter names (for filename and conditions log)
PARAM_KEYS = [
    "temperature_K", "pressure_bar", "length_m", "diameter_mm",
    "mass_flow_rate_kgps", "heat_flux_Wm2",
]


def conditions_to_basename(conditions: Dict[str, Any]) -> str:
    """Build a short, filesystem-safe filename stem from input conditions."""
    parts = []
    for k in PARAM_KEYS:
        v = conditions.get(k)
        if v is None:
            parts.append("_")
            continue
        try:
            x = float(v)
            if k == "temperature_K":
                parts.append(f"T{int(round(x))}")
            elif k == "pressure_bar":
                parts.append(f"p{x:.2f}".replace(".", "p"))
            elif k == "length_m":
                parts.append(f"L{x:.1f}".replace(".", "p"))
            elif k == "diameter_mm":
                parts.append(f"d{x:.1f}".replace(".", "p"))
            elif k == "mass_flow_rate_kgps":
                parts.append(f"mfr{x:.3f}".replace(".", "p"))
            elif k == "heat_flux_Wm2":
                parts.append(f"q{int(round(x))}")
            else:
                parts.append(str(x).replace(".", "p"))
        except (TypeError, ValueError):
            parts.append(str(v).replace(".", "p").replace(" ", "_"))
    return "_".join(parts)


def append_condition_log(csv_path: str, conditions: Dict[str, Any]) -> None:
    """Append one row (current run's input conditions) to the temp conditions log."""
    if not csv_path:
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: conditions.get(k, "") for k in PARAM_KEYS}
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PARAM_KEYS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _make_record_completed_run(txt_path: Path, total: int):
    """Return a callback that appends 'm/n' to completed_runs txt after each run."""
    if total <= 0:
        return lambda m, n: None
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(f"0/{total}\n")

    def record_completed_run(completed: int, total_n: int) -> None:
        with open(txt_path, "a") as f:
            f.write(f"{completed}/{total_n}\n")

    return record_completed_run


def main():
    task_id = int(os.environ.get("TASK_ID", "0"))
    ntasks = int(os.environ.get("NTASKS", "1"))

    cfg_override = (os.environ.get("PLUGFLOWML_ML_CONFIG") or "").strip()
    if cfg_override:
        config_file = Path(cfg_override)
        if not config_file.is_absolute():
            config_file = (project_root / config_file).resolve()
    else:
        config_file = project_root / "configs" / "ml" / "main2_data_generation_config.json"
    if not config_file.exists():
        print(f"[ERROR] Config not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    with open(config_file, "r") as f:
        config = json.load(f)

    reactants = config.get("reactants", ["ethane"])
    output_dir = config.get("output_dir", "data/training")
    out_path = Path(output_dir) / f"task_{task_id}"
    out_path.mkdir(parents=True, exist_ok=True)

    # Temp file: one row per condition run (so we have a clear log of what ran)
    temp_dir = project_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    conditions_log_path = temp_dir / f"conditions_run_task_{task_id}.csv"
    with open(conditions_log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=PARAM_KEYS).writeheader()
    os.environ["CONDITIONS_LOG"] = str(conditions_log_path.resolve())

    max_combinations = config.get("max_combinations_per_reactant", 100)
    param_ranges = config.get("parameter_ranges", {})
    full_grid = 1
    for k, v in param_ranges.items():
        if isinstance(k, str) and (k.startswith("_") or k == "_comment"):
            continue
        if isinstance(v, (list, tuple)) and len(v) == 3:
            full_grid *= int(v[2])
    total_runs = min(full_grid, max_combinations) * len(reactants) if reactants else min(full_grid, max_combinations)
    n_runs_this_task = max(0, (total_runs - task_id + ntasks - 1) // ntasks) if ntasks else total_runs

    completed_runs_path = temp_dir / f"completed_runs_task_{task_id}.txt"
    record_completed_run = _make_record_completed_run(completed_runs_path, n_runs_this_task)
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = log_dir / f"data_generation_progress_task_{task_id}.json"
    save_interval = config.get("save_interval", 10)
    sampling_method = config.get("sampling_method", "latin_hypercube")
    lhs_seed = config.get("lhs_seed", 42)
    random_sample_bounds = config.get("random_sample_bounds")
    # Per-process single-threaded; parallelism is across SLURM tasks
    n_jobs = 1

    # Cluster chunk worker is data-generation only; EDA/coverage plots are handled in Main_3.
    generator = TrainingDataGenerator(output_dir=str(out_path), disable_plots=True)
    if param_ranges:
        for key, value in param_ranges.items():
            if isinstance(value, list) and len(value) == 3:
                generator.param_ranges[key] = np.linspace(value[0], value[1], value[2])

    print(
        f"[TASK {task_id}/{ntasks}] Output: {out_path}  Runs: 0/{n_runs_this_task}  "
        f"Conditions log: {conditions_log_path}  Progress JSON: {progress_path}"
    )

    generator.generate_dataset(
        reactants=reactants,
        max_combinations_per_reactant=max_combinations,
        save_interval=save_interval,
        random_sample_bounds=random_sample_bounds,
        n_jobs=n_jobs,
        sampling_method=sampling_method,
        lhs_seed=lhs_seed,
        save_metadata=True,
        save_training_data=True,
        task_id=task_id,
        ntasks=ntasks,
        # So generator can name pkl files by conditions and log each run to temp
        conditions_log_path=str(conditions_log_path),
        conditions_to_basename=conditions_to_basename,
        append_condition_log=append_condition_log,
        # Record each completed run as m/n in completed_runs txt
        record_completed_run=record_completed_run,
        n_runs_this_task=n_runs_this_task,
        progress_status_path=str(progress_path),
    )

    print(f"[TASK {task_id}] Done. Output in {out_path}")


if __name__ == "__main__":
    main()
