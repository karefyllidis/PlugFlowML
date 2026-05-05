#!/usr/bin/env python3
"""
Run Main_2 (training data generation) for one SLURM task chunk.

Reads TASK_ID and NTASKS from environment. Partitions reactants so this process
only generates data for reactants where index % NTASKS == TASK_ID. Writes to
data/training/task_<TASK_ID> so parallel tasks do not overwrite each other.

Usage (from SLURM script):
  export TASK_ID=0 NTASKS=11
  python scripts/run_main2_slurm_chunk.py

NTASKS can be in the order of 100s (one process per CPU). Run that many
processes in parallel, each with TASK_ID=0..NTASKS-1.

On a single Windows/macOS/Linux workstation (no SLURM), use::

    python scripts/run_main2_local_parallel.py --ntasks 4
"""

import os
import sys
import json
from pathlib import Path

# Project root: script lives in scripts/; project root is parent
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
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


def main():
    task_id = int(os.environ.get("TASK_ID", "0"))
    ntasks = int(os.environ.get("NTASKS", "1"))

    config_file = project_root / "configs" / "ml_data_generation_config.json"
    if not config_file.exists():
        print(f"[ERROR] Config not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    with open(config_file, "r") as f:
        config = json.load(f)

    reactants = config.get("reactants", ["ethane"])
    output_dir = config.get("output_dir", "data/training")
    out_path = Path(output_dir) / f"task_{task_id}"
    out_path.mkdir(parents=True, exist_ok=True)

    max_combinations = config.get("max_combinations_per_reactant", 100)
    save_interval = config.get("save_interval", 10)
    sampling_method = config.get("sampling_method", "latin_hypercube")
    lhs_seed = config.get("lhs_seed", 42)
    random_sample_bounds = config.get("random_sample_bounds")
    param_ranges = config.get("parameter_ranges", {})
    # Per-process single-threaded; parallelism is across SLURM tasks
    n_jobs = 1

    generator = TrainingDataGenerator(output_dir=str(out_path), disable_plots=True)
    if param_ranges:
        for key, value in param_ranges.items():
            if isinstance(value, list) and len(value) == 3:
                generator.param_ranges[key] = np.linspace(value[0], value[1], value[2])

    print(f"[TASK {task_id}/{ntasks}] Output: {out_path} (chunk by simulation index)")

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
    )

    print(f"[TASK {task_id}] Done. Output in {out_path}")


if __name__ == "__main__":
    main()
