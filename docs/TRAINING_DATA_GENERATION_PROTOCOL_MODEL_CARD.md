# Training Data Generation Protocol Model Card

## Purpose

This document defines the **data generation protocol** used to produce HydrAI training datasets for downstream notebooks and surrogate model training.

It is a process model card (how data is produced), not a trained-model performance card.

---

## Protocol Scope

- **System:** Steam-cracking plug-flow reactor simulations
- **Solver:** Cantera-based physics simulation pipeline
- **Output type:** Supervised ML tabular data (features/targets by axial position)
- **Pipeline stage:** Primarily `Main_2_generate_training_data` and `scripts/cluster/run_main2_slurm_chunk.py`

---

## Inputs (Generation Controls)

- **Simulation configs:** `configs/simulation/`
- **ML sweep config:** `configs/ml/ml_data_generation_config.json` (or smoke variant)
- **Feed/mechanism mapping:** `configs/simulation/reactant_database.json`
- **Sampling method:** Latin Hypercube (`latin_hypercube`) or grid/random (config-driven)
- **Key sweep variables:**
  - `initial_temperature_K`
  - `initial_pressure_Pa`
  - `reactor_length_m`
  - `reactor_diameter_m`
  - `mass_flow_rate_kgps`
  - `heat_flux_Wm2`

---

## Outputs (Artifacts)

- **Per-task outputs:** `data/training/task_<N>/training_data_complete_<timestamp>.pkl`
- **Per-task metadata:** `data/training/task_<N>/metadata_<timestamp>.json`
- **Progress logs:** `logs/data_generation_progress_task_<TASK_ID>.json`
- **Consolidated dataset:** `data/training/training_data_complete_<timestamp>.pkl`
- **Consolidated metadata:** `data/training/metadata_<timestamp>.json`

Consolidation is performed with:

`python scripts/dev/consolidate_training_data.py`

Consolidation options:

- Default: merge + write consolidated files + clean old per-task task folders/files
- Keep task outputs: `python scripts/dev/consolidate_training_data.py --no-cleanup`
- Preview only: `python scripts/dev/consolidate_training_data.py --dry-run`

---

## Data Schema (High-Level)

- **Feature families:**
  - Inlet conditions
  - Reactor design variables
  - Operating conditions
  - Spatial coordinate (`z_position_m`, `relative_position`)
- **Target families:**
  - State variables (temperature, pressure, velocity, density)
  - Thermodynamic properties
  - Species mass fractions (`Y_*`)
  - Raw exports may still list mole fractions (`X_*`), but **ML targets in Main_3/Main_4 and `model_training.py` use mass fractions only** (`Y_*` / `Y_lump_*`).

Notebook alignment:

- `notebooks/Main_3_data_exploration_feature_engineering.ipynb` loads the latest consolidated `training_data_complete_*.pkl`
- Exports `df_features` and `df_target` into `data/processed/`

---

## Quality Controls

- Drop NaNs during notebook load stage before EDA/feature export
- Track `total_simulations`, `successful_simulations`, `failed_simulations` in metadata
- Monitor runs via:
  - `bash scripts/dev/monitor_run.sh`
  - `python scripts/dev/check_complete_runs.py`
- For empty task chunks, generation logic skips gracefully (no false failure signal)

---

## HPC / Execution Protocol (CSD3-Tuned Defaults)

- Run from repository root
- SLURM scripts under `scripts/cluster/` are tuned for Cambridge CSD3
- GPU smoke protocol uses short interactive-style budget (`--time=00:10:00`, `--qos=INTR`)
- Production protocol uses longer non-interactive allocation and timeout protection to avoid hanging jobs
- Parallel step pattern uses `--ntasks=N --cpus-per-task=1` to match step-level worker model

---

## Known Limitations

- Dataset quality is bounded by mechanism validity and sweep domain coverage
- Extrapolation outside sampled operating space is not guaranteed reliable
- Failed combinations may cluster in physically unrealistic regions; inspect metadata before training
- Consolidation currently merges latest task artifact per task folder; archived runs should be separated clearly

---

## Reproducibility Checklist

- Record exact config files used (`configs/ml/` and `configs/simulation/`)
- Record mechanism file versions and feed definitions
- Keep SLURM job metadata (`logs/RUN_ROOT.txt`, progress JSON files)
- Preserve consolidated metadata JSON with every merged dataset
- Version control code and docs changes before training

---

## Recommended Hand-off to Modeling

1. Generate data (local or SLURM parallel)
2. Validate completion and success ratios
3. Consolidate task outputs into one training dataset
4. Run `Main_3_data_exploration_feature_engineering.ipynb`
5. Export validated `df_features` and `df_target`
6. Train/evaluate surrogates in `Main_4*` notebooks

