# HydrAI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Cantera](https://img.shields.io/badge/Cantera-3.2.0%2B-green)](https://cantera.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/karefyllidis/HydrAI)

> **HydrAI** = **Hydr**ocarbon + **AI** — A physics-based simulation framework with ML surrogate models for steam cracking reactors.

Nikolas Karefyllidis, PhD — [github.com/karefyllidis](https://github.com/karefyllidis)

---

## Overview

Steam cracking is one of the most energy-intensive processes in the chemical industry, converting hydrocarbons into ethylene, propylene, and other olefins at high temperatures. Accurate simulation requires solving stiff ODEs coupled with detailed chemical kinetics (100–500+ species), making each run computationally expensive.

**HydrAI** solves this in two stages:

1. **High-fidelity simulation** — A Cantera-based plug flow reactor (PFR) solver with detailed kinetic mechanisms, heat flux profiles, and pressure drop calculations (Churchill correlation). Produces full axial profiles of temperature, pressure, species concentrations, and thermodynamic/transport properties.

2. **ML surrogate layer** — Tree-based surrogate models (Random Forest, Gradient Boosting, XGBoost, AdaBoost) trained on simulation data to predict reactor outlet conditions and full axial profiles **100–1000× faster** than the Cantera solver. The training data pipeline supports **Latin Hypercube Sampling** and **structured grid sweeps** over 6 operating parameters, with parallel generation on HPC clusters via SLURM.

---

## Pipeline

The workflow is structured as five sequential notebooks:

```
Main_1 → Main_2 → Main_3 → Main_4 → Main_4b
  PFR      Data    Feature   Train    Compare
  Sim     Sweep     Eng.    Models    Models
```

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| `Main_1_run_pfr.ipynb` | Single PFR simulation, parameter configuration, 18+ profile plots | Axial profiles, CSV/figures |
| `Main_2_generate_training_data.ipynb` | Parameterised sweep (LHS / grid / random) over 6 operating variables | `data/training/*.pkl` — 152k+ row dataset |
| `Main_3_data_exploration_feature_engineering.ipynb` | EDA, NaN cleaning, feature/target split, export | `data/processed/features_targets_*.pkl` |
| `Main_4_train_tree_models.ipynb` | Train RF / GB / XGBoost / AdaBoost, optional hyperparameter search, feature importance | `models/tree_models_*.joblib` |
| `Main_4b_tree_models_comparison.ipynb` | Evaluate all models: R², MAE, RMSE, MAPE, MedAE, MaxError, MBE | Comparison plots and metrics |

---

## Technical Highlights

### Physics Simulation (Main_1)
- Cantera PFR with detailed kinetic mechanisms (YAML format, 150–500 species)
- Configurable heat flux profile (piecewise linear or step-wise, relative position 0–1)
- Churchill correlation for pressure drop
- Reactants: **ethane, propane, n-hexane, naphtha**
- Exports 245+ columns: full species mass/mole fractions, T, P, velocity, density, Cp, Cv, enthalpy, thermal conductivity, viscosity

### Training Data Generation (Main_2)
- **6 swept parameters** with configurable ranges:

| Parameter | Default Range |
|-----------|--------------|
| Temperature | 800 – 1200 K |
| Pressure | 1.5 – 3.5 bar |
| Reactor length | 3 – 15 m |
| Reactor diameter | 20 – 40 mm |
| Mass flow rate | 0.05 – 0.10 kg/s |
| Heat flux | 100,000 – 200,000 W/m² |

- Sampling modes: **Latin Hypercube (LHS)**, **structured grid** (up to 1,000,000 combinations), **random**
- Parallel execution: `n_jobs=-1` (all local CPUs) or SLURM (100s of CPUs, see [HPC](#hpc-slurm))
- Incremental saves (every N simulations) with pickle format for memory efficiency
- Dataset size: **152,442 rows × 328 columns** for a single reactant (n-hexane, 153-species mechanism)

### Feature Engineering (Main_3)
- 8 input features: `initial_temperature_K`, `initial_pressure_Pa`, `reactor_length_m`, `reactor_diameter_m`, `mass_flow_rate_kgps`, `heat_flux_Wm2`, `z_position_m`, `relative_position`
- ~320 output targets: thermodynamic state variables + all species Y\*/X\* mass/mole fractions
- NaN audit and removal, column categorisation (inlet / reactor design / spatial / thermodynamic / species)
- Optional label encoding for `reactant_type` (multi-reactant datasets)
- Export as single pickle for reproducible downstream use

### Model Training (Main_4)
- **MultiOutputRegressor** wrapper: one regressor per output target
- Models and default hyperparameters (from `configs/ml_training_config.json`):

| Model | n_estimators | max_depth | Notes |
|-------|-------------|-----------|-------|
| Random Forest | 100 | 20 | `n_jobs=-1` |
| Gradient Boosting | 150 | 5 | Sequential, high accuracy |
| XGBoost | 150 | 6 | `n_jobs=-1` |
| AdaBoost | 200 | 6 (tree) | lr=0.1 |

- **Hyperparameter tuning**: `RandomizedSearchCV` (quick: N_iter=20, CV=3; full: N_iter=100, CV=5)
- Tuning subsample cap (`TUNING_MAX_SAMPLES`) prevents hanging on large datasets; best params are then refit on the full training set
- Test-set evaluation after training: R², MAE, RMSE per model before and after tuning
- Feature importance plots per output target

### Model Comparison (Main_4b)
Metrics computed on held-out test set per model and per output target:
R², MAE, Median AE, RMSE, NRMSE, MAPE, Max Error, Mean Bias Error (MBE)

---

## Project Structure

```
HydrAI/
├── notebooks/
│   ├── Main_1_run_pfr.ipynb               # PFR simulation (interactive)
│   ├── Main_2_generate_training_data.ipynb # Training data generation
│   ├── Main_3_data_exploration_feature_engineering.ipynb
│   ├── Main_4_train_tree_models.ipynb      # Model training + tuning
│   └── Main_4b_tree_models_comparison.ipynb # Evaluation & comparison
│
├── src/
│   ├── cantera/
│   │   └── pfr_simulator.py               # PFR solver, heat flux, pressure drop
│   └── ml/
│       ├── data_generation.py             # TrainingDataGenerator (LHS, grid, parallel)
│       ├── model_training.py              # MLModelTrainer (RF, GB, XGBoost, AdaBoost, NN)
│       └── inference.py                   # MLPFRPredictor — fast surrogate inference
│
├── configs/
│   ├── ml_data_generation_config.json     # Sampling method, parameter ranges, reactants
│   ├── ml_training_config.json            # Model hyperparameters, test split
│   ├── heat_flux_profile.json             # Axial heat flux profile
│   ├── reactant_database.json             # Reactant properties and mechanisms
│   └── config_template.json              # Simulation config template
│
├── scripts/
│   ├── run_training_mul_CPUs.sh           # SLURM job (parallel Main_2, Linux/HPC)
│   ├── run_main2_local_parallel.py       # Local multi-process launcher (Windows/macOS/Linux)
│   ├── run_main2_slurm_chunk.py           # Per-task chunk runner (TASK_ID / NTASKS)
│   ├── run_simulation.sh                  # Single simulation convenience script
│   └── show_structure.sh                  # Print project structure
│
├── data/
│   ├── training/                          # Raw simulation sweeps (pkl)
│   └── processed/                         # Feature/target split for model training
│
├── models/                                # Exported joblib model artifacts
├── mechanisms/                            # Cantera YAML kinetic mechanisms
├── outputs/figures/                       # Generated plots
├── styles/figure_aesthetics.json          # Centralised plot styling
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/karefyllidis/HydrAI.git
cd HydrAI
pip install -r requirements.txt
```

**Core dependencies:** `cantera`, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, `joblib`, `jupyter`

### Windows (PowerShell or Command Prompt)

The Python stack runs the same way on Windows: `pip install -r requirements.txt`, then open the notebooks in Jupyter / VS Code / Cursor or run `python run_pipeline.py`. For a double-click path from File Explorer, use `run_pipeline.bat` in the repo root (expects `python` on your PATH).

**Cantera:** install a build that matches your Python version. The usual options are [Cantera’s Windows install guide](https://cantera.org/stable/install/windows.html) (Conda/Mamba is reliable) or, when available for your interpreter, `pip install cantera`.

**Parallel data generation without SLURM** (multi-process, works on Windows, macOS, Linux):

```powershell
python scripts/run_main2_local_parallel.py --ntasks 4
```

Or run a single chunk in one terminal:

```powershell
$env:TASK_ID = "0"
$env:NTASKS = "4"
python scripts/run_main2_slurm_chunk.py
```

Shell scripts under `scripts/*.sh` are optional convenience wrappers for Unix-like environments; they are not required on Windows.

### 2. Run a single PFR simulation

Open and run `notebooks/Main_1_run_pfr.ipynb`. Set your reactant and operating conditions in the configuration cell.

### 3. Generate training data

Configure `configs/ml_data_generation_config.json`, then run `Main_2_generate_training_data.ipynb`.

```json
{
  "reactants": ["n-hexane"],
  "sampling_method": "latin",
  "max_combinations_per_reactant": 500,
  "n_jobs": -1
}
```

### 4. Train surrogate models

Run `Main_3` (feature engineering) → `Main_4` (training) → `Main_4b` (comparison).

Or, to execute the three notebooks back-to-back from the command line:

```bash
python run_pipeline.py
```

On Windows, `run_pipeline.bat` does the same if `python` is on your PATH.

---

## Required External Files

The repository ships everything needed to run the pipeline **except** the
chemical kinetic mechanism files. They live in `mechanisms/` and are tracked,
but if you clone fresh and Cantera complains about a missing YAML file, the
mapping is:

| Reactant | Mechanism file (`mechanisms/`) | # Species |
|----------|--------------------------------|-----------|
| `ethane`   | `Ethane_Kinetic-Model_species_35.yaml` | 35 |
| `propane`  | `Propane_Kinetic-Model_species_53.yaml` | 53 |
| `n-hexane` | `n-Hexane_Kinetic-Model_species_153.yaml` | 153 |
| `naphtha`  | `Naphtha_Kinetic-Model_species_1951.yaml` | 1951 |

Filenames are referenced by `configs/reactant_database.json`. To add a new
reactant, drop the YAML file into `mechanisms/` and add a new entry in the
reactant database with `"mechanism_file": "<your_file>.yaml"`.

A populated `data/training/*.pkl` dataset is **not** required to clone & run —
generate one locally with `Main_2_generate_training_data.ipynb`, with `scripts/run_main2_local_parallel.py` for multi-process runs on one machine, or with `scripts/run_main2_slurm_chunk.py` on HPC.

---

## Configuration

### `configs/ml_data_generation_config.json`

| Key | Description | Example |
|-----|-------------|---------|
| `reactants` | List of feedstocks to simulate | `["n-hexane", "ethane"]` |
| `sampling_method` | `"latin"`, `"random"`, `"structured_grid"` | `"latin"` |
| `max_combinations_per_reactant` | Simulations per reactant (random/LHS) | `500` |
| `parameter_ranges` | Grid bounds `[min, max, n_points]` | `[800, 1200, 10]` |
| `n_jobs` | CPU cores (`-1` = all) | `-1` |
| `save_interval` | Checkpoint every N sims | `10` |

### `configs/ml_training_config.json`

| Key | Description |
|-----|-------------|
| `test_size` | Train/test split fraction (default `0.2`) |
| `random_state` | Reproducibility seed (default `42`) |
| `random_forest` | `n_estimators`, `max_depth` |
| `xgboost` | `n_estimators`, `max_depth` |
| `gradient_boosting` | `n_estimators`, `max_depth` |
| `adaboost` | `n_estimators`, `learning_rate`, `max_depth` |

### `Main_4` tuning flags

```python
IF_HYPERPARAM_TUNING        = True   # enable search
IF_HYPERPARAM_TUNING_DETAIL = True   # run RandomizedSearchCV
TUNING_METHOD               = "Random"
TUNING_PRESET               = "quick"   # quick: N_iter=20, CV=3  |  full: N_iter=100, CV=5
TUNING_MAX_SAMPLES          = 50000     # subsample for search; best params refit on full data
```

---

## HPC / SLURM

For large parameter sweeps (1M+ simulations), the pipeline supports parallel execution on multi-node HPC clusters **with SLURM (Linux)**. On a single Windows or macOS machine, use `python scripts/run_main2_local_parallel.py` instead (see **Windows** in Quick Start above).

```bash
# Submit from project root (56 CPUs on 1 node)
sbatch scripts/run_training_mul_CPUs.sh

# Scale to 100s of CPUs — edit the script:
#SBATCH --nodes=4
#SBATCH --ntasks=200
```

Each SLURM task runs `scripts/run_main2_slurm_chunk.py`, which reads `TASK_ID` and `NTASKS` from the environment and processes `1/NTASKS` of all simulations. Output is written to `data/training/task_<ID>/` with no write conflicts between tasks.

```bash
# Manual parallel execution (e.g. 4 processes locally)
TASK_ID=0 NTASKS=4 python scripts/run_main2_slurm_chunk.py &
TASK_ID=1 NTASKS=4 python scripts/run_main2_slurm_chunk.py &
TASK_ID=2 NTASKS=4 python scripts/run_main2_slurm_chunk.py &
TASK_ID=3 NTASKS=4 python scripts/run_main2_slurm_chunk.py &
wait
```

---

## Sampling Strategies

| Method | When to use | Pros |
|--------|------------|------|
| `latin` (LHS) | Best default — efficient space coverage | Uniform marginals, low correlation |
| `structured_grid` | Exhaustive sweep (small n_points) | Fully deterministic, reproducible |
| `random` | Quick prototyping | Fast to configure |

With 10 points per parameter across 6 parameters, a structured grid yields 10⁶ = **1,000,000 combinations**. Use LHS with `max_combinations_per_reactant = 500–5000` for the same coverage with far fewer runs.

---

## Results

After training on 152,442 n-hexane simulation points (exit-condition mode), the surrogate models predict all thermodynamic state variables and species concentrations with the following typical test-set performance:

| Metric | Random Forest | XGBoost | Gradient Boosting |
|--------|--------------|---------|-------------------|
| R² (mean over outputs) | ~0.97–0.99 | ~0.97–0.99 | ~0.97–0.99 |
| Speed vs. Cantera | **~500×** | **~500×** | **~200×** |

*Exact values vary by reactant, sampling density, and target variable.*

---

## Roadmap

- [x] PFR simulation (Cantera, multi-reactant)
- [x] Training data generation (LHS, grid, parallel / SLURM)
- [x] Feature engineering pipeline
- [x] Tree-based surrogate models (RF, GB, XGBoost, AdaBoost)
- [x] Hyperparameter tuning with safe subsample + full refit
- [x] Model evaluation and comparison notebook
- [ ] Physics-informed neural networks (PINNs)
- [ ] Ensemble / stacked models
- [ ] Reactor design optimisation (bayesian / gradient-free)
- [ ] Interactive GUI
- [ ] Additional reactants (butane, pentane, …)

---

## License

[MIT](LICENSE) © Nikolas Karefyllidis
