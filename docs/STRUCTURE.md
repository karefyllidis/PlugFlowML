# PlugFlowML Project Structure

## New Directory Organization

The project has been restructured for better organization and scalability:

```
PlugFlowML/
├── src/                          # Source code
│   ├── __init__.py
│   ├── cantera/                  # Cantera-based simulation
│   │   ├── __init__.py
│   │   └── pfr_simulator.py      # Main PFR simulation code
│   ├── ml/                       # ML Surrogate Models
│   │   ├── __init__.py
│   │   ├── data_generation.py    # Training data generation
│   │   ├── dataframe_pickle.py   # Portable pickle I/O (StringDtype coercion)
│   │   ├── model_training.py     # ML model training
│   │   ├── inference.py           # ML inference
│   │   └── example_usage.py      # ML usage examples
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       ├── plot_style.py          # Matplotlib style (setup_matplotlib + JSON-driven helpers)
│       ├── plot_parallel.py       # Parallel coordinates (Inselberg) & parallel sets (Kosara) for multidim EDA
│       └── run_log.py             # Notebook run-log capture (start_run_log / stop_run_log)
│
├── configs/                      # Configuration files
│   ├── simulation/               # PFR templates, reactants, heat flux
│   │   ├── main1_pfr_run_config_template.json
│   │   ├── main1_reactant_database.json
│   │   └── main1_heat_flux_profile.json
│   ├── style/
│   │   └── figure_aesthetics.json # Matplotlib styling (colors, fonts, save DPI)
│   └── ml/
│       ├── main1_run_pfr_config.json
│       ├── main2_data_generation_config.json
│       ├── main2_data_generation_config.smoke.json
│       ├── main3_eda_feature_engineering_config.json
│       ├── main4_tree_baseline_config.json
│       ├── main5_tree_tuning_config.json
│       ├── main6_simplenn_config.json
│       ├── main7_pinn_config.json
│       ├── main8_symbolic_regression_config.json
│       ├── main9_compare_cantera_pinn_sr_config.json
│       ├── main10_bayesian_optimisation_config.json
│       ├── model_training_script_config.json
│       └── ml_inference_config.json
│
├── mechanisms/                    # Chemical kinetic mechanisms (YAMLs git-ignored; add locally)
│   └── .gitkeep                   # Filenames listed in README "Required External Files"
│
├── data/                         # Data directory
│   ├── training/                 # Training data (generated)
│   ├── processed/                # Feature-engineered data (generated)
│   └── logs/                     # Main_6/7 training progress CSV + Optuna JSON (monitor)
│
├── models/                       # Trained ML models (generated, git-ignored except .gitkeep per subfolder)
│   ├── tree_baseline/                     # Main_4
│   │   └── tree_models_exit.joblib        # Baseline bundle (overwritten each run)
│   ├── tree_tuned/                        # Main_5
│   │   └── tree_model_tuned_exit_full.joblib  # Tuned exit + optional full-profile bundle
│   ├── simple_nn_full_profile/            # Main_6
│   │   ├── simple_nn_full_profile_state_dict.pt   # PyTorch state_dict
│   │   ├── simple_nn_full_profile_scalers.joblib  # X/y scalers + label encoder
│   │   ├── simple_nn_full_profile_manifest.json   # Manifest (feature_cols, run_level_split, groups)
│   │   ├── simple_nn_full_profile_per_target_metrics.csv
│   │   └── simple_nn_full_profile_group_metrics.csv
│   ├── pinn_pfr/                          # Main_7
│   │   ├── pinn_pfr_state_dict.pt         # PINNPFR state_dict
│   │   ├── pinn_pfr_scalers.joblib        # X/y scalers
│   │   └── pinn_pfr_manifest.json         # Manifest (architecture, loss weights, training)
│   ├── sr_full_profile/                   # Main_8, teacher=simple_nn_full_profile
│   │   ├── sr_full_profile_equations.py
│   │   ├── sr_full_profile_manifest.json
│   │   └── sr_full_profile_metrics.csv
│   └── sr_pinn/                           # Main_8, teacher=pinn_pfr
│       ├── sr_pinn_equations.py
│       ├── sr_pinn_manifest.json
│       └── sr_pinn_metrics.csv
│
├── outputs/                      # Simulation outputs
│   ├── results/                  # CSV results and summaries
│   ├── figures/                  # Generated plots (per-notebook subdirs)
│   └── reports/                  # Curated .md summaries + auto-captured <NotebookName>.txt
│                                  # run logs (overwritten each notebook execution)
│
├── docs/                         # Documentation
│   ├── API_REFERENCE.md
│   ├── CHANGELOG.md
│   ├── DIRECTORY_STRUCTURE.md
│   ├── HF_MODEL_CARD_TEMPLATE.md
│   ├── HPC_GUIDE.md
│   ├── ML_CONFIG_GUIDE.md       # ML configuration guide
│   ├── MODEL_CARD.md
│   ├── SPECIES_LUMPING_MODEL_CARD.md
│   ├── STRUCTURE.md              # This file
│   ├── TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md
│   ├── UPDATES_v3.0.md          # Version 3.0 update notes
│   └── ml/                        # ML Surrogate Models documentation
│       ├── README.md
│       ├── QUICKSTART.md
│       └── IMPLEMENTATION_SUMMARY.md
│
├── scripts/                      # Organized by use case (paths from repo root)
│   ├── cluster/
│   │   ├── run_main2_slurm_chunk.py      # Main_2 chunk worker (TASK_ID, NTASKS; optional PLUGFLOWML_ML_CONFIG)
│   │   ├── run_training_mul_CPUs.sh      # Multi-node CPU SLURM example
│   │   ├── run_training_mul_GPUs.sh      # Canonical GPU smoke alias
│   │   └── run_training_smoke_gpu_partition.sh  # Short smoke job (tiny config; edit #SBATCH for site)
│   ├── local/
│   │   ├── run_main2_local_parallel.py   # Multi-process Main_2 on one machine
│   │   └── run_main1_local_simulation.sh # Launches Main_1 notebook (bash)
│   ├── monitor/
│   │   ├── monitor_cluster_jobs.sh             # Live SLURM data-generation status
│   │   └── monitor_nn_training_progress.py   # Main_6 / Main_7 training & Optuna plots
│   └── dev/
│       ├── check_complete_runs.py        # Training sweep summary / manifests
│       ├── clean_completed_runs.py       # Archive completed task artifacts
│       ├── consolidate_training_data.py  # Merge per-task outputs for ML pipeline
│       ├── smoke_main7_cpu_scaling.py    # Main_6 CPU / Optuna thread smoke test
│       └── sbatch_safe.sh                # CRLF-safe sbatch wrapper
│
├── temp/                         # Temporary files (auto-generated, git-ignored)
│   └── .gitkeep                  # Preserves directory structure
│
├── notebooks/
│   ├── Main_1_run_pfr.ipynb                       # Step 1: PFR simulations
│   ├── Main_2_generate_training_data.ipynb        # Step 2: ML training data generation
│   ├── Main_3_data_exploration_feature_engineering.ipynb  # Step 3: EDA + feature engineering
│   ├── Main_4_train_and_evaluate_tree_models_IO.ipynb    # Step 4: Baseline tree evaluation (exit-plane)
│   ├── Main_5_train_evaluate_tune_tree_model_evolution.ipynb  # Step 5: One-model tuning + full PFR evolution
│   ├── Main_6_train_evaluate_SimpleNN_full_profile.ipynb # Step 6: full axial rows + relative_position; run-level split; §9b axial (state+species, fixed/random runs); 4-col parity+shared hexbin colorbar; full_profile exports
│   ├── Main_7_train_evaluate_PINN_full_profile.ipynb    # Step 7: PINN with PFR ODE residuals; `PINNPFR` (`src/models/pinn.py`) + curriculum warmup; algebraic (EOS/mass/species) + energy ODE (autograd) physics loss; collocation points; pinn_pfr_* exports
│   ├── Main_8_symbolic_regression_SR.ipynb              # Step 8: PySR distillation of any NN teacher (Main_6/7) → closed-form equations; exports sr_<teacher>_equations.py
│   ├── Main_9_compare_cantera_pinn_sr.ipynb             # Step 9: Cantera vs PINN vs SR comparison/validation
│   └── Main_10_optimisation_BO_surrogate_vs_cantera.ipynb # Step 10: Optuna GPSampler BO on MLP + SR surrogates; Cantera validation of both optima
│
├── assets/                       # Static assets (images for README etc.)
├── tests/                        # Test suite
├── run_pipeline.py               # Run Main_4-10 in order (run Main_1-3 manually first for data)
├── requirements.txt
├── README.md
└── LICENSE
```

## Key Changes

### 1. Source Code Organization
- **Before**: `Main_GeneralizedPFR.py` at root
- **After**: `src/cantera/pfr_simulator.py` in organized package structure

### 2. Configuration Files
- **Before**: Config files at root (`config_template.json`, `reactant_database.json`)
- **After**: Configs live under `configs/simulation/`, `configs/ml/`, and `configs/style/` (see tree above). `plot_style` prefers `configs/style/figure_aesthetics.json`; legacy flat `configs/figure_aesthetics.json` or `styles/figure_aesthetics.json` still work if present.

### 3. Mechanisms
- **Before**: `mechanisms/` directory
- **After**: `mechanisms/` directory (plural, more standard)

### 4. Notebooks
- **Location**: All interactive entry points are in **`notebooks/`**
- **Naming**: Notebooks use **`Main_N_`** prefix for pipeline order, **`Main_1`** through **`Main_10`**.

### 4b. Scripts & SLURM monitoring

- **Cluster:** submit `scripts/cluster/*.sh` from the repo root; each task runs `run_main2_slurm_chunk.py`. Override the JSON config with `export PLUGFLOWML_ML_CONFIG=...` (absolute path or relative to repo root).
- **Cluster tuning:** current `scripts/cluster/*.sh` defaults are tuned for the University of Cambridge **CSD3** environment. On other SLURM systems, update account/partition/QoS/module settings in `#SBATCH` and `module load` lines.
- **Progress files:** during chunk runs, each task updates `logs/data_generation_progress_task_<TASK_ID>.json` after every completed simulation. Per-run CSV logs: `temp/conditions_run_task_<TASK_ID>.csv`; completion lines: `temp/completed_runs_task_<TASK_ID>.txt`.
- **Diagnostics:** `python scripts/dev/check_complete_runs.py` aggregates sweep status from config + `data/training/`. `bash scripts/monitor/monitor_cluster_jobs.sh` shows live status (run from repo root).
- **NN training progress:** `python scripts/monitor/monitor_nn_training_progress.py` while Main_6 / Main_7 runs (`MAIN_6` or `MAIN_7`, optional `LIVE=True`; auto-picks newest log in `data/logs/`; see [`data/logs/README.md`](../data/logs/README.md)).
- **Main_3 smoke data:** pin `RUN_STAMP_DEVEL` (`20260507_DEVEL`) in Main_3 for small `data/training/` files; export full `features_targets_*_095243.pkl` for production ML notebooks.
- **Data consolidation:** After a parallel run, merge per-task outputs for the ML notebook:
  `python scripts/dev/consolidate_training_data.py`
  This creates `data/training/training_data_complete_<timestamp>.pkl` which `Main_3_data_exploration_feature_engineering.ipynb` auto-detects.
  By default, old per-task files are cleaned after successful merge; use `--no-cleanup` to keep them, or `--dry-run` for preview only.
- **Linux line endings:** if CSD3 reports `/bin/bash^M` or DOS line breaks, run:
  `find scripts -type f -name "*.sh" -exec sed -i 's/\r$//' {} \;`

### 5. ML Surrogate Models
- **Before**: `phase_b/` directory with mixed files
- **After**: `src/ml/` with organized modules and `docs/ml/` for documentation

### 6. Outputs
- **Before**: `results/` and `fig/` at root
- **After**: `outputs/results/` and `outputs/figures/` organized under outputs

### 7. Data and Models
- **New**: `data/` directory for training data
- **New**: `models/` directory for trained ML models

### 8. Version control (`.gitignore`)
- **Generated / large**: `data/training/`, `data/processed/`, `data/logs/` (NN progress CSV + Optuna JSON), `data/figures/`, `outputs/results/`, `outputs/figures/`, `models/`, `logs/` (SLURM), `temp/`, common ML binaries (`*.pkl`, `*.joblib`, `*.pt`, `*.pth`, …), run metadata (`metadata_*.json`), and training CSVs matching `training_data_*.csv`.
- **Mechanisms**: `mechanisms/*.yaml` are excluded by default so clones stay small; only `mechanisms/.gitkeep` is tracked. Add YAMLs locally per `README.md`.
- **Local-only**: `.cursor/`, `.vscode/`, `.env`, `.env.local` (see root `.gitignore`).
- Summary table for contributors: **Version control** section in `README.md`.

## Usage

### Running Simulations

**Interactive Jupyter Notebook (Recommended):**
```bash
jupyter notebook notebooks/Main_1_run_pfr.ipynb
# Or: jupyter lab notebooks/Main_1_run_pfr.ipynb
```

The notebook provides an interactive interface where you can:
- Select reactants interactively
- See real-time simulation progress
- View inline visualizations
- Modify parameters easily

### ML Surrogate Models Workflow

**1. Generate training data (Jupyter Notebook):**
```bash
jupyter notebook notebooks/Main_2_generate_training_data.ipynb
```

The notebook provides:
- Interactive configuration
- Real-time progress tracking
- Comprehensive data visualization
- Data quality checks

**2. Train surrogate models (Jupyter Notebooks):**
```bash
jupyter notebook notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb
jupyter notebook notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb
jupyter notebook notebooks/Main_6_train_evaluate_SimpleNN_full_profile.ipynb
jupyter notebook notebooks/Main_7_train_evaluate_PINN_full_profile.ipynb
```
- Main_4 trains baseline trees (RF, Gradient Boosting, XGBoost, AdaBoost) and saves them to `models/tree_baseline/tree_models_exit.joblib` (overwritten each run); optional `IF_HYPERPARAM_TUNING` (§7) runs BayesSearchCV per model before the export.
- Main_5 tunes one tree model and, when enabled, also fits the full-profile model; both are bundled into `models/tree_tuned/tree_model_tuned_exit_full.joblib`.
- Main_6 trains the `SimpleNN` on **all axial rows** with **`relative_position`**, **run-level** train/test split (§4), optional Optuna §6b on **validation rows from train data only**, and §8 **ReduceLROnPlateau** / **early stopping** / **best test-R² checkpoint restore**. Overfitting during tuning = validation R² across trials; during production training = **train vs test R²** (and gap) at §8 checkpoints. Optional **`FULL_PROFILE_MAX_ROWS`** for smoke runs; exports **`models/simple_nn_full_profile/simple_nn_full_profile_*`** + CSVs. §9b: **`full_profile_cantera_vs_nn_axial_evolution.png`**. §10: **four columns**, shared hexbin colorbar or scatter (`PARITY_HEXBIN_MIN_POINTS`). Monitor: `scripts/monitor/monitor_nn_training_progress.py` (`MAIN_6=True`); see `docs/ML_CONFIG_GUIDE.md`.
- Main_7 trains `PINNPFR` with the composite data + physics loss; exports `models/pinn_pfr/pinn_pfr_*`. Monitor: same script with `MAIN_7=True`.
- Each notebook also tees its terminal output to `outputs/reports/<NotebookName>.txt` via `src.utils.run_log.start_run_log` (stable path, **overwritten on every run**).

**Alternative (all model types including neural network):**
```bash
python src/ml/model_training.py configs/ml/model_training_script_config.json
```

**Note:** All workflows use Jupyter notebooks for interactive use. Command-line scripts are available in `src/ml/` for batch processing.

## Migration Notes

### Import Changes

**Old:**
```python
from Main_GeneralizedPFR import load_reactant_database
```

**New:**
```python
from src.cantera.pfr_simulator import load_reactant_database
```

### ML Surrogate Models

**Import ML modules:**
```python
from src.ml.data_generation import TrainingDataGenerator
from src.ml.model_training import MLModelTrainer
from src.ml.inference import MLPFRPredictor
```

**Use figure aesthetics:**
```python
from src.utils.plot_style import plot_profile, load_aesthetics

# Create plot with aesthetics
fig, ax = plot_profile(z, temperature, 'temperature', output_path='outputs/figures/temp.png')
```

### Path Changes

All file paths are now relative to project root:
- Configs: `configs/simulation/`, `configs/ml/`, `configs/style/` (see project tree)
- Mechanisms: `mechanisms/`
- Outputs: `outputs/results/` and `outputs/figures/`
- Training data: `data/training/`
- Models: `models/`
- SLURM progress (generated): `logs/data_generation_progress_task_*.json`
- Run logs / temp CSV (generated): `temp/conditions_run_task_*.csv`, `temp/completed_runs_task_*.txt`

## Benefits

1. **Better Organization**: Clear separation of concerns
2. **Scalability**: Easy to add new modules and features
3. **Maintainability**: Standard Python package structure
4. **Clarity**: Self-documenting directory structure
5. **Professional**: Follows Python best practices
6. **ML Integration**: ML Surrogate Models seamlessly integrated
7. **Consistent Styling**: Centralized figure aesthetics
8. **Reproducibility**: JSON-based configuration for all workflows

## File Path Resolution

The code uses helper functions to resolve paths relative to project root:
- `get_project_root()` - Returns project root directory
- `get_config_path(filename)` - Returns path to config file
- `get_output_path(subdir, filename)` - Returns path to output file

This ensures paths work regardless of where the script is run from.
