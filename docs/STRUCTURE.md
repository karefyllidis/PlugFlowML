# HydrAI Project Structure

## New Directory Organization

The project has been restructured for better organization and scalability:

```
HydrAI/
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
│   │   ├── config_template.json
│   │   ├── reactant_database.json
│   │   └── heat_flux_profile.json
│   ├── style/
│   │   └── figure_aesthetics.json # Matplotlib styling (colors, fonts, save DPI)
│   └── ml/
│       ├── ml_data_generation_config.json
│       ├── ml_data_generation_config.smoke.json
│       ├── ml_training_config.json
│       └── ml_inference_config.json
│
├── mechanisms/                    # Chemical kinetic mechanisms (YAMLs git-ignored; add locally)
│   └── .gitkeep                   # Filenames listed in README "Required External Files"
│
├── data/                         # Data directory
│   ├── training/                 # Training data (generated)
│   └── processed/                # Feature-engineered data (generated)
│
├── models/                       # Trained ML models (generated, git-ignored)
│   ├── tree_models_exit.joblib            # Main_4 baseline bundle (overwritten each run)
│   ├── tree_model_tuned_exit_full.joblib  # Main_5 tuned exit + optional full-profile bundle
│   ├── simple_nn_exit_state_dict.pt       # Main_6 PyTorch state_dict (exit-plane)
│   ├── simple_nn_exit_scalers.joblib      # Main_6 X/y scalers + label encoder
│   ├── simple_nn_exit_manifest.json       # Main_6 manifest (h1–h3, training, grouped metrics, chemistry_groups, metrics_by_group, tuning)
│   ├── simple_nn_exit_per_target_metrics.csv   # Main_6 per-target test metrics (CSV)
│   ├── simple_nn_exit_group_metrics.csv        # Main_6 uniform-average metrics by state vs chemistry group (CSV)
│   ├── simple_nn_full_profile_state_dict.pt   # Main_7 full-profile PyTorch state_dict
│   ├── simple_nn_full_profile_scalers.joblib  # Main_7 X/y scalers + label encoder
│   ├── simple_nn_full_profile_manifest.json   # Main_7 manifest (includes feature_cols, run_level_split, groups)
│   ├── simple_nn_full_profile_per_target_metrics.csv
│   └── simple_nn_full_profile_group_metrics.csv
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
│   │   ├── run_main2_slurm_chunk.py      # Main_2 chunk worker (TASK_ID, NTASKS; optional HYDRAI_ML_CONFIG)
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
│   ├── Main_6__train_evaluate_SimpleNN_IO.ipynb          # Step 6: PyTorch MLP (3 hidden layers) + optional Optuna; §8 LR plateau / early stop / best ckpt; 3-col parity+residuals (all targets); exit exports
│   └── Main_7_train_evaluate_SimpleNN_full_profile.ipynb # Step 7: full axial rows + relative_position; run-level split; §9b axial (state+species, fixed/random runs); 4-col parity+shared hexbin colorbar; full_profile exports
│
├── assets/                       # Static assets (images for README etc.)
├── tests/                        # Test suite
├── run_pipeline.py               # Run all notebooks in order
├── run_pipeline.bat              # Windows batch wrapper
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
- **Naming**: Notebooks use **`Main_N_`** prefix for pipeline order through **`Main_7`** (exit-plane PyTorch in Main_6; full-profile PyTorch in Main_7).

### 4b. Scripts & SLURM monitoring

- **Cluster:** submit `scripts/cluster/*.sh` from the repo root; each task runs `run_main2_slurm_chunk.py`. Override the JSON config with `export HYDRAI_ML_CONFIG=...` (absolute path or relative to repo root).
- **Cluster tuning:** current `scripts/cluster/*.sh` defaults are tuned for the University of Cambridge **CSD3** environment. On other SLURM systems, update account/partition/QoS/module settings in `#SBATCH` and `module load` lines.
- **Progress files:** during chunk runs, each task updates `logs/data_generation_progress_task_<TASK_ID>.json` after every completed simulation. Per-run CSV logs: `temp/conditions_run_task_<TASK_ID>.csv`; completion lines: `temp/completed_runs_task_<TASK_ID>.txt`.
- **Diagnostics:** `python scripts/dev/check_complete_runs.py` aggregates sweep status from config + `data/training/`. `bash scripts/monitor/monitor_cluster_jobs.sh` shows live status (run from repo root).
- **NN training progress:** edit flags in `scripts/monitor/monitor_nn_training_progress.py` (`MAIN_6` or `MAIN_7`, `OPTUNA` for §6b vs §8, `FOLLOW`), then `python scripts/monitor/monitor_nn_training_progress.py` from the repo root while Main_6 / Main_7 runs.
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
- **Generated / large**: `data/training/`, `data/processed/`, `data/figures/`, `outputs/results/`, `outputs/figures/`, `models/`, `logs/`, `temp/`, common ML binaries (`*.pkl`, `*.joblib`, `*.pt`, `*.pth`, …), run metadata (`metadata_*.json`), and training CSVs matching `training_data_*.csv`.
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
jupyter notebook notebooks/Main_6__train_evaluate_SimpleNN_IO.ipynb
jupyter notebook notebooks/Main_7_train_evaluate_SimpleNN_full_profile.ipynb
```
- Main_4 trains baseline trees (RF, Gradient Boosting, XGBoost, AdaBoost) and saves them to `models/tree_models_exit.joblib` (overwritten each run).
- Main_5 tunes one tree model and, when enabled, also fits the full-profile model; both are bundled into `models/tree_model_tuned_exit_full.joblib`.
- Main_6 trains a PyTorch `SimpleNN` (optional Optuna §6b on a validation fold; test held out), applies **ReduceLROnPlateau** / **early stopping** / **best test-R² checkpoint restore** in Section 8, and writes `models/simple_nn_exit_state_dict.pt`, `_scalers.joblib`, `_manifest.json`, plus **`simple_nn_exit_per_target_metrics.csv`** and **`simple_nn_exit_group_metrics.csv`** when `IF_MODEL_EXPORT` (overwritten each run). Parity and residual figure grids use **three columns** and cover **all state + species** targets. §8 appends a **training progress CSV** under `outputs/reports/`; §6b updates `optuna_tuning_plot_data.json` incrementally. External live plots: `scripts/monitor/monitor_nn_training_progress.py` (`OPTUNA=True` during §6b, `OPTUNA=False` during §8).
- Main_7 trains the same `SimpleNN` on **all axial rows** with **`relative_position`**, **run-level** train/test split (§4), Optuna §6b on **validation rows from train data only**, and the same §8 controls on **held-out test runs**. Overfitting during tuning = validation R² across trials; during production training = **train vs test R²** (and gap) at §8 checkpoints. Optional **`FULL_PROFILE_MAX_ROWS`** for smoke runs; exports **`models/simple_nn_full_profile_*`** + CSVs. §9b: **`full_profile_cantera_vs_nn_axial_evolution.png`**. §10: **four columns**, shared hexbin colorbar or scatter (`PARITY_HEXBIN_MIN_POINTS`). Monitor: same script with `MAIN_7=True`; see `docs/ML_CONFIG_GUIDE.md`.
- Each notebook also tees its terminal output to `outputs/reports/<NotebookName>.txt` via `src.utils.run_log.start_run_log` (stable path, **overwritten on every run**).

**Alternative (all model types including neural network):**
```bash
python src/ml/model_training.py configs/ml/ml_training_config.json
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
