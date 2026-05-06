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
│   └── ml/                       # ML Surrogate Models
│       ├── __init__.py
│       ├── data_generation.py    # Training data generation
│       ├── model_training.py     # ML model training
│       ├── inference.py           # ML inference
│       └── example_usage.py      # ML usage examples
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
│   └── .gitkeep                   # Filenames listed in README “Required External Files”
│
├── data/                         # Data directory
│   ├── training/                 # Training data (generated)
│   └── raw/                      # Raw simulation data
│
├── models/                       # Trained ML models (generated)
│   ├── random_forest_primary.joblib
│   ├── gradient_boosting_primary.joblib
│   ├── xgboost_primary.joblib
│   ├── adaboost_primary.joblib
│   └── (optional: neural_network_*.h5, training_summary.json from model_training.py)
│
├── outputs/                      # Simulation outputs
│   ├── results/                  # CSV results and summaries
│   └── figures/                  # Generated plots
│
├── docs/                         # Documentation
│   ├── API_REFERENCE.md
│   ├── ML_CONFIG_GUIDE.md       # ML configuration guide
│   ├── UPDATES_v3.0.md          # Version 3.0 update notes
│   └── ml/                        # ML Surrogate Models documentation
│       ├── README.md
│       ├── QUICKSTART.md
│       └── IMPLEMENTATION_SUMMARY.md
│
├── examples/                     # Usage examples
│   └── basic_usage.py
│
├── scripts/                      # Organized by use case (paths from repo root)
│   ├── cluster/
│   │   ├── run_main2_slurm_chunk.py      # Main_2 chunk worker (TASK_ID, NTASKS; optional HYDRAI_ML_CONFIG)
│   │   ├── run_training_mul_CPUs.sh      # Multi-node CPU SLURM example
│   │   ├── run_training_smoke_gpu_partition.sh  # Short smoke job (tiny config; edit #SBATCH for site)
│   │   ├── run_trainning_mul_CPUs.sh     # Legacy typo alias (for compatibility)
│   │   ├── run_training_mul_GPUs.sh      # Canonical GPU smoke alias
│   │   └── run_trainning_mul_GPUs.sh     # Legacy typo alias (for compatibility)
│   ├── local/
│   │   ├── run_main2_local_parallel.py   # Multi-process Main_2 on one machine
│   │   └── run_main1_local_simulation.sh # Launches Main_1 notebook (bash)
│   ├── notebook/
│   │   ├── run_simulation.sh             # Launches Main_1 notebook (bash)
│   │   └── run_simulation_ipynb.sh
│   └── dev/
│       ├── check_complete_runs.py        # Training sweep summary / manifests
│       ├── consolidate_training_data.py  # Merge per-task outputs for ML pipeline
│       ├── monitor_run.sh                # Live cluster run status
│       ├── clean_completed_runs.py       # Archive completed task artifacts
│       └── show_structure.sh             # Requires `tree`
│
├── styles/                       # Figure aesthetics docs + examples (JSON in configs/style/)
│   ├── README.md                 # Points to configs/style/figure_aesthetics.json
│   └── example_usage.py          # Optional plot_style demos
│
├── temp/                         # Temporary files (auto-generated, git-ignored)
│   └── .gitkeep                  # Preserves directory structure
│
├── notebooks/
│   ├── Main_1_run_pfr.ipynb                       # Step 1: PFR simulations
│   ├── Main_2_generate_training_data.ipynb        # Step 2: ML training data generation
│   ├── Main_3_data_exploration_feature_engineering.ipynb
│   ├── Main_4_train_tree_models.ipynb             # Step 4: Tree-based ML training
│   └── Main_4b_tree_models_comparison.ipynb       # Step 5: Model comparison metrics & plots
├── requirements.txt
├── README.md
├── LICENSE
└── CHANGELOG.md
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
- **Naming**: Notebooks use **`Main_N_`** prefix for pipeline order through **`Main_4b`** (comparison after training).

### 4b. Scripts & SLURM monitoring

- **Cluster:** submit `scripts/cluster/*.sh` from the repo root; each task runs `run_main2_slurm_chunk.py`. Override the JSON config with `export HYDRAI_ML_CONFIG=...` (absolute path or relative to repo root).
- **Cluster tuning:** current `scripts/cluster/*.sh` defaults are tuned for the University of Cambridge **CSD3** environment. On other SLURM systems, update account/partition/QoS/module settings in `#SBATCH` and `module load` lines.
- **Progress files:** during chunk runs, each task updates `logs/data_generation_progress_task_<TASK_ID>.json` after every completed simulation. Per-run CSV logs: `temp/conditions_run_task_<TASK_ID>.csv`; completion lines: `temp/completed_runs_task_<TASK_ID>.txt`.
- **Diagnostics:** `python scripts/dev/check_complete_runs.py` aggregates sweep status from config + `data/training/`. `bash scripts/dev/monitor_run.sh` shows live status (run from repo root).
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

**2. Train tree-based ML models (Jupyter Notebook):**
```bash
jupyter notebook notebooks/Main_4_train_tree_models.ipynb
```
Trains Random Forest, Gradient Boosting, XGBoost, and AdaBoost (one model per primary target). Saves artifacts to `models/` (e.g. `random_forest_primary.joblib`).

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
