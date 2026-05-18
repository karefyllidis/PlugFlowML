# ML Configuration Guide

All ML scripts now use JSON configuration files instead of command-line arguments.

## Configuration Files

### 1. Training Data Generation

**File**: `configs/ml/ml_data_generation_config.json`

```json
{
    "_comment": "Configuration for ML training data generation. See docs/ML_CONFIG_GUIDE.md for detailed documentation.",
    
    "reactants": ["ethane", "propane"],
    "max_combinations_per_reactant": 100,
    "output_dir": "data/training",
    "save_interval": 10,
    "sampling_method": "latin",
    "lhs_seed": 42,
    "n_jobs": -1,
    
    "parameter_ranges": {
        "temperature_K": [800, 1200, 10],
        "pressure_bar": [1.5, 3.0, 8],
        "length_m": [3.0, 7.0, 6],
        "diameter_mm": [20.0, 40.0, 5],
        "mass_flow_rate_kgps": [0.05, 0.10, 6],
        "heat_flux_Wm2": [100000, 200000, 5]
    }
}
```

**Usage:**
```bash
python src/ml/data_generation.py configs/ml/ml_data_generation_config.json
```

**Optional — SLURM chunk runner** (`scripts/cluster/run_main2_slurm_chunk.py`): set `HYDRAI_ML_CONFIG` to a JSON path (absolute or relative to repo root) to override the default `configs/ml/ml_data_generation_config.json`. For a minimal test workload use `configs/ml/ml_data_generation_config.smoke.json`. Each task writes live status to `logs/data_generation_progress_task_<TASK_ID>.json`. See `README.md` (HPC / SLURM).

**Post-run consolidation (parallel runs):**

After SLURM/local parallel generation that writes into `data/training/task_*`, consolidate to one dataset for `Main_3`:

```bash
python scripts/dev/consolidate_training_data.py
```

- Default behavior: writes consolidated `training_data_complete_<timestamp>.pkl` and `metadata_<timestamp>.json`, then cleans old per-task files/folders.
- Keep per-task artifacts: `python scripts/dev/consolidate_training_data.py --no-cleanup`
- Preview without writing/deleting: `python scripts/dev/consolidate_training_data.py --dry-run`

**Parameters:**

- **`reactants`** (list of strings): List of reactants to generate data for. Available options: `"ethane"`, `"propane"`, `"naphtha"`, `"n-hexane"`. Each reactant will generate up to `max_combinations_per_reactant` simulations. If not specified or empty, uses all available reactants.

- **`max_combinations_per_reactant`** (integer): Maximum number of parameter combinations to simulate per reactant. Total simulations = `reactants.length × max_combinations_per_reactant`. Higher values = more training data but longer generation time. Recommended: 50-200 for initial testing, 500-1000+ for production datasets.

- **`output_dir`** (string): Directory where training data files will be saved. Final dataset is saved as both pickle (`.pkl`) and CSV (`.csv`) formats with timestamps: `training_data_complete_YYYYMMDD_HHMMSS.pkl` and `training_data_complete_YYYYMMDD_HHMMSS.csv`. Partial saves during generation use pickle format: `training_data_partial_YYYYMMDD_HHMMSS.pkl`. Partial files are automatically cleaned up after successful completion.

- **`save_interval`** (integer): Save progress every N simulations. Prevents data loss if generation is interrupted. Partial saves are stored as pickle files for efficiency. Set to `1` to save after every simulation (slower but safer). Recommended: 10-50 for long runs. Partial files are automatically deleted after successful completion to save disk space.

- **`n_jobs`** (integer): Number of parallel workers for data generation. 
  - `1`: Sequential execution (default, safer for debugging)
  - `2-8`: Use specific number of CPU cores
  - `-1`: Use all available CPU cores (recommended for large datasets)
  
  **Note**: Parallel execution significantly speeds up data generation but uses more memory. Each worker runs a separate Cantera simulation, so ensure you have enough RAM. Recommended: Use `-1` for production runs, `1` for testing.

- **`sampling_method`** (string): How to sample the parameter space.
  - `"random"`: Random sample of up to `max_combinations_per_reactant` points within the bounds.
  - `"full_grid"`, `"structured_grid"`, or `"grid"`: **Structured grid** – use all combinations from `parameter_ranges` (regular spacing per dimension). Total runs = product of `n_points` over parameters (can be large).
  - `"latin"` or `"latin_hypercube"`: **Latin Hypercube Sampling (LHS)** – better space-filling with fewer runs. Uses `max_combinations_per_reactant` as the number of LHS points. Bounds from `random_sample_bounds` (or `parameter_ranges`) apply.
  
  **Recommendation**: Use `"latin"` for efficient exploration; use `"structured_grid"` or `"grid"` when you want a regular grid; use `"random"` for compatibility with older configs.

- **`lhs_seed`** (integer): Random seed for Latin Hypercube Sampling. Default: `42`. Only used when `sampling_method` is `"latin"` or `"latin_hypercube"`.
  
  **Example**: With 6 parameters each having 10 values, full grid = 10⁶ combinations. Random or LHS with `max_combinations_per_reactant=100` generates 100 combinations.

- **`random_sample_bounds`** (object, optional): Bounds for **both** random and Latin Hypercube sampling. If provided, sampling is constrained to these [min, max] per parameter.
  
  **Format**: `{"parameter_name": [min_value, max_value], ...}`
  
  **Example**:
  ```json
  "random_sample_bounds": {
      "temperature_K": [850, 1200],
      "pressure_bar": [2.0, 3.0],
      "length_m": [10.0, 15.0]
  }
  ```
  
  **Use cases**:
  - Focus on a specific region of interest (e.g., high-temperature, high-pressure conditions)
  - Exclude physically unrealistic combinations
  - Prioritize certain operating conditions
  - Reduce parameter space for faster generation
  
  **Note**: For random/LHS, if a parameter is not listed in `random_sample_bounds`, its range comes from `parameter_ranges` (min/max). For grid sampling, only `parameter_ranges` is used.

- **`parameter_ranges`** (object): Used for **grid / structured_grid / full_grid** sampling. Define the parameter space: each parameter uses format `[min_value, max_value, n_points]`. The script creates `n_points` evenly spaced values between `min` and `max` using `np.linspace(min, max, n_points)`. Total runs = product of all `n_points`.
  
  - **`temperature_K`**: Reactor inlet temperature range. Format: `[min_K, max_K, n_points]`. Typical steam cracking temperatures: 800-1200 K. Example: `[800, 1200, 10]` creates 10 points: 800, 844, 889, ..., 1200 K.
  
  - **`pressure_bar`**: Reactor pressure range. Format: `[min_bar, max_bar, n_points]`. Note: internally converted to Pa (multiply by 1e5). Example: `[1.5, 3.0, 8]` creates 8 points between 1.5-3.0 bar.
  
  - **`length_m`**: Reactor length range. Format: `[min_m, max_m, n_points]`. Longer reactors = more residence time = higher conversion. Example: `[3.0, 7.0, 6]` creates 6 points between 3-7 meters.
  
  - **`diameter_mm`**: Reactor diameter range. Format: `[min_mm, max_mm, n_points]`. Note: internally converted to meters (divide by 1000). Example: `[20.0, 40.0, 5]` creates 5 points between 20-40 mm.
  
  - **`mass_flow_rate_kgps`**: Feed mass flow rate range. Format: `[min_kgps, max_kgps, n_points]`. Higher flow = shorter residence time. Example: `[0.05, 0.10, 6]` creates 6 points between 0.05-0.10 kg/s.
  
  - **`heat_flux_Wm2`**: External heat flux range. Format: `[min_Wm2, max_Wm2, n_points]`. Higher heat flux = faster reaction rates. Example: `[100000, 200000, 5]` creates 5 points between 100-200 kW/m².

**Total Combinations Calculation:**

If `sampling_method` is `"full_grid"`, `"structured_grid"`, or `"grid"`, total combinations per reactant = product of all `n_points` in `parameter_ranges`:
- Example: `10 × 8 × 6 × 5 × 6 × 5 = 72,000` combinations per reactant
- With 2 reactants: `72,000 × 2 = 144,000` total simulations

If `sampling_method` is `"random"` or `"latin"`, only `max_combinations_per_reactant` combinations are generated per reactant; bounds come from `random_sample_bounds` (or `parameter_ranges`).

### Notebook run control (`notebooks/Main_2_generate_training_data.ipynb`)

`Main_2` is generation-focused. Training-space plotting controls were intentionally removed and moved to `Main_3`.

- **`IF_SAVE_METADATA`**: If `True`, the generator writes the metadata JSON file; if `False`, metadata is not saved (dataset still returned).
- **`IF_SAVE_TRAINING_DATA`**: If `True`, partial and final training data (pkl/csv) are written; if `False`, no training files are written (dataset is still built and returned in memory).

### Notebook run control (`notebooks/Main_1_run_pfr.ipynb`)

- **`IF_SAVE_PLOTS`**: If `True`, quick inline figures are saved to `outputs/figures/`:
  - `quick_profiles_<reactant>.png`
  - `conversion_products_<reactant>.png`

### Notebook run control (`notebooks/Main_3_data_exploration_feature_engineering.ipynb`)

- **`IF_PIN_SPECIFIC_FILES`**: If `True`, load pinned files from `data/training/` instead of the newest `training_data_complete_*.pkl`.
- **`RUN_STAMP_DEVEL` / `RUN_STAMP_FULL`**: Filename stamps — e.g. `20260507_DEVEL` (small smoke-test campaign) vs `20260507_095243` (full production set). Set `RUN_STAMP = RUN_STAMP_DEVEL` for fast pipeline checks; use `RUN_STAMP_FULL` or `IF_PIN_SPECIFIC_FILES = False` for full data. Files: `training_data_complete_<stamp>.pkl` and `metadata_<stamp>.json`.
- **`IF_SAVE_EDA_PLOTS`**: If `True`, EDA figures are saved to `outputs/figures/Main_3_data_exploration_feature_engineering/eda/`.
- **`IF_SEPARATE_SPECIES_BY_CARBON`**: If `True`, species are grouped by carbon-number lumps (`C1`, `C2`, `C3`, ... and `inert`) for dimensionality reduction.
- **`IF_CATEGORIZE_BY_CHEMISTRY`**: If `True`, species are grouped by process-role lumps (`olefins`, `aromatics`, `paraffins`, `coke_precursors`, `radicals`, `feedstock`, `hydrogen`, `diluent`, `other`).
- **`EXPORT_SPECIES_AS`**: `individual` (default) keeps every **`Y_*` mass-fraction** column in the exported `df_target` (mole fractions `X_*` are **not** ML targets). Set to `lumped_chemistry` or `lumped_carbon` to replace those `Y_*` columns with summed **`Y_lump_*`** columns for a smaller `features_targets_*.pkl`. Requires the matching flag above.
- **Saved EDA figures** include:
  - `main3_input_distributions_inlet.png`
  - `main3_primary_output_distributions_exit.png` (computed at reactor exit: max `z` / `relative_position≈1`)
  - `main3_top12_species_exit_mean_mass_fraction.png`
  - `main3_species_lumped_by_carbon_bar_exit.png`
  - `main3_species_lumped_by_chemistry_bar_exit.png`
- **Methodology model card (species lumping):** [`docs/SPECIES_LUMPING_MODEL_CARD.md`](SPECIES_LUMPING_MODEL_CARD.md) — carbon vs chemistry taxonomy, sum-of-`Y_*` aggregation, export column names, limitations.

### 2. ML Model Training

**Baseline notebook:** `notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb` — fast workflow for **inlet→outlet** (exit-plane) baseline comparison:
- Trains default RF, Gradient Boosting, XGBoost, and optionally AdaBoost
- Does **not** run hyperparameter tuning
- Evaluates held-out test metrics (R², MAE, RMSE, MAPE) and Train/Test R² gap
- Plots actual-vs-predicted state variables using consistent scatter styling
- Reports species-lump error by chemistry/carbon group using **Normalized MAE (%)**
- Reports state / thermo / aero target error using **Normalized MAE (%)** for targets such as exit temperature, pressure, velocity, density, Cp/Cv, enthalpy, and thermal conductivity
- Reports ML inference speed and optional Cantera/PFR speedup when `CANTERA_EXIT_SECONDS_PER_RUN` is set from a measured baseline
- Exports baseline models to `models/tree_models_exit.joblib` (overwritten each run)

Loads the latest `data/processed/features_targets_*.pkl`. If Main_3 used **`EXPORT_SPECIES_AS=lumped_chemistry`** (or `lumped_carbon`), targets are already **`Y_lump_*`** mass-fraction lumps; the notebook trains on those (far fewer outputs than hundreds of species).

**Tuning / evolution notebook:** `notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb`
- Tunes **one selected tree model** via `MODEL_TO_TUNE` (`random_forest`, `gradient_boosting`, `xgboost`, or `adaboost`)
- Runs `BayesSearchCV` for the exit-plane inlet→outlet model
- Optionally runs full axial/PFR evolution workflow with `TRAIN_FULL_PROFILE=True`
- Full-profile model training runs when `TRAIN_FULL_PROFILE=True` (no separate hyperparameter search): it **reuses** the exit-plane model’s `estimator__*` hyperparameters—either from `BayesSearchCV` when tuning is enabled or from the defaults used when training cell 6 without tuning.
- Full-profile uses all axial rows and includes `relative_position` as an input
- Full-profile train/test data are split by simulation run, not by row, to avoid leakage between axial points from the same PFR profile
- `FULL_PROFILE_MAX_ROWS` can be set for a quick full-profile training smoke test on large datasets
- **Inlet BC anchoring (§8):** after full-profile `predict`, `src/utils/profile_predictions.anchor_inlet_profile_predictions()` sets each test run’s prediction at **min `relative_position`** to match Cantera truth so axial overlay plots (§9) start at the same inlet state. Same helper is used in **Main_7 §9**.
- Reports tuned ML inference speed for exit-plane and full-profile prediction; set `CANTERA_EXIT_SECONDS_PER_RUN` / `CANTERA_FULL_PROFILE_SECONDS_PER_RUN` to print speedup factors against measured Cantera timings
- Exports tuned exit and full-profile artifacts to `models/tree_model_tuned_exit_full.joblib` (overwritten each run)
- Adds axial diagnostics and regime diagnostics:
  - `full_profile_cantera_vs_ml_axial_evolution.png` (Main_5 full-profile trees: Cantera vs ML along `x/L` at selected stations)
  - `full_profile_cantera_vs_nn_axial_evolution.png` (Main_7 §9b: Cantera / test vs PyTorch NN; preferred state columns plus all `species_cols` present in `target_cols`, same `x/L` grid)
  - `exit_error_vs_conditions_boxplots.png` (error vs inlet-condition bins)
  - `exit_error_tp_map.png` (temperature-pressure error map)

**Why full-profile matters:** Steam cracking is an axial-evolving reacting flow. Exit-only surrogates predict final yields; full-profile surrogates predict the entire temperature/species evolution along the reactor — essential for coil design, coking analysis, and process optimization.

**Legacy notebooks** (for advanced use):
- `Main_4_train_tree_models.ipynb` — older combined tree training workflow
- `Main_4b_tree_models_comparison.ipynb` — older detailed per-target metrics and ranking workflow

**Script (tree / boosting models from JSON):** `python src/ml/model_training.py configs/ml/ml_training_config.json`

**File**: `configs/ml/ml_training_config.json`

```json
{
    "data_file": "data/training/training_data_complete_*.csv",
    "output_dir": "models",
    "target_types": ["primary"],
    "models": ["all"],
    "test_size": 0.2,
    "random_state": 42,
    "neural_network": {
        "epochs": 200,
        "batch_size": 256,
        "learning_rate": 0.001,
        "h1": 128,
        "h2": 64,
        "h3": 32,
        "dropout": 0.1,
        "tuning": {
            "n_trials": 30,
            "epochs_per_trial": 50,
            "validation_fraction": 0.2,
            "timeout_seconds": null
        }
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 20
    },
    "xgboost": {
        "n_estimators": 150,
        "max_depth": 6
    },
    "gradient_boosting": {
        "n_estimators": 150,
        "max_depth": 5
    },
    "adaboost": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6
    }
}
```

**Usage (baseline notebook):** `jupyter notebook notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb`

**Usage (tuning/evolution notebook):** `jupyter notebook notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb`

**Usage (script for all types):**
```bash
python src/ml/model_training.py configs/ml/ml_training_config.json
```

**Parameters:**
- `data_file`: Path to training data CSV (supports glob patterns)
- `output_dir`: Directory to save trained models
- `target_types`: List of target types (`primary`, `secondary`, `species`, `all`)
- `models`: List of models to train (`neural_network` is a PyTorch placeholder in `src/ml/model_training.py` — production NN training is in `notebooks/Main_6__train_evaluate_SimpleNN_IO.ipynb` (exit-plane) and `notebooks/Main_7_train_evaluate_SimpleNN_full_profile.ipynb` (full axial profile). The CLI `all` keyword expands to RF + XGBoost + GB only.)
- `test_size`: Fraction of data for testing (0.0-1.0)
- `random_state`: Random seed for reproducibility

**`neural_network` parameters (consumed by `notebooks/Main_6__train_evaluate_SimpleNN_IO.ipynb` and `notebooks/Main_7_train_evaluate_SimpleNN_full_profile.ipynb`):**

- `epochs` (int, default 200): number of Adam epochs over the training set.
- `batch_size` (int, default 256): mini-batch size for `DataLoader(shuffle=True)`. Capped at `len(train_ds)` if larger.
- `learning_rate` (float, default 1e-3): Adam learning rate.
- `h1`, `h2`, `h3` (int, defaults 128 / 64 / 32): number of units in hidden layers 1–3 of the multi-output `SimpleNN` MLP.
- `dropout` (float, default 0.1): probability passed to `nn.Dropout` after each hidden ReLU; off automatically under `model.eval()`.

**`neural_network.tuning` parameters (consumed by Main_6 / Main_7 Section 6b only when `IF_HYPERPARAM_TUNING=True`):**

- `n_trials` (int, default 30): number of Optuna TPE trials in the study.
- `epochs_per_trial` (int, default 50): training epochs per Optuna trial; intentionally smaller than the final-model `epochs` to keep the search cheap. Median pruner stops weak trials early.
- `validation_fraction` (float, default 0.2): fraction of the **training** split carved out as the Optuna validation fold. The held-out test set is never seen during tuning.
- `timeout_seconds` (int or `null`, default `null`): wallclock cap for the entire study. `null` means no time limit; the study stops only when `n_trials` is reached.

The tuning search space (`h1 ∈ [32,256] step 32`, `h2 ∈ [16,128] step 16`, `h3 ∈ [8,64] step 8`, `dropout ∈ [0.0,0.3]`, `learning_rate ∈ [1e-4,1e-2]` log, `batch_size ∈ {64,128,256,512}`) is defined in the notebook objective function. The objective maximises validation R² (uniform average across all targets, physical units). The best trial's parameters overwrite the top-level `neural_network.{h1,h2,h3,dropout,learning_rate,batch_size}` values inside the notebook, and the final model is rebuilt before the training loop in Section 8. Requires `pip install optuna`.

**Main_6 / Main_7 production training (Section 8 — not separate JSON keys):** Both notebooks apply **`ReduceLROnPlateau`** stepped on the same periodic **test** R² checkpoints used for the convergence figure, **early stopping** if test R² fails to improve across several consecutive checkpoints, then **reload the best test-R² weights** before Section 9 metrics and Section 11 export. Main_6 saves `models/simple_nn_exit_manifest.json`; Main_7 saves `models/simple_nn_full_profile_manifest.json` with the same training keys plus **`workflow`**, **`run_level_split`**, **`feature_cols`** (includes `relative_position`), **`run_cols`**, row/run counts, and (like Main_6) **`chemistry_groups`**, **`metrics_by_group`**, **`auxiliary_exports`** pointing at per-target and group metric CSVs.

**Main_6 auxiliary exports (not separate JSON keys):** When `IF_MODEL_EXPORT=True`, Main_6 also writes `models/simple_nn_exit_per_target_metrics.csv` and `models/simple_nn_exit_group_metrics.csv`, and the manifest lists their absolute paths under **`auxiliary_exports`** together with **`chemistry_groups`** and **`metrics_by_group`** (uniform-average test R² / MAE / RMSE per chemistry role and for state/thermo).

**Main_7 auxiliary exports:** Same CSV + manifest pattern with stem **`simple_nn_full_profile_*`**. Section **§9** may write the two CSVs when `IF_MODEL_EXPORT`; **§11** overwrites them when exporting the final **`simple_nn_full_profile_manifest.json`**.

**Main_6 / Main_7 notebook-only controls (not in `ml_training_config.json`):**

- **External training progress (Section 2, Main_6 / Main_7):** `WRITE_TRAINING_PROGRESS_LOG` (default `True`) appends `data/logs/<notebook_stem>_training_progress.csv` during §8 (per-epoch train MSE; checkpoint rows include test MSE/R² and LR). Optuna §6b rewrites `data/logs/<notebook_stem>_optuna_tuning_plot_data.json` after each completed trial; the final snapshot includes fANOVA importances when available. §8b / §6b-ii notebook cells still export static PNGs; Optuna PNGs from §6b-ii JSON. **Main_7** also defines **`USE_CUDA_AMP`**, **`USE_TORCH_COMPILE`**, and **`OPTUNA_N_JOBS`** (keep **`OPTUNA_N_JOBS=1`** on a single GPU when tuning).

- **CPU parallelism (Main_7 Section 2):** `N_CPU_CORES` (`None` = all logical CPUs, or e.g. `10` to cap). `OPTUNA_N_JOBS` (`None` = auto: `1` on CUDA/MPS; on CPU, `min(4, n_cores//2)`). Thread limits via `src/utils/cpu_threads.py`: Section 2 bootstraps with **full cores for §8** (`parallel_jobs=1`); §6b re-applies `n_cores // OPTUNA_N_JOBS` threads per trial inside each Optuna worker. **On one GPU, keep `OPTUNA_N_JOBS=1`.** Example CPU tuning: `N_CPU_CORES=10`, `OPTUNA_N_JOBS=4` → 2 threads/trial. More cores is not always faster for this MLP (memory bandwidth).

- **External monitor** (`scripts/monitor/monitor_nn_training_progress.py`) — from repo root:

  ```bash
  python scripts/monitor/monitor_nn_training_progress.py
  ```

  | Flag | Purpose |
  |------|---------|
  | `MAIN_6` / `MAIN_7` | Exactly one `True` |
  | `LIVE` | `False` = one-shot; `True` = refresh until log idle (~90s) |

  Picks the **newest mtime** in `data/logs/` (Optuna JSON vs training CSV). §8 view: train MSE, test R², train−test gap. §6b view: trial curve + parallel coordinates. `LIVE` waits 30s for logs to appear, then exits if missing. Details: [`data/logs/README.md`](../data/logs/README.md).

- **Full-profile inlet anchoring (Main_7 §9):** same `anchor_inlet_profile_predictions()` as Main_5 — applied after test `predict`, before metrics and §9b axial overlays.

- **Main_7 — data splits and overfitting:**
  - **§4 test runs** (~`test_size` of simulation runs): held out for §8 checkpoints, LR scheduler, early stopping, and best-weight restore. Never used in Optuna.
  - **§6b validation rows** (`validation_fraction` of **train** rows, row-level shuffle): Optuna objective = **validation R²** only. Test runs stay blind. Many trials can overfit this val fold (hyperparameter selection bias); §8 test R² is the honest generalization check.
  - **§8 overfitting diagnostic:** growing **train R² − test R²** gap at checkpoints → raise `dropout` or shrink `h1`–`h3` in config (see notebook **Overfitting controls used here**).

- **Main_7 — row cap, axial parity:** Section 2 **`FULL_PROFILE_MAX_ROWS`** optionally caps total train+test rows after the run-level split (`None` = all rows). Section **§9b** — **`AXIAL_PROFILE_N_RUNS`**, **`AXIAL_PROFILE_RUNS_RANDOM`**, overlays **state + species/lumps** along **`x/L`**. Section **§10** — **`PARITY_HEXBIN_MIN_POINTS`** selects **hexbin** (shared log colorbar) vs **scatter** when `n_test` is small.

Any missing key falls back to the inline notebook defaults shown above. Edit the JSON and re-run Section 3 in Main_6 or Main_7 — no kernel restart needed.

### 3. ML Inference

**File**: `configs/ml/ml_inference_config.json`

```json
{
    "model_dir": "models",
    "model_type": "xgboost",
    "target_type": "primary",
    "simulation_parameters": {
        "initial_temperature_K": 925.0,
        "initial_pressure_bar": 2.0,
        "reactor_length_m": 5.0,
        "reactor_diameter_mm": 30.0,
        "mass_flow_rate_kgps": 0.07,
        "heat_flux_Wm2": 150000.0,
        "reactant_type": null
    },
    "prediction_settings": {
        "n_points": 200,
        "adaptive_step": false,
        "max_step_size": 0.1
    },
    "output": {
        "file": "outputs/predictions.csv",
        "format": "csv"
    }
}
```

**Usage:**
```bash
python src/ml/inference.py configs/ml/ml_inference_config.json
```

**Parameters:**
- `model_dir`: Directory containing trained models
- `model_type`: Tree model key to use (`random_forest`, `xgboost`, `gradient_boosting`, `adaboost`)
- `target_type`: Type of targets (`primary`, `secondary`, `species`)
- `simulation_parameters`: Reactor operating conditions
- `prediction_settings`: Prediction configuration
  - `n_points`: Number of points along reactor (if `adaptive_step` is false)
  - `adaptive_step`: Use adaptive step size (true/false)
  - `max_step_size`: Maximum step size for adaptive stepping
- `output`: Output file configuration
  - `file`: Output file path
  - `format`: Output format (`csv` or `json`)

## Creating Custom Configurations

1. Copy the template config file from `configs/ml/`
2. Modify parameters as needed
3. Save with a descriptive name
4. Run the script with your config file

**Example:**
```bash
# Copy template
cp configs/ml/ml_data_generation_config.json configs/ml/my_training_config.json

# Edit configs/ml/my_training_config.json with your parameters

# Run with custom config
python src/ml/data_generation.py configs/ml/my_training_config.json
```

## Benefits of JSON Configuration

1. **Reproducibility**: All parameters saved in one file
2. **Version Control**: Easy to track configuration changes
3. **Flexibility**: Easy to create multiple configurations
4. **Documentation**: Self-documenting with comments
5. **No Command-Line Limits**: No need to remember long argument lists
