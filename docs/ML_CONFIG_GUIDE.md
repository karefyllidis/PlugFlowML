# ML Configuration Guide

All ML scripts now use JSON configuration files instead of command-line arguments.

**Convention**: notebook flag cells hold only boolean (`IF_*`) literals and `Path` objects. Every number, string, or list a user might reasonably tune — model lists, tuning budgets, row caps, CPU/Optuna job counts, teacher/model selectors — is read from the matching JSON config via `config.get(key, default)`. This keeps "what changes behaviour" entirely in the config files and out of code cells, so re-running a notebook can't accidentally pick up a hand-edited number.

## Configuration Files

### 1. Training Data Generation

**File**: `configs/ml/main2_data_generation_config.json`

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
python src/ml/data_generation.py configs/ml/main2_data_generation_config.json
```

**Optional — SLURM chunk runner** (`scripts/cluster/run_main2_slurm_chunk.py`): set `HYDRAI_ML_CONFIG` to a JSON path (absolute or relative to repo root) to override the default `configs/ml/main2_data_generation_config.json`. For a minimal test workload use `configs/ml/main2_data_generation_config.smoke.json`. Each task writes live status to `logs/data_generation_progress_task_<TASK_ID>.json`. See `README.md` (HPC / SLURM).

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

**File**: `configs/ml/main1_run_pfr_config.json` — `reactant_key` (one of the keys in `configs/simulation/main1_reactant_database.json`, e.g. `ethane`, `propane`, `naphtha`, `n-hexane`).

- **`IF_SAVE_PLOTS`**: If `True`, quick inline figures are saved to `outputs/figures/`:
  - `quick_profiles_<reactant>.png`
  - `conversion_products_<reactant>.png`

### Notebook run control (`notebooks/Main_3_data_exploration_feature_engineering.ipynb`)

**File**: `configs/ml/main3_eda_feature_engineering_config.json` — loaded in cell 4 ("Load config"); every key below falls back to the inline default shown if the file or key is missing. Edit the JSON and re-run cell 4 — no kernel restart needed.

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
- **`IF_VELOCITY_QC`** / **`VELOCITY_QC_QUANTILE`** (§2.1b): drop simulation runs with non-physical or extreme-outlier velocity before feature/target export.
- **`IF_COMPUTE_RATES`** (§3c): compute `df_rates` (finite-difference reaction rate proxies) for Main_8; requires `EXPORT_SPECIES_AS=lumped_chemistry`.
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
- Exports baseline models to `models/tree_baseline/tree_models_exit.joblib` (overwritten each run)

Loads the latest `data/processed/features_targets_*.pkl`. If Main_3 used **`EXPORT_SPECIES_AS=lumped_chemistry`** (or `lumped_carbon`), targets are already **`Y_lump_*`** mass-fraction lumps; the notebook trains on those (far fewer outputs than hundreds of species).

**Tuning / evolution notebook:** `notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb`
- Tunes **one selected tree model** via `MODEL_TO_TUNE` (`random_forest`, `gradient_boosting`, `xgboost`, or `adaboost`)
- Runs `BayesSearchCV` for the exit-plane inlet→outlet model
- Optionally runs full axial/PFR evolution workflow with `TRAIN_FULL_PROFILE=True`
- Full-profile model training runs when `TRAIN_FULL_PROFILE=True` (no separate hyperparameter search): it **reuses** the exit-plane model’s `estimator__*` hyperparameters—either from `BayesSearchCV` when tuning is enabled or from the defaults used when training cell 6 without tuning.
- Full-profile uses all axial rows and includes `relative_position` as an input
- Full-profile train/test data are split by simulation run, not by row, to avoid leakage between axial points from the same PFR profile
- `FULL_PROFILE_MAX_ROWS` can be set for a quick full-profile training smoke test on large datasets
- **Inlet BC anchoring (§8):** after full-profile `predict`, `src/utils/profile_predictions.anchor_inlet_profile_predictions()` sets each test run’s prediction at **min `relative_position`** to match Cantera truth so axial overlay plots (§9) start at the same inlet state. Same helper is used in **Main_6 §9**.
- Reports tuned ML inference speed for exit-plane and full-profile prediction; set `CANTERA_EXIT_SECONDS_PER_RUN` / `CANTERA_FULL_PROFILE_SECONDS_PER_RUN` to print speedup factors against measured Cantera timings
- Exports tuned exit and full-profile artifacts to `models/tree_tuned/tree_model_tuned_exit_full.joblib` (overwritten each run)
- Adds axial diagnostics and regime diagnostics:
  - `full_profile_cantera_vs_ml_axial_evolution.png` (Main_5 full-profile trees: Cantera vs ML along `x/L` at selected stations)
  - `full_profile_cantera_vs_nn_axial_evolution.png` (Main_6 §9b: Cantera / test vs PyTorch NN; preferred state columns plus all `species_cols` present in `target_cols`, same `x/L` grid)
  - `exit_error_vs_conditions_boxplots.png` (error vs inlet-condition bins)
  - `exit_error_tp_map.png` (temperature-pressure error map)

**Why full-profile matters:** Steam cracking is an axial-evolving reacting flow. Exit-only surrogates predict final yields; full-profile surrogates predict the entire temperature/species evolution along the reactor — essential for coil design, coking analysis, and process optimization.

**Every Main notebook that reads JSON owns its own dedicated config file** — no file is shared across two Main_N notebooks. This keeps "which config changes which notebook" unambiguous; see the table in `CLAUDE.md` § Config structure for the full list.

**File**: `configs/ml/main4_tree_baseline_config.json` (Main_4):

```json
{
    "test_size": 0.2,
    "random_state": 42,
    "models_to_train": ["random_forest", "gradient_boosting", "xgboost", "adaboost"],
    "random_forest": { "n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1 },
    "xgboost": { "n_estimators": 150, "max_depth": 6, "learning_rate": 0.1, "reg_alpha": 0.0, "reg_lambda": 1.0 },
    "gradient_boosting": { "n_estimators": 150, "max_depth": 5, "min_samples_leaf": 1 },
    "adaboost": { "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6 },
    "tuning": { "n_iter": 60, "patience": 10, "min_delta": 0.0005, "cv": 3, "scoring": "neg_mean_absolute_error" }
}
```

**File**: `configs/ml/main5_tree_tuning_config.json` (Main_5):

```json
{
    "test_size": 0.2,
    "random_state": 42,
    "model_to_tune": "xgboost",
    "full_profile_max_rows": null,
    "random_forest": { "n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1 },
    "xgboost": { "n_estimators": 100, "max_depth": 6, "learning_rate": 0.3, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 1.0 },
    "gradient_boosting": { "n_estimators": 150, "max_depth": 5, "min_samples_leaf": 1 },
    "adaboost": { "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6 },
    "tuning": { "n_iter": 60, "patience": 10, "min_delta": 0.0005, "cv": 3, "scoring": "neg_mean_absolute_error" }
}
```

**Usage (baseline notebook):** `jupyter notebook notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb` — reads `main4_tree_baseline_config.json` in cell 3. `IF_HYPERPARAM_TUNING` (off by default) is the only notebook-only flag; when `True`, Section 7 runs `BayesSearchCV` on every model in `models_to_train` using the `tuning.*` budget.

**Usage (tuning/evolution notebook):** `jupyter notebook notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb` — reads `main5_tree_tuning_config.json` in cell 3. `IF_HYPERPARAM_TUNING` and `TRAIN_FULL_PROFILE` are the only notebook-only flags; `model_to_tune` and the `tuning.*` budget are config-driven. The tree blocks above are the **untuned** defaults used when `IF_HYPERPARAM_TUNING=False`.

**Parameters** (both files): `test_size` (float, fraction held out), `random_state` (int seed), and one block per tree model with its `n_estimators`/`max_depth`/etc. Any missing key falls back to the inline notebook default shown as the second argument to `config.get(...)`.

**File**: `configs/ml/main6_simplenn_config.json` (Main_6 — PyTorch `SimpleNN`, full axial profile):

```json
{
    "test_size": 0.2,
    "random_state": 42,
    "runtime": {
        "subsample_max_rows": 1000,
        "n_cpu_cores": 10,
        "optuna_n_jobs": 10
    },
    "neural_network": {
        "epochs": 200,
        "batch_size": 256,
        "learning_rate": 0.001,
        "h1": 128,
        "h2": 64,
        "h3": 32,
        "dropout": 0.1,
        "tuning": {
            "n_trials": 15,
            "epochs_per_trial": 50,
            "validation_fraction": 0.2,
            "timeout_seconds": null
        }
    },
    "evaluation": {
        "axial_stations": [0.25, 0.5, 0.75, 1.0],
        "nrmse_good_pct": 5.0,
        "nrmse_acceptable_pct": 15.0,
        "mc_dropout_samples": 50,
        "exit_station_tol": 0.02
    }
}
```

`runtime.*` is read early in Section 2 (Paths & Flags), before `neural_network.*`/`tuning.*` (read in Section 3) — thread pools must be configured before any data operations. `subsample_max_rows` caps rows when `SUBSAMPLE_ROWS=True` (the only notebook-only boolean here); `n_cpu_cores` (`null` = all logical CPUs) and `optuna_n_jobs` control `src/utils/cpu_threads.py` thread allocation.

**`neural_network` parameters:**
- `epochs` (int): number of Adam epochs over the training set.
- `batch_size` (int): mini-batch size for `DataLoader(shuffle=True)`. Capped at `len(train_ds)` if larger.
- `learning_rate` (float): Adam learning rate.
- `h1`, `h2`, `h3` (int): number of units in hidden layers 1–3 of the multi-output `SimpleNN` MLP.
- `dropout` (float): probability passed to `nn.Dropout` after each hidden ReLU; off automatically under `model.eval()`.

**`neural_network.tuning` parameters (Section 6b only when `IF_HYPERPARAM_TUNING=True`):**
- `n_trials` (int): number of Optuna TPE trials in the study.
- `epochs_per_trial` (int): training epochs per Optuna trial; intentionally smaller than the final-model `epochs` to keep the search cheap. Median pruner stops weak trials early.
- `validation_fraction` (float): fraction of the **training** split carved out as the Optuna validation fold. The held-out test set is never seen during tuning.
- `timeout_seconds` (int or `null`): wallclock cap for the entire study; `null` = no time limit.

The tuning search space (`h1 ∈ [32,256] step 32`, `h2 ∈ [16,128] step 16`, `h3 ∈ [8,64] step 8`, `dropout ∈ [0.0,0.3]`, `learning_rate ∈ [1e-4,1e-2]` log, `batch_size ∈ {64,128,256,512}`) is defined in the notebook objective function. The objective maximises validation R² (uniform average across all targets, physical units). Requires `pip install optuna`.

**`evaluation` parameters (physics-aware diagnostics, read in Section 3; used in Sections 9c/9d/10d/10e):**
- `axial_stations` (list of float): x/L fractions along the reactor at which the §10e NRMSE% station bars (and §9b overlay guide lines) are evaluated; the largest value is treated as the exit plane for the §9c species-sum exit check.
- `nrmse_good_pct` / `nrmse_acceptable_pct` (float): green (<good) and yellow (<acceptable) acceptance bands drawn on the §10e NRMSE% bars.
- `mc_dropout_samples` (int): stochastic forward passes for the §9d MC-Dropout ±σ uncertainty bands (`SimpleNN.predict_with_uncertainty`).
- `exit_station_tol` (float): x/L tolerance reserved for isolating exit-plane rows in the §10d joint panels (exit rows default to each test run's maximum `relative_position`).

Each diagnostic is gated by a Section 2 boolean (`IF_EXCLUDE_INLET_ROW`, `IF_SPECIES_SUM_CHECK`, `IF_MC_DROPOUT`, `IF_EXIT_JOINT_PANELS`, `IF_AXIAL_STATION_BARS`); the core train/eval/export pipeline runs whether they are on or off. Diagnostic helpers live in `src/utils/profile_evaluation.py`.

**Main_6 production training (Section 8):** `ReduceLROnPlateau` stepped on periodic **test** R² checkpoints, **early stopping** if test R² fails to improve across several consecutive checkpoints, then **reload the best test-R² weights** before Section 9 metrics and Section 11 export. Exports `models/simple_nn_full_profile/simple_nn_full_profile_manifest.json` (plus `workflow`, `run_level_split`, `inlet_row_excluded`, `feature_cols` incl. `relative_position`, `run_cols`, row/run counts, `chemistry_groups`, `metrics_by_group`, an `evaluation` block, `species_sum` and `mc_dropout` diagnostic summaries, and `auxiliary_exports`) and `simple_nn_full_profile_{per_target,group}_metrics.csv` in the same subfolder. When `IF_AXIAL_STATION_BARS` is on it also writes `simple_nn_full_profile_axial_station_metrics.csv`. The per-target/group CSVs are written once (Section 11); Section 9 no longer duplicates them.

**Main_6 notebook-only controls (booleans and paths only; everything else lives in the JSON config above):**

- **External training progress (Section 2):** `WRITE_TRAINING_PROGRESS_LOG` (default `True`) appends `data/logs/Main_6_..._training_progress.csv` during §8. Optuna §6b rewrites `data/logs/Main_6_..._optuna_tuning_plot_data.json` after each completed trial. `USE_CUDA_AMP`, `USE_TORCH_COMPILE` are notebook booleans; `OPTUNA_N_JOBS` is config-driven (`runtime.optuna_n_jobs`, keep at `1` on a single GPU when tuning).
- **CPU parallelism:** `N_CPU_CORES` is config-driven (`runtime.n_cpu_cores`, `null` = all logical CPUs). Thread limits via `src/utils/cpu_threads.py`.
- **External monitor** (`scripts/monitor/monitor_nn_training_progress.py`): picks the newest mtime in `data/logs/` (Optuna JSON vs training CSV); `LIVE` flag toggles one-shot vs refresh-until-idle. Details: [`data/logs/README.md`](../data/logs/README.md).
- **Inlet BC anchoring (§9):** `src/utils/profile_predictions.anchor_inlet_profile_predictions()` sets each test run's prediction at min `relative_position` to match Cantera truth, so axial overlay plots start at the same inlet state. Same helper used in Main_5 §8.
- **Inlet-row exclusion (§4):** `IF_EXCLUDE_INLET_ROW` (default `True`) drops each run's x/L≈0 row from train/test — the τ=0 state is trivially known from the inputs, so keeping it flatters metrics; the inlet is still anchored at inference (above). §4 also asserts `run_cols` are constant within each run (the intent of the reference's `enforce_inlet_bc`, redundant here since runs are defined by `groupby(run_cols)`).
- **Physics-aware diagnostics (§9c/9d/10d/10e):** `IF_SPECIES_SUM_CHECK`, `IF_MC_DROPOUT`, `IF_EXIT_JOINT_PANELS`, `IF_AXIAL_STATION_BARS` gate the species-sum closure, MC-Dropout ±σ bands, exit-plane joint panels, and axial NRMSE% station bars respectively. Numeric controls live in `evaluation.*` (above); helpers in `src/utils/profile_evaluation.py`.
- **Data splits and overfitting:** §4 test runs (~`test_size` of simulation runs) are held out for §8 checkpoints, LR scheduler, and best-weight restore (training always runs the full `epochs`, no early stopping), and are never used in Optuna; §6b validation rows come from the **train** split only. Growing train R² − test R² gap at §8 checkpoints → raise `dropout` or shrink `h1`–`h3`.
- **Row cap, axial parity:** `FULL_PROFILE_MAX_ROWS` is derived from `runtime.subsample_max_rows` when `SUBSAMPLE_ROWS=True` (the notebook boolean); it caps total train+test rows after the run-level split. §9b `AXIAL_PROFILE_N_RUNS`, `AXIAL_PROFILE_RUNS_RANDOM`. §10 `PARITY_HEXBIN_MIN_POINTS` selects hexbin vs scatter.

Any missing key falls back to the inline notebook defaults. Edit the JSON and re-run Section 3 in Main_6 — no kernel restart needed.

**File**: `configs/ml/main7_pinn_config.json` (Main_7 — PINNPFR with PFR ODE residuals):

```json
{
    "test_size": 0.2,
    "random_state": 42,
    "full_profile_max_rows": null,
    "neural_network": {
        "epochs": 400, "batch_size": 256, "learning_rate": 0.001,
        "h1": 128, "h2": 64, "h3": 32, "dropout": 0.1,
        "tuning": { "n_trials": 15, "epochs_per_trial": 25, "validation_fraction": 0.2, "timeout_seconds": null }
    },
    "pinn": {
        "loss_weights": {
            "lambda_data": 1.0, "lambda_phys": 0.1, "lambda_eos": 1.0,
            "lambda_mass": 1.0, "lambda_species_sum": 1.0,
            "lambda_species_nonneg": 0.5, "lambda_energy_ode": 1.0
        },
        "training": {
            "curriculum_warmup_epochs": 20, "n_colloc_per_batch": 256,
            "phys_loss_freq": 1, "use_cantera_residuals": false
        }
    }
}
```

`neural_network.*` mirrors Main_6's architecture block but lives in its own file (self-contained — PINN changes never touch Main_6's config). `full_profile_max_rows` (int or `null`) is a smoke-test row cap. `pinn.loss_weights.*` are per-term multipliers for the composite loss `L = λ_data·MSE + λ_phys·L_physics`; `pinn.training.*` controls the curriculum warmup and collocation-point budget. See `CLAUDE.md` § PINN specifics for the full physics description.

**Main_7 optional tuning (Section 6b, `IF_HYPERPARAM_TUNING=True`, off by default):** unlike Main_6, each trial trains on the **data loss only** (no physics/collocation terms, no curriculum) for `neural_network.tuning.epochs_per_trial` epochs — a cheap proxy search over `h1, h2, h3, dropout, learning_rate, batch_size`, not over the physics loss weights. Adopts the best trial and rebuilds `model` before Section 7 (collocation) onward.

**Legacy standalone script (not part of the Main_N notebook pipeline):**

**File**: `configs/ml/model_training_script_config.json`

```json
{
    "data_file": "data/training/training_data_complete_*.csv",
    "output_dir": "models",
    "target_types": ["primary"],
    "models": ["all"]
}
```

**Usage:**
```bash
python src/ml/model_training.py configs/ml/model_training_script_config.json
```

**Parameters:**
- `data_file`: Path to training data CSV (supports glob patterns)
- `output_dir`: Directory to save trained models
- `target_types`: List of target types (`primary`, `secondary`, `species`, `all`)
- `models`: List of models to train (`neural_network` is an unimplemented placeholder — production NN training is the Main_6/Main_7 notebooks. `all` expands to RF + XGBoost + GB only.)

`test_size` and `random_state` are **not** read by this script (it uses fixed in-code defaults); the tree hyperparameters are likewise fixed in-code rather than sourced from JSON. This script predates the Main_4/Main_5 notebooks and is kept for quick CLI-only smoke tests.

### 3. Symbolic Regression (Main_8)

**File**: `configs/ml/main8_symbolic_regression_config.json`

```json
{
    "if_plot_shown": true,
    "if_plot_export": true,
    "if_sr_export": true,
    "teacher_stem": "simple_nn_full_profile",
    "n_distill_samples": 5000,
    "random_state": 42,
    "sr_n_iterations": 100,
    "sr_populations": 30,
    "sr_population_size": 33,
    "sr_maxsize": 25,
    "sr_procs": 0
}
```

- `teacher_stem`: which trained NN to distil — `simple_nn_full_profile` (Main_6) or `pinn_pfr` (Main_7).
- `n_distill_samples`: number of (run, z) rows sampled from the processed dataset to query the teacher for distillation targets.
- `sr_n_iterations`, `sr_populations`, `sr_population_size`, `sr_maxsize`, `sr_procs`: PySR search budget — increase for production runs (see notebook Summary for tips).

**Usage:** `jupyter notebook notebooks/Main_8_symbolic_regression_SR.ipynb` — reads the config in cell 2. Requires the source notebook (Main_6 or Main_7) to have been run with `IF_MODEL_EXPORT=True` first.

### 4. Cantera / PINN / SR Comparison (Main_9)

**File**: `configs/ml/main9_compare_cantera_pinn_sr_config.json`

```json
{
    "if_plot_shown": true,
    "if_plot_export": true,
    "if_export_report": true,
    "sr_teacher_stem": "pinn_pfr",
    "n_comparison_runs": 6,
    "random_state": 42
}
```

- `sr_teacher_stem`: which Main_8 SR export to load (`pinn_pfr` → `models/sr_pinn/`, `simple_nn_full_profile` → `models/sr_full_profile/`). Use `pinn_pfr` so SR is compared against the same PINN it was distilled from.
- `n_comparison_runs`: number of full-profile runs sampled (from the full processed dataset, not a strict held-out test set) for the axial-overlay plots.

**Usage:** `jupyter notebook notebooks/Main_9_compare_cantera_pinn_sr.ipynb` — reads the config in cell 2. Requires Main_7 (`IF_MODEL_EXPORT=True`) and Main_8 (matching `TEACHER_STEM`, `IF_SR_EXPORT=True`) to have been run first.

### 5. Bayesian Optimisation (Main_10)

**File**: `configs/ml/main10_bayesian_optimisation_config.json`

```json
{
    "opt_target": "Y_lump_olefins",
    "n_trials": 150,
    "random_state": 42,
    "reactant_key": "n-hexane",
    "if_plot_shown": true,
    "if_plot_export": true,
    "bounds": {
        "T_K": [800.0, 900.0],
        "P_Pa": [150000.0, 350000.0],
        "L_m": [10.0, 15.0],
        "D_m": [0.025, 0.040],
        "mdot": [0.05, 0.10],
        "q_Wm2": [100000.0, 250000.0]
    }
}
```

- `opt_target`: column name in the SR `target_cols` to maximise at reactor exit.
- `n_trials`: Optuna `GPSampler` study budget.
- `bounds`: inlet-condition search domain for the 6 optimisation variables — should stay within the training-data domain (see Main_2/Main_3 parameter ranges).

**Usage:** `jupyter notebook notebooks/Main_10_optimisation_BO_surrogate_vs_cantera.ipynb` — reads the config in cell 2. Requires Main_8 to have been run with `IF_SR_EXPORT=True` first.

### 6. ML Inference

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
cp configs/ml/main2_data_generation_config.json configs/ml/my_training_config.json

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
