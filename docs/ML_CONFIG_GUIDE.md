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
- Exports baseline models to `models/tree_models_exit_<timestamp>.joblib`

Loads the latest `data/processed/features_targets_*.pkl`. If Main_3 used **`EXPORT_SPECIES_AS=lumped_chemistry`** (or `lumped_carbon`), targets are already **`Y_lump_*`** mass-fraction lumps; the notebook trains on those (far fewer outputs than hundreds of species).

**Tuning / evolution notebook:** `notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb`
- Tunes **one selected tree model** via `MODEL_TO_TUNE` (`random_forest`, `gradient_boosting`, `xgboost`, or `adaboost`)
- Runs `BayesSearchCV` for the exit-plane inlet→outlet model
- Optionally runs full axial/PFR evolution workflow with `TRAIN_FULL_PROFILE=True`
- Full-profile model training runs when `TRAIN_FULL_PROFILE=True`:
  - with `IF_HYPERPARAM_TUNING=True`: `BayesSearchCV` tuning
  - with `IF_HYPERPARAM_TUNING=False`: default-parameter training (no tuning)
- Full-profile uses all axial rows and includes `relative_position` as an input
- Full-profile train/test data are split by simulation run, not by row, to avoid leakage between axial points from the same PFR profile
- `FULL_PROFILE_MAX_ROWS` can be set for a quick tuning smoke test on large datasets
- Reports tuned ML inference speed for exit-plane and full-profile prediction; set `CANTERA_EXIT_SECONDS_PER_RUN` / `CANTERA_FULL_PROFILE_SECONDS_PER_RUN` to print speedup factors against measured Cantera timings
- Exports tuned exit and full-profile artifacts to `models/tree_model_tuned_exit_full_<timestamp>.joblib`
- Adds axial diagnostics and regime diagnostics:
  - `full_profile_cantera_vs_ml_axial_evolution.png` (Cantera vs ML axial overlays at selected `x/L`)
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
        "epochs": 50,
        "batch_size": 256
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
- `models`: List of models to train (`neural_network` is a PyTorch placeholder — omit or list explicitly; `random_forest`, `xgboost`, `gradient_boosting`, `adaboost`; `all` = RF + XGBoost + GB only until NN is implemented)
- `test_size`: Fraction of data for testing (0.0-1.0)
- `random_state`: Random seed for reproducibility
- Model-specific parameters: See individual model sections

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
