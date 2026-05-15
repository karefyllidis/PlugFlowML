# ML Surrogate Models for PFR Simulation

This module implements machine learning models to replace Cantera simulations, providing **100-1000x faster** predictions while maintaining accuracy.

## Overview

The ML Surrogate Models module consists of three main components:

1. **Training Data Generation** - Generate massive datasets from Cantera simulations
2. **Data Exploration & Feature Engineering** - Clean data, define features and targets
3. **ML Model Training** - Train multiple ML algorithms (Random Forest, Gradient Boosting, XGBoost, AdaBoost, Neural Networks)
4. **Model Comparison** - Evaluate all models on held-out test set with comprehensive error metrics and visual comparison
5. **ML Inference** - Use trained models for fast predictions

## Quick Start

### Step 1: Generate Training Data

Generate a large training dataset by running parameter sweeps:

```bash
# Generate training data using JSON config file
python src/ml/data_generation.py configs/ml/ml_data_generation_config.json
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/Main_2_generate_training_data.ipynb
```

**Expected Output:**
- Training data files in `data/training/`:
  - `training_data_complete_YYYYMMDD_HHMMSS.pkl` - Complete dataset (primary format; pickle for fast loading)
  - Optional: `metadata_YYYYMMDD_HHMMSS.json` - Generation metadata (when saving enabled)
- Partial saves during generation: `training_data_partial_*.pkl` (automatically cleaned up after completion)

### Step 2: Train ML Models

**Option A – Baseline tree models (recommended first, Jupyter notebook):**
```bash
jupyter notebook notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb
```
Trains default Random Forest, Gradient Boosting, XGBoost, and AdaBoost via `MultiOutputRegressor` on inlet→outlet / exit-plane data. This notebook is intentionally **not tuned**: it is for fast model-family comparison. It also plots Normalized MAE by chemistry/carbon lump and by exit state/thermo/aero target, and reports ML inference speed with optional Cantera/PFR speedup.

**Step 2b — One-model tuning and PFR evolution (Jupyter notebook):**
```bash
jupyter notebook notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb
```
Tunes one selected tree model (`MODEL_TO_TUNE`) with `BayesSearchCV` on the exit-plane task when tuning is enabled; the full axial/PFR model reuses those hyperparameters. It covers inlet→outlet exit-plane prediction and, when enabled, full axial evolution using `relative_position` as an input. Full-profile train/test splitting is done by simulation run to avoid leakage between axial points from the same reactor profile. Speed reports compare tuned ML inference against measured Cantera baselines when `CANTERA_EXIT_SECONDS_PER_RUN` / `CANTERA_FULL_PROFILE_SECONDS_PER_RUN` are set.

**Step 2c — PyTorch MLP baseline (Jupyter notebook):**
```bash
jupyter notebook notebooks/Main_6__train_evaluate_SimpleNN_IO.ipynb
```
Defaults-only PyTorch counterpart of Main_4 on the same exit-plane data. Inputs are restricted to inlet / run-design columns (no axial coordinates); architecture is a 3-hidden-layer ReLU MLP (defaults `128 → 64 → 32`) with `nn.Dropout` between hidden blocks, `nn.MSELoss`, Adam. Reads `test_size`, `random_state`, and `neural_network.{epochs, batch_size, learning_rate, h1, h2, h3, dropout}` from `configs/ml/ml_training_config.json` (Section 3 of the notebook). Plots train + test convergence (MSE and R² vs epoch), **3-column** parity and residual grids for **all state + species** targets (`actual_vs_predicted_scatter_by_target.png`, `residuals_scatter_by_target.png`), and a per-target R² bar chart. Optional architecture diagram (matplotlib + standalone TikZ source) and `torchinfo` summary are gated by flags. Exports under `models/`: `simple_nn_exit_state_dict.pt`, `simple_nn_exit_scalers.joblib`, `simple_nn_exit_manifest.json` (includes **`chemistry_groups`**, **`metrics_by_group`**, **`auxiliary_exports`**), plus **`simple_nn_exit_per_target_metrics.csv`** and **`simple_nn_exit_group_metrics.csv`** (each run overwrites the previous files).

**Step 2d — PyTorch full axial profile (optional notebook):** `notebooks/Main_7_train_evaluate_SimpleNN_full_profile.ipynb` — same `neural_network.*` keys as Main_6, **run-level** train/test split, inputs include **`relative_position`**, optional **`FULL_PROFILE_MAX_ROWS`** for capped-row smoke runs. **§9** evaluates on the test tensor and (when `IF_MODEL_EXPORT`) writes **`simple_nn_full_profile_per_target_metrics.csv`** and **`simple_nn_full_profile_group_metrics.csv`** (same filenames are refreshed in **§11** with the full manifest). **§9b** overlays Cantera/test vs NN along **`x/L`** for preferred state targets **plus all species/lump columns** in `target_cols`; test runs are either **fixed order** (first *N* distinct `run_id` in DataFrame order) or **random** (`AXIAL_PROFILE_RUNS_RANDOM=True`, seed `RANDOM_STATE`). **§10** is a **four-column** actual-vs-predicted grid with a **single shared log colorbar** for hexbin counts when `n_test` is large; otherwise scatter. Figures live under `outputs/figures/Main_7_train_evaluate_SimpleNN_full_profile/` (e.g. `full_profile_cantera_vs_nn_axial_evolution.png`, `actual_vs_predicted_scatter_by_target.png`, `residuals_scatter_by_target.png`, `training_convergence.png`, optional Optuna PNGs). Exports `simple_nn_full_profile_*` under `models/`. See root `README.md` and `docs/ML_CONFIG_GUIDE.md`.

*Optional Optuna tuning (Section 6b):* set `IF_HYPERPARAM_TUNING=True` in Section 2 to run an in-notebook TPE search over `h1, h2, h3, dropout, learning_rate, batch_size` on a validation fold carved from the training split (test set held out). The best trial overwrites the notebook's hyperparameters and rebuilds the model before the main training loop. Search budget and validation fraction are configured under `neural_network.tuning` in the config. Requires `pip install optuna`. Section 6b-ii (after tuning) writes `optuna_optimization_history.png`, `optuna_parallel_coordinate.png`, `optuna_param_importance.png`, and a `tuning` block in the exported manifest. **Main_7:** test runs are split at §4 by `run_id`; Optuna val is a random row fraction of train rows only — monitor with `OPTUNA=True` during §6b, then `OPTUNA=False` for §8 train/test gap plots (`docs/ML_CONFIG_GUIDE.md`).

**Option B – All model types (command-line):**
```bash
python src/ml/model_training.py configs/ml/ml_training_config.json
```

**Available models (notebooks: trees in Main_4/Main_5, PyTorch MLP in Main_6/Main_7; CLI script: RF + XGBoost + gradient boosting; CLI `neural_network` = placeholder):**
- `random_forest` - Random Forest (scikit-learn)
- `gradient_boosting` - Gradient Boosting (scikit-learn)
- `xgboost` - XGBoost
- `adaboost` - AdaBoost with tree base (scikit-learn)
- `neural_network` - **PyTorch MLP** training lives in `notebooks/Main_6__train_evaluate_SimpleNN_IO.ipynb` (exit-plane) and `notebooks/Main_7_train_evaluate_SimpleNN_full_profile.ipynb` (full axial profile). The same key in `src/ml/model_training.py` is still a no-op CLI placeholder.

**Target Types:**
- `primary` - Core outputs (temperature, pressure, velocity, density)
- `secondary` - Thermodynamic properties (enthalpy, entropy, heat capacity, etc.)
- `species` - Species concentrations (mass/mole fractions)
- `all` - All targets combined

### Step 3: Use ML Models for Prediction

Use trained models instead of Cantera:

```bash
# Predict reactor profile using ML model
python src/ml/inference.py configs/ml/ml_inference_config.json
```

## Architecture

### Training Data Generation

The `generate_training_data.py` script:

1. **Parameter Sweeps**: Varies key parameters:
   - Temperature: 800-1200 K (10 points)
   - Pressure: 1.5-3.0 bar (8 points)
   - Reactor length: 3.0-7.0 m (6 points)
   - Reactor diameter: 20-40 mm (5 points)
   - Mass flow rate: 0.05-0.10 kg/s (6 points)
   - Heat flux: 100,000-200,000 W/m² (5 points)

2. **Data Collection**: For each simulation, collects:
   - **Input Features**: Initial conditions, geometry, position
   - **Output Targets**: Temperature, pressure, species, properties at each position

3. **Efficient Generation**: 
   - Disables plots and CSV exports during generation
   - Saves partial data periodically as pickle files (faster I/O)
   - **Latin Hypercube Sampling (LHS)** - Use `sampling_method: "latin"` in config for better parameter-space coverage with fewer runs
   - **Random** - `sampling_method: "random"`; bounds via `random_sample_bounds`
   - **Structured grid** - `sampling_method: "full_grid"`, `"structured_grid"`, or `"grid"`; grid from `parameter_ranges` (product of `n_points`)
   - **Parallel processing** - Use multiple CPU cores (configure via `n_jobs`)
   - **Memory efficient** - Clears data from memory after each save when saving to disk
   - **Automatic cleanup** - Deletes partial files after successful completion
   - **Real-time progress** - Progress, success rate, and ETA after each simulation
   - **Run control flags** (notebook) - `IF_SHOW_PLOTS`, `IF_SAVE_PLOTS`, `IF_SAVE_METADATA`, `IF_SAVE_TRAINING_DATA` to control display and saving
   - **Training space visualization** - Step 2.1 (sampling preview) and Step 4.1 (from generated data): 1D marginals and 2D pairwise plots to assess exploration quality

### ML Model Training

The `model_training.py` script and the current notebooks:
- `Main_4_train_and_evaluate_tree_models_IO.ipynb` for default-parameter exit-plane tree-model baseline evaluation.
- `Main_5_train_evaluate_tune_tree_model_evolution.ipynb` for one-tree-model tuning and full PFR evolution.
- `Main_6__train_evaluate_SimpleNN_IO.ipynb` for the PyTorch MLP exit-plane baseline (configurable via `neural_network.*` in `ml_training_config.json`; optional Optuna TPE in Section 6b via `IF_HYPERPARAM_TUNING=True` and `neural_network.tuning`; Section 8 applies LR reduction on stalled test R², early stopping, and restores the best test-R² checkpoint before evaluation/export). §8 training progress CSV; §6b incremental Optuna JSON — `scripts/monitor/monitor_nn_training_progress.py` (`MAIN_6=True`, `OPTUNA`/`FOLLOW`).
- `Main_7_train_evaluate_SimpleNN_full_profile.ipynb` for the PyTorch **full-profile** surrogate (same config block; **run-level** test holdout in §4; Optuna §6b on validation rows from train only; §8 train vs test R² overfitting check; optional row cap; monitor with `MAIN_7=True`; `USE_CUDA_AMP` / `USE_TORCH_COMPILE` / `OPTUNA_N_JOBS`; §9b axial state+species; §10 4-column parity + shared hexbin colorbar; exports `simple_nn_full_profile_*` + CSVs + figure PNGs).

1. **Data Preparation**:
   - Splits data into train/test sets
   - Scales features and targets
   - Handles categorical variables (reactant type)

2. **Model Training**:
   - `model_training.py` trains tree / boosting models from JSON (no PyTorch path in the CLI).
   - `Main_6__train_evaluate_SimpleNN_IO.ipynb` trains the PyTorch `SimpleNN` baseline with optional Optuna tuning; the main loop uses **`ReduceLROnPlateau`** on test R² checkpoints, **early stopping** when test R² stalls, then **restores the best test-R² checkpoint** before evaluation and export.
   - Notebooks evaluate on a held-out test split after training.

3. **Model Saving**:
   - Saves trained models
   - Saves scalers for preprocessing
   - Saves training metadata and metrics (Main_6 also writes `simple_nn_exit_manifest.json` with architecture, grouped R², **`chemistry_groups`**, **`metrics_by_group`**, **`auxiliary_exports`**, tuning block when used, and training-run fields such as `early_stopped` / best-checkpoint test R²; plus the two **CSV** exports beside the `.pt` / `.joblib` / `.json` bundle)

### ML Inference

The `ml_inference.py` script:

1. **Model Loading**: Loads trained model and scalers
2. **Prediction**: Fast predictions at any position
3. **Profile Generation**: Complete reactor profiles

## Features

### Input Features

- `initial_temperature_K` - Initial temperature
- `initial_pressure_Pa` - Initial pressure
- `reactor_length_m` - Reactor length
- `reactor_diameter_m` - Reactor diameter
- `mass_flow_rate_kgps` - Mass flow rate
- `heat_flux_Wm2` - Heat flux
- `z_position_m` - Axial position
- `relative_position` - Relative position (0-1)
- `reactant_type` - Reactant type (if included in training)

### Output Targets

**Primary Targets:**
- `temperature_K` - Temperature profile
- `pressure_Pa` - Pressure profile
- `velocity_ms` - Velocity profile
- `density_kgm3` - Density profile

**Secondary Targets:**
- `heat_capacity_cp_JkgK` - Heat capacity at constant pressure
- `heat_capacity_cv_JkgK` - Heat capacity at constant volume
- `mean_molecular_weight_kgkmol` - Mean molecular weight
- `enthalpy_Jkg` - Specific enthalpy
- `entropy_JkgK` - Specific entropy
- `viscosity_Pas` - Dynamic viscosity
- `thermal_conductivity_WmK` - Thermal conductivity

**Species Targets:**
- `Y_species` - Mass fractions for all species
- `X_species` - Mole fractions for all species

## Performance

### Speed Comparison

| Method | Time per Simulation | Speedup |
|--------|---------------------|---------|
| Cantera | ~10-60 seconds | 1x |
| ML (Neural Network) | ~0.01-0.1 seconds | 100-1000x |
| ML (Random Forest) | ~0.1-1 seconds | 10-100x |
| ML (XGBoost) | ~0.05-0.5 seconds | 20-200x |

### Accuracy

Model accuracy depends on:
- Training data size and diversity
- Model complexity
- Target type (primary targets typically more accurate)

Typical R² scores:
- Primary targets: 0.95-0.99
- Secondary targets: 0.90-0.95
- Species targets: 0.85-0.95

For surrogate use, **MAPE (or mean % error) around 5% or lower** is often considered good; single-digit percentage error is a typical goal in engineering surrogates.

## Usage Examples

### Python API

```python
from src.ml.inference import MLPFRPredictor

# Load most recent artifact in models/ (auto-discovery)
predictor = MLPFRPredictor(
    artifact_path='models',
    model_key='xgboost',     # or random_forest, gradient_boosting, adaboost
    mode='exit',             # or 'full_profile'
)

# Predict reactor exit conditions
result = predictor.predict_exit(
    initial_temperature_K=925.0,
    initial_pressure_Pa=200000.0,
    reactor_length_m=5.0,
    reactor_diameter_m=0.03,
    mass_flow_rate_kgps=0.07,
    heat_flux_Wm2=150000.0,
)

print(f"Temperature: {result['temperature_K']:.1f} K")
print(f"Pressure: {result['pressure_Pa']/1e5:.2f} bar")

# Predict complete axial profile
profile = predictor.predict_profile(
    initial_temperature_K=925.0,
    initial_pressure_Pa=200000.0,
    reactor_length_m=5.0,
    reactor_diameter_m=0.03,
    mass_flow_rate_kgps=0.07,
    heat_flux_Wm2=150000.0,
    n_points=200,
)

profile.to_csv('predictions.csv', index=False)

# Hot-swap to a different model from the same artifact
print(predictor.available_models())
predictor.switch_model('random_forest')
```

### Batch Processing

```python
# Generate training data for multiple reactants
from src.ml.data_generation import TrainingDataGenerator

generator = TrainingDataGenerator(output_dir='data/training')
dataset = generator.generate_dataset(
    reactants=['ethane', 'propane'],
    max_combinations_per_reactant=100
)
```

## File Structure

```
src/ml/
├── data_generation.py           # Training data generation script
├── model_training.py            # ML training script
├── inference.py                 # ML inference script
└── example_usage.py             # Usage examples

data/training/                  # Generated training data
├── training_data_complete_*.pkl  # Complete dataset (pickle format)
├── training_data_complete_*.csv  # Complete dataset (CSV format)
└── metadata_*.json              # Generation metadata

models/                         # Trained models
├── tree_models_exit.joblib              # From Main_4: models, scaler, splits, config (overwrite)
├── tree_model_tuned_exit_full.joblib    # From Main_5: tuned exit + optional full-profile bundle
├── simple_nn_exit_state_dict.pt         # From Main_6: PyTorch state_dict
├── simple_nn_exit_scalers.joblib        # From Main_6: X/y scalers + label encoder
├── simple_nn_exit_manifest.json         # From Main_6: arch / training / metrics / tuning / chemistry_groups
├── simple_nn_exit_per_target_metrics.csv
├── simple_nn_exit_group_metrics.csv
├── simple_nn_full_profile_state_dict.pt     # From Main_7 (optional)
├── simple_nn_full_profile_scalers.joblib
├── simple_nn_full_profile_manifest.json
├── simple_nn_full_profile_per_target_metrics.csv
├── simple_nn_full_profile_group_metrics.csv
└── training_summary.json        # (legacy, written by src/ml/model_training.py)
```

## Dependencies

### Required
- `scikit-learn>=1.0.0` - ML algorithms
- `joblib>=1.0.0` - Model serialization
- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation

### Optional (but recommended)
- `torch>=2.0.0` (optional, future) - PyTorch for neural-network path in `model_training.py`
- `xgboost>=1.5.0` - XGBoost

Install with:
```bash
pip install scikit-learn joblib xgboost
```

## Tips and Best Practices

1. **Training Data Size**: 
   - Start with 100-500 combinations per reactant
   - Increase for better accuracy (1000+ for production)
   - With limited data: use a smaller test_size (e.g. 0.15), enable hyperparameter tuning so max_depth/min_samples_leaf regularize trees, and add more runs when possible

2. **Model Selection**:
   - Neural networks: Best for large datasets, complex relationships
   - Random Forest: Good baseline, interpretable
   - XGBoost: Often best accuracy, fast training

3. **Target Types**:
   - Train separate models for primary/secondary/species
   - Primary targets are most important and accurate
   - Species models may need more training data

4. **Validation**:
   - Always validate ML predictions against Cantera
   - Check physical constraints (e.g., mass conservation)
   - Monitor for extrapolation (predictions outside training range)

## Troubleshooting

### Model Not Found
```
FileNotFoundError: Model not found: models/xgboost_primary.pkl
```
**Solution**: Train models first (`Main_4_train_and_evaluate_tree_models_IO.ipynb`, `Main_5_train_evaluate_tune_tree_model_evolution.ipynb`, `Main_6__train_evaluate_SimpleNN_IO.ipynb` for the PyTorch NN, or `python src/ml/model_training.py`). Ensure `model_type` in inference config matches an artifact model key you actually trained (e.g. `xgboost`, `random_forest`, `adaboost`).

### Out of Memory
**Solution**: Reduce `max_combinations` or use random sampling

### Poor Accuracy
**Solutions**:
- Increase training data size
- Expand parameter ranges
- Try different model types
- Check for data quality issues

## Future Enhancements

- [ ] Ensemble models (combine multiple models)
- [ ] Uncertainty quantification
- [ ] Online learning (update models with new data)
- [ ] Transfer learning (adapt models to new reactants)
- [ ] Physics-informed neural networks
- [ ] GPU acceleration for neural networks

## Citation

If you use the ML Surrogate Models in your research, please cite:

```
HydrAI: ML Surrogate Models for Plug Flow Reactor Simulation
Nikolas Karefyllidis, PhD
2025
```

## License

Same as main HydrAI project (MIT License).
