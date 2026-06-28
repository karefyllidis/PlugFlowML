# ML Surrogate Models: Quick Start Guide

Get started with ML surrogate models in 3 simple steps!

## Prerequisites

Install ML dependencies:
```bash
pip install scikit-learn joblib
# Optional but recommended:
pip install xgboost
```

## Step 1: Generate Training Data (5-30 minutes)

Generate training data from Cantera simulations:

```bash
# Using JSON config file (recommended)
python src/ml/data_generation.py configs/ml/ml_data_generation_config.json
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/Main_2_generate_training_data.ipynb
```

**What this does:**
- Runs multiple Cantera simulations with varied parameters
- **Sampling**: LHS (`sampling_method: "latin"`), random (`"random"`), or structured grid (`"full_grid"` / `"structured_grid"` / `"grid"`) via config
- Collects input features and output targets
- Saves partial data periodically as pickle files (when saving is enabled)
- Automatically cleans up partial files after completion (when saving to disk)
- Shows real-time progress with success rate and ETA
- In the notebook: run control flags (`IF_SHOW_PLOTS`, `IF_SAVE_PLOTS`, `IF_SAVE_METADATA`, `IF_SAVE_TRAINING_DATA`) and training-space plots (Step 2.1 preview, Step 4.1 from data)

**Expected output:**
- `training_data_complete_YYYYMMDD_HHMMSS.pkl` - Complete dataset (primary format; pickle for fast loading)
- Optional: `metadata_YYYYMMDD_HHMMSS.json` - Generation metadata (when saving enabled)

## Step 2: Explore and Engineer Features (Optional)

Explore the generated data and perform feature engineering:

```bash
jupyter notebook notebooks/Main_3_data_exploration_feature_engineering.ipynb
```

This notebook is for:
- Data exploration and visualization
- **Organized column categories** (inlet conditions, reactor design, operating conditions, state variables, thermodynamic properties, species Y/X) for ML-ready feature/target separation
- Feature engineering
- Data quality checks
- Statistical analysis

## Step 3: Train ML Models (2-10 minutes)

**Option A – Tree models (Jupyter notebook, recommended):**
```bash
jupyter notebook notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb
```
Runs the default-parameter inlet→outlet baseline evaluation for Random Forest, Gradient Boosting, XGBoost, and AdaBoost.

For hyperparameter tuning and full axial/PFR evolution with one selected model:
```bash
jupyter notebook notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb
```

For the PyTorch MLP exit-plane baseline (multi-output regression on inlet-only features):
```bash
jupyter notebook notebooks/Main_6_train_evaluate_SimpleNN_IO.ipynb
```
Hyperparameters are read from `configs/ml/ml_training_config.json` → `neural_network.{epochs, batch_size, learning_rate, h1, h2, h3, dropout}`. To run an in-notebook Optuna TPE search, flip `IF_HYPERPARAM_TUNING = True` in Section 2; the search space and trial budget are controlled by `neural_network.tuning.{n_trials, epochs_per_trial, validation_fraction, timeout_seconds}` (`pip install optuna` required). The main training loop (Section 8) also applies **`ReduceLROnPlateau`** on test R² checkpoints, **early stopping** when test R² stalls, and **reloads the best test-R² weights** before final metrics and export. While Main_6 or Main_7 runs, use `scripts/monitor/monitor_nn_training_progress.py` in another terminal (logs in `data/logs/`): set `MAIN_6` or `MAIN_7` and `LIVE=True` for live refresh (monitor auto-picks Optuna JSON vs training CSV in `data/logs/`; see `docs/ML_CONFIG_GUIDE.md`).

**Option B – All model types (command-line):**
```bash
python src/ml/model_training.py configs/ml/ml_training_config.json
```

**What this does:**
- Trains multiple ML algorithms (Main_4 notebook: RF, GB, XGBoost, AdaBoost baselines; Main_5 notebook: one tuned tree model; script: RF/GB/XGBoost/AdaBoost)
- Evaluates model performance (R², MSE, MAE, Normalized MAE diagnostics)
- Reports ML inference speed; set measured Cantera runtime variables in the notebook to print speedup factors
- Saves trained models to `models/`

**Expected output (notebook):**
- Main_4: `tree_models_exit.joblib` (models, scaler, label encoder, feature/target lists, config).
- Main_5: `tree_model_tuned_exit_full.joblib` (tuned exit model and, when trained, full-profile bundle).
- Main_6 (PyTorch): `simple_nn_exit_state_dict.pt`, `simple_nn_exit_scalers.joblib`, and `simple_nn_exit_manifest.json` (three-hidden-layer architecture `h1`–`h3`, training settings including best-checkpoint / early-stop fields, columns, headline + grouped metrics; with `IF_HYPERPARAM_TUNING=True` also Optuna PNGs and a compact `tuning` block in the manifest).

## Step 4: Use ML Models (Instant!)

Use trained models for fast predictions:

```bash
# Predict reactor profile using JSON config
python src/ml/inference.py configs/ml/ml_inference_config.json
```

**What this does:**
- Loads trained ML model
- Predicts reactor profile in milliseconds (vs. seconds/minutes with Cantera)
- Saves predictions to CSV

## Python API Example

```python
from src.ml.inference import MLPFRPredictor

# Load predictor
predictor = MLPFRPredictor(
    model_dir='models',
    model_type='xgboost',
    target_type='primary'
)

# Predict single point
result = predictor.predict_single_point(
    initial_temperature_K=925.0,
    initial_pressure_Pa=200000.0,
    reactor_length_m=5.0,
    reactor_diameter_m=0.03,
    mass_flow_rate_kgps=0.07,
    heat_flux_Wm2=150000.0,
    z_position_m=2.5
)

print(f"Temperature: {result['temperature_K']:.1f} K")
print(f"Pressure: {result['pressure_Pa']/1e5:.2f} bar")

# Predict complete profile
profile = predictor.predict_profile(
    initial_temperature_K=925.0,
    initial_pressure_Pa=200000.0,
    reactor_length_m=5.0,
    reactor_diameter_m=0.03,
    mass_flow_rate_kgps=0.07,
    heat_flux_Wm2=150000.0,
    n_points=200
)

profile.to_csv('predictions.csv', index=False)
```

## Run Examples

Try the example script:

```bash
python src/ml/example_usage.py
```

## Troubleshooting

### "Model not found" error
**Solution**: Train models first (Step 3); run `Main_4_train_and_evaluate_tree_models_IO.ipynb` for baseline tree models, `Main_5_train_evaluate_tune_tree_model_evolution.ipynb` for tuned tree models, `Main_6_train_evaluate_SimpleNN_IO.ipynb` for the PyTorch MLP baseline, or `python src/ml/model_training.py configs/ml/ml_training_config.json` for the CLI tree trainer.

### "Out of memory" during training
**Solution**: Reduce `max_combinations_per_reactant` in config file

### Poor prediction accuracy
**Solutions**:
- Increase training data size (`max_combinations_per_reactant` in config)
- Try different model types (edit config file)
- Check training data quality (use data exploration notebook)

## Next Steps

1. **Expand training data**: More combinations = better accuracy (edit config file)
2. **Train multiple target types**: Edit config to include `secondary` targets
3. **Compare models**: Try different model types in config
4. **Explore data**: Use the data exploration notebook to understand your dataset
5. **Validate predictions**: Compare ML predictions with Cantera

## Performance Tips

- **Fast predictions**: **XGBoost** or **Random Forest** (tree surrogates)
- **Interpretable models**: Use `random_forest`
- **Large datasets**: XGBoost / gradient boosting; the **PyTorch** MLP baseline (`Main_6`) scales linearly and runs on CPU/CUDA/MPS
- **Small datasets**: Random Forest or XGBoost often better than the NN baseline; raise `dropout` if the NN overfits

## Full Workflow

```bash
# 1. Generate data (edit configs/ml/ml_data_generation_config.json first)
python src/ml/data_generation.py configs/ml/ml_data_generation_config.json

# 2. (Optional) Explore data
jupyter notebook notebooks/Main_3_data_exploration_feature_engineering.ipynb

# 3. Baseline tree evaluation
jupyter notebook notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb

# 4. Optional one-model tuning + full PFR evolution
jupyter notebook notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb

# 4b. PyTorch MLP baseline (inlet -> outlet); reads neural_network.* from ml_training_config.json
#     Flip IF_HYPERPARAM_TUNING=True for an optional Optuna TPE search (neural_network.tuning).
#     Section 8: LR-on-plateau (test R²), early stopping, best-checkpoint restore before export.
jupyter notebook notebooks/Main_6_train_evaluate_SimpleNN_IO.ipynb

# Alternative script route (trees only; NN remains placeholder):
# Or: python src/ml/model_training.py configs/ml/ml_training_config.json

# 5. Test predictions
python src/ml/inference.py configs/ml/ml_inference_config.json

# 6. Run examples
python src/ml/example_usage.py
```

That's it! You now have ML models that can replace Cantera with 100-1000x speedup!
