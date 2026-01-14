# ML Surrogate Models: Quick Start Guide

Get started with ML surrogate models in 3 simple steps!

## Prerequisites

Install ML dependencies:
```bash
pip install scikit-learn joblib
# Optional but recommended:
pip install tensorflow xgboost
```

## Step 1: Generate Training Data (5-30 minutes)

Generate training data from Cantera simulations:

```bash
# Quick start: 50 combinations per reactant (faster)
python src/ml/data_generation.py \
    --reactants ethane \
    --max-combinations 50 \
    --output-dir data/training

# Full dataset: 100+ combinations per reactant (better accuracy)
python src/ml/data_generation.py \
    --reactants ethane propane \
    --max-combinations 100 \
    --output-dir data/training
```

**What this does:**
- Runs multiple Cantera simulations with varied parameters
- Collects input features and output targets
- Saves to CSV files in `data/training/`

**Expected output:**
- `training_data_complete_YYYYMMDD_HHMMSS.csv` - Complete dataset
- `metadata_YYYYMMDD_HHMMSS.json` - Generation metadata

## Step 2: Train ML Models (2-10 minutes)

Train ML models on the generated data:

```bash
# Train all models on primary targets
python src/ml/model_training.py configs/ml_training_config.json
    --target-types primary \
    --models all \
```

**What this does:**
- Trains multiple ML algorithms (Neural Network, Random Forest, XGBoost, etc.)
- Evaluates model performance
- Saves trained models to `models/`

**Expected output:**
- `neural_network_primary.h5` - Neural network model
- `neural_network_primary_scalers.pkl` - Preprocessing scalers
- `random_forest_primary.pkl` - Random forest model
- `training_summary.json` - Training metrics

## Step 3: Use ML Models (Instant!)

Use trained models for fast predictions:

```bash
# Predict reactor profile
python src/ml/inference.py \
    --model-type neural_network \
    --target-type primary \
    --temperature 925.0 \
    --pressure 2.0 \
    --length 5.0 \
    --diameter 30.0 \
    --mass-flow 0.07 \
    --heat-flux 150000.0 \
    --n-points 200 \
    --output outputs/predictions.csv
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
    model_type='neural_network',
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
**Solution**: Train models first (Step 2)

### "Out of memory" during training
**Solution**: Reduce `--max-combinations` in Step 1

### Poor prediction accuracy
**Solutions**:
- Increase training data size (`--max-combinations`)
- Try different model types (`--models`)
- Check training data quality

## Next Steps

1. **Expand training data**: More combinations = better accuracy
2. **Train multiple target types**: `--target-types primary secondary`
3. **Compare models**: Try different `--model-type` options
4. **Validate predictions**: Compare ML predictions with Cantera

## Performance Tips

- **Fast predictions**: Use `neural_network` or `xgboost`
- **Interpretable models**: Use `random_forest`
- **Large datasets**: Neural networks scale best
- **Small datasets**: Random Forest or XGBoost often better

## Full Workflow

```bash
# 1. Generate data (start with small dataset for testing)
python src/ml/data_generation.py \
    --reactants ethane \
    --max-combinations 50

# 2. Train models
python src/ml/model_training.py \
    --data data/training/training_data_complete_*.csv \
    --models all

# 3. Test predictions
python src/ml/inference.py \
    --model-type neural_network \
    --temperature 925.0 \
    --pressure 2.0

# 4. Run examples
python src/ml/example_usage.py
```

That's it! You now have ML models that can replace Cantera with 100-1000x speedup!
