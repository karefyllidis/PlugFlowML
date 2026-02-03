# ML Surrogate Models for PFR Simulation

This module implements machine learning models to replace Cantera simulations, providing **100-1000x faster** predictions while maintaining accuracy.

## Overview

The ML Surrogate Models module consists of three main components:

1. **Training Data Generation** - Generate massive datasets from Cantera simulations
2. **ML Model Training** - Train multiple ML algorithms (Neural Networks, Random Forest, XGBoost, etc.)
3. **ML Inference** - Use trained models for fast predictions

## Quick Start

### Step 1: Generate Training Data

Generate a large training dataset by running parameter sweeps:

```bash
# Generate training data using JSON config file
python src/ml/data_generation.py configs/ml_data_generation_config.json
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/Main_generate_training_data.ipynb
```

**Expected Output:**
- Training data files in `data/training/`:
  - `training_data_complete_YYYYMMDD_HHMMSS.pkl` - Complete dataset (pickle format, faster loading)
  - `training_data_complete_YYYYMMDD_HHMMSS.csv` - Complete dataset (CSV format, for compatibility)
  - `metadata_YYYYMMDD_HHMMSS.json` - Generation metadata
- Partial saves during generation: `training_data_partial_*.pkl` (automatically cleaned up after completion)

### Step 2: Train ML Models

Train multiple ML models on the generated data:

```bash
# Train all models on primary targets (temperature, pressure, velocity, density)
python src/ml/model_training.py configs/ml_training_config.json
```

**Available Models:**
- `neural_network` - Deep neural network (TensorFlow/Keras)
- `random_forest` - Random Forest (scikit-learn)
- `xgboost` - XGBoost gradient boosting
- `gradient_boosting` - Gradient Boosting (scikit-learn)

**Target Types:**
- `primary` - Core outputs (temperature, pressure, velocity, density)
- `secondary` - Thermodynamic properties (enthalpy, entropy, heat capacity, etc.)
- `species` - Species concentrations (mass/mole fractions)
- `all` - All targets combined

### Step 3: Use ML Models for Prediction

Use trained models instead of Cantera:

```bash
# Predict reactor profile using ML model
python src/ml/inference.py configs/ml_inference_config.json
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

The `train_ml_models.py` script:

1. **Data Preparation**:
   - Splits data into train/test sets
   - Scales features and targets
   - Handles categorical variables (reactant type)

2. **Model Training**:
   - Trains multiple algorithms
   - Uses early stopping for neural networks
   - Evaluates on test set

3. **Model Saving**:
   - Saves trained models
   - Saves scalers for preprocessing
   - Saves training metadata and metrics

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

## Usage Examples

### Python API

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

# Save results
profile.to_csv('predictions.csv', index=False)
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
├── neural_network_primary.h5
├── neural_network_primary_scalers.pkl
├── random_forest_primary.pkl
├── random_forest_primary_scalers.pkl
└── training_summary.json
```

## Dependencies

### Required
- `scikit-learn>=1.0.0` - ML algorithms
- `joblib>=1.0.0` - Model serialization
- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation

### Optional (but recommended)
- `tensorflow>=2.8.0` - Neural networks
- `xgboost>=1.5.0` - XGBoost

Install with:
```bash
pip install scikit-learn joblib tensorflow xgboost
```

## Tips and Best Practices

1. **Training Data Size**: 
   - Start with 100-500 combinations per reactant
   - Increase for better accuracy (1000+ for production)

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
FileNotFoundError: Model not found: models/neural_network_primary.h5
```
**Solution**: Train models first using `train_ml_models.py`

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
