# ML Configuration Guide

All ML scripts now use JSON configuration files instead of command-line arguments.

## Configuration Files

### 1. Training Data Generation

**File**: `configs/ml_data_generation_config.json`

```json
{
    "reactants": ["ethane", "propane"],
    "max_combinations_per_reactant": 100,
    "output_dir": "data/training",
    "save_interval": 10,
    "random_sample": true,
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
python src/ml/data_generation.py configs/ml_data_generation_config.json
```

**Parameters:**
- `reactants`: List of reactants to use (optional, uses all if not specified)
- `max_combinations_per_reactant`: Maximum parameter combinations per reactant
- `output_dir`: Directory to save training data
- `save_interval`: Save data every N simulations
- `random_sample`: Use random sampling (true) or all combinations (false)
- `parameter_ranges`: Parameter ranges as `[min, max, n_points]` arrays

### 2. ML Model Training

**File**: `configs/ml_training_config.json`

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
        "n_estimators": 100,
        "max_depth": 6
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "max_depth": 5
    }
}
```

**Usage:**
```bash
python src/ml/model_training.py configs/ml_training_config.json
```

**Parameters:**
- `data_file`: Path to training data CSV (supports glob patterns)
- `output_dir`: Directory to save trained models
- `target_types`: List of target types (`primary`, `secondary`, `species`, `all`)
- `models`: List of models to train (`neural_network`, `random_forest`, `xgboost`, `gradient_boosting`, `all`)
- `test_size`: Fraction of data for testing (0.0-1.0)
- `random_state`: Random seed for reproducibility
- Model-specific parameters: See individual model sections

### 3. ML Inference

**File**: `configs/ml_inference_config.json`

```json
{
    "model_dir": "models",
    "model_type": "neural_network",
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
python src/ml/inference.py configs/ml_inference_config.json
```

**Parameters:**
- `model_dir`: Directory containing trained models
- `model_type`: Type of model to use (`neural_network`, `random_forest`, `xgboost`, `gradient_boosting`)
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

1. Copy the template config file from `configs/`
2. Modify parameters as needed
3. Save with a descriptive name
4. Run the script with your config file

**Example:**
```bash
# Copy template
cp configs/ml_data_generation_config.json configs/my_training_config.json

# Edit my_training_config.json with your parameters

# Run with custom config
python src/ml/data_generation.py configs/my_training_config.json
```

## Benefits of JSON Configuration

1. **Reproducibility**: All parameters saved in one file
2. **Version Control**: Easy to track configuration changes
3. **Flexibility**: Easy to create multiple configurations
4. **Documentation**: Self-documenting with comments
5. **No Command-Line Limits**: No need to remember long argument lists
