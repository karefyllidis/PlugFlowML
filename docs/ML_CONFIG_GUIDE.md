# ML Configuration Guide

All ML scripts now use JSON configuration files instead of command-line arguments.

## Configuration Files

### 1. Training Data Generation

**File**: `configs/ml_data_generation_config.json`

```json
{
    "_comment": "Configuration for ML training data generation. See docs/ML_CONFIG_GUIDE.md for detailed documentation.",
    
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

- **`reactants`** (list of strings): List of reactants to generate data for. Available options: `"ethane"`, `"propane"`, `"naphtha"`, `"n-hexane"`. Each reactant will generate up to `max_combinations_per_reactant` simulations. If not specified or empty, uses all available reactants.

- **`max_combinations_per_reactant`** (integer): Maximum number of parameter combinations to simulate per reactant. Total simulations = `reactants.length × max_combinations_per_reactant`. Higher values = more training data but longer generation time. Recommended: 50-200 for initial testing, 500-1000+ for production datasets.

- **`output_dir`** (string): Directory where training data files will be saved. Final dataset is saved as both pickle (`.pkl`) and CSV (`.csv`) formats with timestamps: `training_data_complete_YYYYMMDD_HHMMSS.pkl` and `training_data_complete_YYYYMMDD_HHMMSS.csv`. Partial saves during generation use pickle format: `training_data_partial_YYYYMMDD_HHMMSS.pkl`. Partial files are automatically cleaned up after successful completion.

- **`save_interval`** (integer): Save progress every N simulations. Prevents data loss if generation is interrupted. Partial saves are stored as pickle files for efficiency. Set to `1` to save after every simulation (slower but safer). Recommended: 10-50 for long runs. Partial files are automatically deleted after successful completion to save disk space.

- **`n_jobs`** (integer): Number of parallel workers for data generation. 
  - `1`: Sequential execution (default, safer for debugging)
  - `2-8`: Use specific number of CPU cores
  - `-1`: Use all available CPU cores (recommended for large datasets)
  
  **Note**: Parallel execution significantly speeds up data generation but uses more memory. Each worker runs a separate Cantera simulation, so ensure you have enough RAM. Recommended: Use `-1` for production runs, `1` for testing.

- **`random_sample`** (boolean): 
  - `true`: Randomly sample from parameter space (recommended for large parameter spaces)
  - `false`: Use full grid search (exhaustive - can be very large). 
  
  **Example**: With 6 parameters each having 10 values, full grid = 10⁶ = 1,000,000 combinations. Random sampling with `max_combinations_per_reactant=100` generates only 100 combinations.

- **`random_sample_bounds`** (object, optional): Constrain random sampling to a subset of the parameter space. If provided, only parameter combinations within these bounds will be considered for random sampling.
  
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
  
  **Note**: Bounds must be within the `parameter_ranges` for each parameter. If a parameter is not listed in `random_sample_bounds`, it uses the full range from `parameter_ranges`.

- **`parameter_ranges`** (object): Define the parameter space for simulations. Each parameter uses format: `[min_value, max_value, n_points]`. The script creates `n_points` evenly spaced values between `min` and `max` using `np.linspace(min, max, n_points)`.
  
  - **`temperature_K`**: Reactor inlet temperature range. Format: `[min_K, max_K, n_points]`. Typical steam cracking temperatures: 800-1200 K. Example: `[800, 1200, 10]` creates 10 points: 800, 844, 889, ..., 1200 K.
  
  - **`pressure_bar`**: Reactor pressure range. Format: `[min_bar, max_bar, n_points]`. Note: internally converted to Pa (multiply by 1e5). Example: `[1.5, 3.0, 8]` creates 8 points between 1.5-3.0 bar.
  
  - **`length_m`**: Reactor length range. Format: `[min_m, max_m, n_points]`. Longer reactors = more residence time = higher conversion. Example: `[3.0, 7.0, 6]` creates 6 points between 3-7 meters.
  
  - **`diameter_mm`**: Reactor diameter range. Format: `[min_mm, max_mm, n_points]`. Note: internally converted to meters (divide by 1000). Example: `[20.0, 40.0, 5]` creates 5 points between 20-40 mm.
  
  - **`mass_flow_rate_kgps`**: Feed mass flow rate range. Format: `[min_kgps, max_kgps, n_points]`. Higher flow = shorter residence time. Example: `[0.05, 0.10, 6]` creates 6 points between 0.05-0.10 kg/s.
  
  - **`heat_flux_Wm2`**: External heat flux range. Format: `[min_Wm2, max_Wm2, n_points]`. Higher heat flux = faster reaction rates. Example: `[100000, 200000, 5]` creates 5 points between 100-200 kW/m².

**Total Combinations Calculation:**

If `random_sample=false`, total combinations per reactant = product of all `n_points`:
- Example: `10 × 8 × 6 × 5 × 6 × 5 = 72,000` combinations per reactant
- With 2 reactants: `72,000 × 2 = 144,000` total simulations

If `random_sample=true`, only `max_combinations_per_reactant` combinations are generated per reactant, regardless of the parameter space size.

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
