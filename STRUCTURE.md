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
│   ├── config_template.json      # Configuration template
│   ├── reactant_database.json    # Reactant definitions
│   └── heat_flux_profile.json    # Heat flux profiles
│
├── mechanisms/                    # Chemical kinetic mechanisms
│   ├── Ethane_Kinetic-Model_species_35.yaml
│   ├── Propane_Kinetic-Model_species_53.yaml
│   ├── Naphtha_Kinetic-Model_species_1951.yaml
│   └── n-Hexane_Kinetic-Model_species_153.yaml
│
├── data/                         # Data directory
│   ├── training/                 # Training data (generated)
│   └── raw/                      # Raw simulation data
│
├── models/                       # Trained ML models (generated)
│   ├── neural_network_*.h5
│   ├── random_forest_*.pkl
│   └── training_summary.json
│
├── outputs/                      # Simulation outputs
│   ├── results/                  # CSV results and summaries
│   └── figures/                  # Generated plots
│
├── docs/                         # Documentation
│   ├── API_REFERENCE.md
│   ├── ML_CONFIG_GUIDE.md       # ML configuration guide
│   └── ml/                        # ML Surrogate Models documentation
│       ├── README.md
│       ├── QUICKSTART.md
│       └── IMPLEMENTATION_SUMMARY.md
│
├── examples/                     # Usage examples
│   └── basic_usage.py
│
├── scripts/                      # Utility scripts
│   ├── run_simulation.sh
│   └── show_structure.sh
│
├── styles/                       # Figure aesthetics
│   ├── figure_aesthetics.json   # Centralized styling config
│   └── README.md                 # Aesthetics documentation
│
├── run_pfr.ipynb                 # Main entry point - PFR simulations (Jupyter notebook)
├── generate_training_data.ipynb  # ML training data generation (Jupyter notebook)
├── train_ml_models.ipynb         # ML model training (Jupyter notebook - coming soon)
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
- **After**: All configs in `configs/` directory

### 3. Mechanisms
- **Before**: `mechanism/` directory
- **After**: `mechanisms/` directory (plural, more standard)

### 4. ML Surrogate Models
- **Before**: `phase_b/` directory with mixed files
- **After**: `src/ml/` with organized modules and `docs/ml/` for documentation

### 5. Outputs
- **Before**: `results/` and `fig/` at root
- **After**: `outputs/results/` and `outputs/figures/` organized under outputs

### 6. Data and Models
- **New**: `data/` directory for training data
- **New**: `models/` directory for trained ML models

## Usage

### Running Simulations

**Interactive Jupyter Notebook (Recommended):**
```bash
jupyter notebook run_pfr.ipynb
# Or: jupyter lab run_pfr.ipynb
```

The notebook provides an interactive interface where you can:
- Select reactants interactively
- See real-time simulation progress
- View inline visualizations
- Modify parameters easily

### ML Surrogate Models Workflow

**1. Generate training data (Jupyter Notebook):**
```bash
jupyter notebook generate_training_data.ipynb
```

The notebook provides:
- Interactive configuration
- Real-time progress tracking
- Comprehensive data visualization
- Data quality checks

**2. Train ML models (Jupyter Notebook - Coming Soon):**
```bash
jupyter notebook train_ml_models.ipynb
```

**For now, use command-line:**
```bash
python src/ml/model_training.py configs/ml_training_config.json
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
- Configs: `configs/`
- Mechanisms: `mechanisms/`
- Outputs: `outputs/results/` and `outputs/figures/`
- Training data: `data/training/`
- Models: `models/`

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
