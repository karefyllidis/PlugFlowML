# HydrAI: Generalized Plug Flow Reactor Simulation System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Cantera](https://img.shields.io/badge/Cantera-3.1.0%2B-green.svg)](https://cantera.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)](https://github.com/karefyllidis/HydrAI)

**HydrAI** = **Hydr**ocarbon + **AI** (Machine Learning)

*A comprehensive simulation framework for modeling steam cracking reactions in plug flow reactors using Cantera*

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Reactant Support** | Ethane, propane, naphtha, and n-hexane with dedicated mechanisms |
| **Automatic Configuration** | Dynamic configuration generation for each reactant type |
| **Species Name Handling** | Intelligent handling of different naming conventions across mechanisms |
| **Professional Output** | Systematic file naming and comprehensive result export (245+ columns) |
| **Visualizations** | 18+ customizable plots with centralized aesthetics configuration |
| **Export Controls** | Optional CSV and plot generation with flexible workflow options |
| **ML Surrogate Models** | Machine learning models for 100-1000x faster predictions |
| **Parallel Processing** | Multiprocessing support for fast training data generation (use all CPU cores) |
| **Centralized Styling** | Consistent figure aesthetics via JSON configuration |
| **JSON Configuration** | All ML workflows use JSON config files for reproducibility |
| **Extensible Design** | Easy addition of new reactants and mechanisms |
| **Robust Error Handling** | Comprehensive error handling with informative messages |
| **Convenience Scripts** | Easy-to-use shell scripts for streamlined operation |

## Table of Contents

- [Quick Start](#quick-start)
- [Available Reactants](#available-reactants)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [ML Surrogate Models](#ml-surrogate-models)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Figure Aesthetics](#figure-aesthetics)
- [Adding New Reactants](#adding-new-reactants)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

> **Get started in 3 simple steps!**

### Method 1: Using Convenience Script (Recommended)
```bash
# List available reactants
./run_simulation.sh --list

# Run simulations
./run_simulation.sh ethane
./run_simulation.sh propane
./run_simulation.sh naphtha
./run_simulation.sh n-hexane
```

### Method 2: Jupyter Notebook (Interactive)
```bash
# Launch Jupyter
jupyter notebook run_pfr.ipynb

# Or use JupyterLab
jupyter lab run_pfr.ipynb
```

### Method 3: Using Convenience Script
```bash
# List available reactants
./scripts/run_simulation.sh --list

# Run simulations
./scripts/run_simulation.sh ethane
./scripts/run_simulation.sh propane
./scripts/run_simulation.sh naphtha
./scripts/run_simulation.sh n-hexane
```

### View Project Structure
```bash
# Show clean project structure (excluding generated files)
./show_structure.sh
```

---

## Available Reactants

| Reactant | Formula | Mechanism File | Species | Reactions |
|----------|---------|----------------|---------|-----------|
| **ethane** | C₂H₆ | `mechanisms/Ethane_Kinetic-Model_species_35.yaml` | 35 | 135 |
| **propane** | C₃H₈ | `mechanisms/Propane_Kinetic-Model_species_53.yaml` | 53 | 325 |
| **naphtha** | Mixed C₅-C₁₂ | `mechanisms/Naphtha_Kinetic-Model_species_1951.yaml` | 1,951 | 82,557 |
| **n-hexane** | C₆H₁₄ | `mechanisms/n-Hexane_Kinetic-Model_species_153.yaml` | 153 | 2,146 |

> **Performance Note:** The naphtha mechanism is significantly larger and will take longer to simulate due to its complexity (1,951 species, 82,557 reactions).

---

## Installation

### Prerequisites

- **Python** 3.8 or higher
- **Cantera** 3.1.0 or higher (installed via pip)
- **Required Python packages** (see requirements.txt)

### Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/karefyllidis/HydrAI.git
cd HydrAI
```

#### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Ensure Cantera is installed
pip install cantera

# Install Jupyter for interactive notebooks
pip install jupyter jupyterlab
```

> **Note:** Mechanism YAML files are not tracked in git (they are in `.gitignore`). You need to provide your own mechanism files in the `mechanisms/` directory. See [Adding New Reactants](#adding-new-reactants) for details.

#### 3. Verify Installation
```bash
# Launch Jupyter notebook
jupyter notebook run_pfr.ipynb

# Or use convenience script
./scripts/run_simulation.sh
```

> **Success!** If you see the list of available reactants, your installation is complete!

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **cantera** | ≥ 3.1.0 | Chemical kinetics and thermodynamics |
| **numpy** | ≥ 1.20.0 | Numerical computing |
| **matplotlib** | ≥ 3.5.0 | Data visualization |
| **pandas** | ≥ 1.3.0 | Data manipulation and analysis |
| **scipy** | ≥ 1.7.0 | Scientific computing |

## Usage Examples

### Basic Simulation
```bash
# Method 1: Using Jupyter notebook (recommended)
jupyter notebook run_pfr.ipynb

# Method 2: Using convenience script
./scripts/run_simulation.sh
jupyter notebook run_pfr.ipynb

# Expected output:
# - Temperature and pressure profiles
# - Species concentration profiles
# - Conversion and yield calculations
# - Combined reactant consumption and product formation plot
# - CSV data export
# - Summary report
```

**Note:** The Jupyter notebook (`run_pfr.ipynb`) automatically handles import order correctly. If you're creating custom scripts, ensure you import `cantera` before adding `src` to `sys.path` to avoid namespace conflicts.

### Batch Processing
```bash
# Use Jupyter notebook for interactive batch processing
jupyter notebook run_pfr.ipynb

# In the notebook, you can change REACTANT_KEY and run cells multiple times
# for different reactants
```

### Project Management
```bash
# View clean project structure
./scripts/show_structure.sh

# List available reactants
./scripts/run_simulation.sh --list

# Or launch Jupyter notebook
jupyter notebook run_pfr.ipynb

# Check project structure
cat STRUCTURE.md
```

### Performance Notes:
- **Naphtha mechanism is significantly slower** due to its large size (1,951 species, 82,557 reactions)
- **Simulation time scales with mechanism complexity** - expect longer runtimes for larger mechanisms
- **Consider reducing step count** for faster naphtha simulations if needed

### Mechanism Generation:
All kinetic mechanisms in this system were generated using [**Reaction Mechanism Generator (RMG)**](https://reactionmechanismgenerator.github.io/RMG-Py/), an automatic chemical reaction mechanism generator developed by the RMG team at MIT. RMG constructs kinetic models composed of elementary chemical reaction steps using a general understanding of how molecules react.

**Naming Convention:** Mechanism files follow the pattern `[Reactant]_Kinetic-Model_species_[Count].yaml` where `[Count]` indicates the number of species in the mechanism (e.g., `Ethane_Kinetic-Model_species_35.yaml` contains 35 species).

### Custom Analysis
```python
from src.cantera.pfr_simulator import load_reactant_database, generate_config_for_reactant

# Load database
db = load_reactant_database()

# Generate custom configuration
config = generate_config_for_reactant('ethane', db)

# Access specific parameters
print(f"Feed composition: {config['initial_conditions']['composition']}")
print(f"Temperature: {config['initial_conditions']['temperature_K']} K")
```

## Configuration

### Reactant Database Structure

The system uses `configs/reactant_database.json` to define all available reactants:

```json
{
    "reactants": {
        "ethane": {
            "name": "Ethane",
            "formula": "C2H6",
            "mechanism_file": "mechanisms/Ethane_Kinetic-Model_species_35.yaml",
            "feed_species": "ethane(1)",
            "diluent": "Water",
            "diluent_ratio": 0.326,
            "target_products": ["C2H4(8)", "CH4(12)", "H2(19)", "C2H2(47)"],
            "product_names": ["Ethylene", "Methane", "Hydrogen", "Acetylene"],
            "description": "Ethane steam cracking for ethylene production"
        }
    }
}
```

### Simulation Parameters

Key parameters that can be adjusted:

- **Temperature**: Initial reactor temperature (925.15 K)
- **Pressure**: Initial reactor pressure (2.0 bar = 200,000 Pa) - **absolute pressure**
- **Reactor Length**: Total reactor length (5.0 m)
- **Reactor Diameter**: Reactor diameter (30.0 mm = 0.03 m)
- **Mass Flow Rate**: Feed mass flow rate (0.07 kg/s)
- **Number of Steps**: Simulation resolution (200 steps)
- **Step Size**: **Automatically calculated** from reactor length ÷ number of steps (0.025 m)
- **Heat Flux Profile**: External heating profile (150,000 W/m² constant)
- **Wall Roughness**: Surface roughness (0.0 m - smooth wall)
- **Reactor Volume**: **Automatically calculated** from geometry (area × step_size)
- **Wall Surface Area**: **Automatically calculated** from geometry (perimeter × step_size)
- **Export Controls**: Optional CSV and plot generation control
  - `if_csv_out`: 1 = enable CSV export, 0 = disable (default: 1)
  - `if_plot_out`: 1 = enable plot generation, 0 = disable (default: 1)

## Output Files

### Generated Files Structure
```
outputs/
├── results/
│   ├── results_[Reactant]_T[Temp]K_P[Press]bar_L[Length]m_D[Diam]mm_M[MassFlow]kgps_n[Steps].csv
│   └── summary_[Reactant]_T[Temp]K_P[Press]bar_L[Length]m_D[Diam]mm_M[MassFlow]kgps_n[Steps].dat
└── figures/
    ├── temperature_profile.png
    ├── pressure_profile.png
    ├── velocity_profile.png
    ├── density_profile.png
    ├── heat_flux_profile.png
    ├── heat_flux_vs_relative_position.png
    ├── molecular_weight_profile.png
    ├── heat_capacity_cp.png
    ├── heat_capacity_cv.png
    ├── heat_capacity_ratio.png
    ├── enthalpy_profile.png
    ├── entropy_profile.png
    ├── viscosity_profile.png
    ├── thermal_conductivity_profile.png
    ├── residence_time.png
    ├── reactant_conversion.png
    ├── product_mass_fractions.png
    └── product_mole_fractions.png
```

### CSV Data Export
The CSV files contain comprehensive data from Cantera (245 columns total):

**Basic Properties (7):**
- Axial position (z)
- Temperature and pressure profiles
- Velocity and density
- Heat flux profile

**Thermodynamic Properties (8):**
- Heat capacity (Cp, Cv) and heat capacity ratio
- Enthalpy, entropy, internal energy, Gibbs free energy
- Mean molecular weight

**Transport Properties (2):**
- Dynamic viscosity
- Thermal conductivity

**Composition Data (228):**
- Mass and mole fractions for all 114 species (Y_species, X_species)

**Note:** Advanced reaction kinetics and transport properties are not available in this Cantera version but the core simulation data is fully exported.

### Summary Reports
The DAT files contain:
- Simulation metadata
- Key performance indicators
- Conversion and yield calculations
- File references

### Heat Flux Profile
The JSON heat flux profile (`heat_flux_profile.json`) contains:
- **6 data points** distributed from 0.0 to 1.0 (relative positions)
- **Constant heat flux**: 150,000 W/m² (high-temperature pyrolysis conditions)
- **Relative position format**: 0.0 = inlet, 1.0 = outlet (automatically scaled to reactor length)
- **Step interpolation method**: Heat flux remains constant between data points
- Helpful comments for each heating zone
- Simple structure with essential data and documentation
- Optimized for steam cracking pyrolysis conditions

**Example heat flux profile configuration:**
```json
{
  "heat_flux_profile": {
    "interpolation_method": "step",
    "data_points": [
      {"position": 0.0, "heat_flux": 150000, "_comment": "Inlet region"},
      {"position": 0.2, "heat_flux": 150000, "_comment": "Position 20% of reactor length"},
      {"position": 0.4, "heat_flux": 150000, "_comment": "Position 40% of reactor length"},
      {"position": 0.6, "heat_flux": 150000, "_comment": "Position 60% of reactor length"},
      {"position": 0.8, "heat_flux": 150000, "_comment": "Position 80% of reactor length"},
      {"position": 1.0, "heat_flux": 150000, "_comment": "Outlet region"}
    ]
  }
}
```

### Export Controls

The system provides flexible export controls to optimize performance and storage:

```json
{
  "export_controls": {
    "if_csv_out": 1,
    "if_plot_out": 1,
    "_comment": "Export controls: 1 = enable, 0 = disable"
  }
}
```

**Options:**
- **`if_csv_out`**: Controls CSV data export (245+ columns)
  - `1` (default): Export comprehensive CSV data
  - `0`: Skip CSV export (simulation only)
- **`if_plot_out`**: Controls plot generation (18+ figures)
  - `1` (default): Generate all visualization plots
  - `0`: Skip plot generation (simulation only)

**Use Cases:**
- **Full Export** (`if_csv_out: 1, if_plot_out: 1`): Complete analysis with data and plots
- **Data Only** (`if_csv_out: 1, if_plot_out: 0`): Export data for external analysis
- **Plots Only** (`if_csv_out: 0, if_plot_out: 1`): Quick visualization without large CSV files
- **Simulation Only** (`if_csv_out: 0, if_plot_out: 0`): Fast simulation for parameter studies

## Project Structure

```
HydrAI/
├── src/                            # Source code
│   ├── cantera/                    # Cantera-based simulation
│   │   └── pfr_simulator.py        # Main PFR simulation code
│   ├── ml/                         # ML Surrogate Models
│   │   ├── data_generation.py     # Training data generation
│   │   ├── model_training.py      # ML model training
│   │   ├── inference.py           # ML inference
│   │   └── example_usage.py       # ML usage examples
│   └── utils/                      # Utility modules
│       └── plot_style.py          # Figure aesthetics utilities
├── configs/                        # Configuration files
│   ├── config_template.json       # Configuration template
│   ├── reactant_database.json     # Reactant definitions
│   ├── heat_flux_profile.json     # Heat flux profiles
│   ├── ml_data_generation_config.json
│   ├── ml_training_config.json
│   └── ml_inference_config.json
├── mechanisms/                     # Chemical kinetic mechanisms (not tracked in git)
│   ├── Ethane_Kinetic-Model_species_35.yaml
│   ├── Propane_Kinetic-Model_species_53.yaml
│   ├── Naphtha_Kinetic-Model_species_1951.yaml
│   └── n-Hexane_Kinetic-Model_species_153.yaml
│   └── .gitkeep                   # Ensures directory structure is tracked
├── data/                           # Data directory
│   ├── training/                   # Training data (generated, not tracked)
│   │   └── .gitkeep               # Ensures directory structure is tracked
│   └── raw/                        # Raw simulation data (not tracked)
│       └── .gitkeep               # Ensures directory structure is tracked
├── models/                         # Trained ML models (generated, not tracked)
│   └── .gitkeep                   # Ensures directory structure is tracked
├── outputs/                        # Simulation outputs (not tracked)
│   ├── results/                    # CSV results and summaries
│   │   └── .gitkeep               # Ensures directory structure is tracked
│   └── figures/                    # Generated plots
│       └── .gitkeep               # Ensures directory structure is tracked
├── styles/                         # Figure aesthetics
│   ├── figure_aesthetics.json     # Centralized styling config
│   └── README.md                  # Aesthetics documentation
├── docs/                           # Documentation
│   ├── API_REFERENCE.md
│   ├── ML_CONFIG_GUIDE.md        # ML configuration guide
│   └── ml/                          # ML Surrogate Models docs
│       ├── README.md
│       ├── QUICKSTART.md
│       └── IMPLEMENTATION_SUMMARY.md
├── examples/                       # Usage examples
│   └── basic_usage.py
├── scripts/                        # Utility scripts
│   ├── run_simulation.sh
│   └── show_structure.sh
├── run_pfr.ipynb                  # Main entry point - PFR simulations (Jupyter notebook)
├── generate_training_data.ipynb              # ML training data generation (Jupyter notebook)
├── data_exploration_feature_engineering.ipynb # Data exploration and feature engineering (Jupyter notebook)
├── train_ml_models.ipynb                     # ML model training (Jupyter notebook - coming soon)
├── requirements.txt
├── README.md                       # This file
├── LICENSE
├── CHANGELOG.md
└── STRUCTURE.md                    # Detailed structure documentation
```

### Key Files
- **`run_pfr.ipynb`**: Main interactive entry point for PFR simulations (Jupyter notebook)
- **`generate_training_data.ipynb`**: Interactive ML training data generation (Jupyter notebook)
- **`data_exploration_feature_engineering.ipynb`**: Data exploration and feature engineering (Jupyter notebook)
- **`train_ml_models.ipynb`**: Interactive ML model training (Jupyter notebook - coming soon)
- **`scripts/run_simulation.sh`**: Convenience script for command-line execution
- **`scripts/show_structure.sh`**: Displays the clean project structure excluding generated files
- **`STRUCTURE.md`**: Detailed documentation of the project structure
- **`DIRECTORY_STRUCTURE.md`**: Comprehensive analysis of the project structure and compatibility

## Adding New Reactants

### Step 1: Add Mechanism File
Place your mechanism file in the `mechanisms/` directory following the naming convention:
```
mechanisms/[Reactant]_Kinetic-Model_species_[Count].yaml
```

**Notes:**
- The `[Count]` should be the number of species in your mechanism (e.g., `Ethane_Kinetic-Model_species_35.yaml` for a mechanism with 35 species).
- Mechanism YAML files are **not tracked in git** (they are in `.gitignore`). This is intentional as they are large files that should remain local to your repository.

### Step 2: Update Database
Add your reactant to `configs/reactant_database.json`:

```json
{
    "reactants": {
        "your-reactant": {
            "name": "Your Reactant Name",
            "formula": "Chemical Formula",
            "mechanism_file": "mechanisms/YourReactant_Kinetic-Model_species_X.yaml",
            "feed_species": "SpeciesName",
            "diluent": "H2O",
            "diluent_ratio": 0.5,
            "target_products": ["C2H4", "CH4", "H2"],
            "product_names": ["Ethylene", "Methane", "Hydrogen"],
            "description": "Description of your process"
        }
    }
}
```

### Step 3: Test
```bash
# Launch Jupyter notebook
jupyter notebook run_pfr.ipynb

# Or use command-line script
./scripts/run_simulation.sh your-reactant
```

## API Documentation

### Core Functions

#### `load_reactant_database()`
Loads the reactant database containing all available feedstocks.

**Returns:** `dict` - Complete reactant database

#### `generate_config_for_reactant(reactant_key, database)`
Generates a complete simulation configuration for a specific reactant.

**Parameters:**
- `reactant_key` (str): Reactant identifier
- `database` (dict): Loaded reactant database

**Returns:** `dict` - Complete configuration dictionary

#### `setup_mechanism(config)`
Sets up the gas phase model with kinetics, thermodynamics, and transport.

**Parameters:**
- `config` (dict): Simulation configuration

**Returns:** `ct.Solution` - Cantera gas solution object

### Simulation Functions

#### `run_simulation(gas, config, reactant_info, ...)`
Runs the main simulation loop for the PFR.

**Parameters:**
- `gas` (ct.Solution): Gas phase model
- `config` (dict): Simulation configuration
- `reactant_info` (dict): Reactant-specific information
- Additional reactor and operating parameters

**Returns:** `ct.SolutionArray` - Simulation results

For detailed API documentation, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

## Troubleshooting

### Common Issues

| Issue | Error Message | Solution |
|-------|---------------|----------|
| **Missing Dependencies** | `ModuleNotFoundError: No module named 'scipy'` | Install dependencies: `pip install -r requirements.txt` |
| **Species Not Found** | `Error: Unknown species 'C2H6'` | Check species names in mechanism file and update database accordingly |
| **Mechanism File Missing** | `Error: Could not load mechanism file` | Ensure mechanism file exists and path in database is correct |
| **Convergence Issues** | `Warning: Solver convergence warnings` | Normal for complex mechanisms; simulation continues and produces valid results |
| **Permission Denied** | `Permission denied: ./run_simulation.sh` | Make scripts executable: `chmod +x run_simulation.sh show_structure.sh` |
| **Cantera Import Error** | `AttributeError: module 'cantera' has no attribute 'Solution'` | Import `cantera` before adding `src` to `sys.path` in custom scripts |
| **Species Access Error** | `IndexError: only integers, slices...` | Use `states1.Y[:, species_idx]` instead of string indexing for SolutionArray |

### Quick Fixes

```bash
# Fix permission issues
chmod +x scripts/run_simulation.sh scripts/show_structure.sh

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Verify installation - launch Jupyter notebook
jupyter notebook run_pfr.ipynb

# Or use convenience script
./scripts/run_simulation.sh --list

# Check project structure
./scripts/show_structure.sh
```

### Performance Optimization

#### For Faster Simulations:
```python
# Reduce number of steps for faster computation
config['simulation_settings']['n_steps'] = 100  # Instead of 200
```

#### For Better Accuracy:
```python
# Increase number of steps for higher resolution
config['simulation_settings']['n_steps'] = 500  # More detailed profiles
```

### Debug Mode
For detailed debugging information, modify the simulation parameters:
```python
# Reduce step size for better convergence
config['simulation_settings']['n_steps'] = 500
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add comprehensive tests**
4. **Update documentation**
5. **Submit a pull request**

### Development Setup
```bash
# Install all dependencies
pip install -r requirements.txt

# Install ML dependencies (optional, for ML Surrogate Models)
pip install scikit-learn joblib tensorflow xgboost

# Install Jupyter for interactive development
pip install jupyter jupyterlab

# Run tests (if available)
python -m pytest tests/

# Check code style
flake8 src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:

- **Repository**: [GitHub Repository](https://github.com/karefyllidis/HydrAI)
- **Issues**: [GitHub Issues](https://github.com/karefyllidis/HydrAI/issues)
- **Documentation**: 
  - [API Reference](docs/API_REFERENCE.md)
  - [ML Surrogate Models](docs/ml/README.md)
  - [ML Configuration Guide](docs/ML_CONFIG_GUIDE.md)
  - [Project Structure](STRUCTURE.md)

## Figure Aesthetics

All figure styling is centralized in `styles/figure_aesthetics.json`. This allows you to:

- Maintain consistent appearance across all plots
- Easily customize colors, fonts, line styles, and more
- Update styling in one place for all figures

**Quick Customization:**
1. Edit `styles/figure_aesthetics.json`
2. Run your simulation
3. All figures automatically use the new styling

See `styles/README.md` for detailed documentation.

## Acknowledgments

- **Cantera Team** for the excellent chemical kinetics library
- **RMG Team at MIT** for the [Reaction Mechanism Generator](https://reactionmechanismgenerator.github.io/RMG-Py/) used to generate the kinetic mechanisms
- **Chemical Engineering Community** for feedback and testing
- **Contributors** who have helped improve this system

---

**Version:** 3.0.0  
**Last Updated:** January 2025  
**Maintainer:** Nikolas Karefyllidis, PhD

### Recent Updates (v3.0.0)
- **Project restructuring** - Organized into `src/`, `configs/`, `data/`, `models/`, `outputs/` directories
- **ML Surrogate Models** - Complete ML framework for fast predictions (100-1000x speedup)
- **Parallel processing** - Multiprocessing support for training data generation (use all CPU cores for faster generation)
- **JSON configuration** - All ML workflows use JSON config files instead of command-line arguments
- **Centralized figure aesthetics** - Consistent styling via `styles/figure_aesthetics.json`
- **Improved code organization** - Modular package structure following Python best practices
- **Enhanced documentation** - Updated all docs to reflect new structure and features
- **Jupyter notebook improvements** - Fixed import order, added combined conversion/product plots
- **Better visualization** - Combined reactant consumption and product formation in single plot

