# HydrAI: Generalized Plug Flow Reactor Simulation System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Cantera](https://img.shields.io/badge/Cantera-3.1.0%2B-green.svg)](https://cantera.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow.svg)](https://github.com/karefyllidis/HydrAI)

*A comprehensive simulation framework for modeling steam cracking reactions in plug flow reactors using Cantera*

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Reactant Support** | Ethane, propane, naphtha, and n-hexane with dedicated mechanisms |
| **Automatic Configuration** | Dynamic configuration generation for each reactant type |
| **Species Name Handling** | Intelligent handling of different naming conventions across mechanisms |
| **Professional Output** | Systematic file naming and comprehensive result export |
| **Rich Visualizations** | 12+ customizable plots and data export formats |
| **Extensible Design** | Easy addition of new reactants and mechanisms |
| **Robust Error Handling** | Comprehensive error handling with informative messages |
| **Convenience Scripts** | Easy-to-use shell scripts for streamlined operation |

## Table of Contents

- [Quick Start](#quick-start)
- [Available Reactants](#available-reactants)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
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

### Method 2: Direct Python Execution
```bash
# Activate external virtual environment first
source /Users/nikolaskarefyllidis/ct-env/bin/activate

# List available reactants
python Main_GeneralizedPFR.py --list

# Run simulations
python Main_GeneralizedPFR.py --reactant ethane
python Main_GeneralizedPFR.py --reactant propane
python Main_GeneralizedPFR.py --reactant naphtha
python Main_GeneralizedPFR.py --reactant n-hexane
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
| **ethane** | C₂H₆ | `Ethane_Kinetic-Model_species_35.yaml` | 35 | 135 |
| **propane** | C₃H₈ | `Propane_Kinetic-Model_species_53.yaml` | 53 | 325 |
| **naphtha** | Mixed C₅-C₁₂ | `Naphtha_Kinetic-Model_species_1951.yaml` | 1,951 | 82,557 |
| **n-hexane** | C₆H₁₄ | `n-Hexane_Kinetic-Model_species_153.yaml` | 153 | 2,146 |

> **Performance Note:** The naphtha mechanism is significantly larger and will take longer to simulate due to its complexity (1,951 species, 82,557 reactions).

---

## Installation

### Prerequisites

- **Python** 3.8 or higher
- **Cantera** 3.1.0 or higher
- **Required Python packages** (see requirements.txt)

> **Note:** This project is configured to use an external virtual environment located at `/Users/nikolaskarefyllidis/ct-env/`. The convenience script `run_simulation.sh` automatically uses this environment, eliminating the need for local environment setup.

### Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/karefyllidis/HydrAI.git
cd HydrAI
```

#### 2. Install Dependencies
```bash
# Option 1: Use existing external virtual environment (recommended)
source /Users/nikolaskarefyllidis/ct-env/bin/activate
pip install -r requirements.txt

# Option 2: Create new virtual environment
python -m venv ct-env
source ct-env/bin/activate
pip install -r requirements.txt
```

#### 3. Verify Installation
```bash
# Using convenience script (recommended)
./run_simulation.sh --list

# Or direct execution
source /Users/nikolaskarefyllidis/ct-env/bin/activate && python Main_GeneralizedPFR.py --list
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
# Method 1: Using convenience script (recommended)
./run_simulation.sh ethane

# Method 2: Direct execution
source /Users/nikolaskarefyllidis/ct-env/bin/activate && python Main_GeneralizedPFR.py --reactant ethane

# Expected output:
# - Temperature and pressure profiles
# - Species concentration profiles
# - Conversion and yield calculations
# - CSV data export
# - Summary report
```

### Batch Processing
```bash
# Using convenience script
for reactant in ethane propane n-hexane; do
    ./run_simulation.sh $reactant
done

# Or direct execution
source /Users/nikolaskarefyllidis/ct-env/bin/activate
for reactant in ethane propane n-hexane; do
    python Main_GeneralizedPFR.py --reactant $reactant
done
```

### Project Management
```bash
# View clean project structure
./show_structure.sh

# List available reactants
./run_simulation.sh --list

# Check project compatibility
cat DIRECTORY_STRUCTURE.md
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
from Main_GeneralizedPFR import load_reactant_database, generate_config_for_reactant

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

The system uses `reactant_database.json` to define all available reactants:

```json
{
    "reactants": {
        "ethane": {
            "name": "Ethane",
            "formula": "C2H6",
            "mechanism_file": "mechanism/Ethane_Kinetic-Model_species_35.yaml",
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

- **Temperature**: Initial reactor temperature (K)
- **Pressure**: Initial reactor pressure (Pa) - **absolute pressure**
- **Reactor Length**: Total reactor length (m)
- **Reactor Diameter**: Reactor diameter (m) - **3 cm (0.03 m)**
- **Mass Flow Rate**: Feed mass flow rate (kg/s)
- **Number of Steps**: Simulation resolution (200 steps)
- **Step Size**: **Automatically calculated** from reactor length ÷ number of steps (0.025 m)
- **Heat Flux Profile**: External heating profile (47,516 W/m² for 900°C wall)
- **Reactor Volume**: **Automatically calculated** from geometry (area × step_size)
- **Wall Surface Area**: **Automatically calculated** from geometry (perimeter × step_size)

## Output Files

### Generated Files Structure
```
results/
├── results_[Reactant]_T[Temp]K_P[Press]bar_L[Length]m_D[Diam]mm_M[MassFlow]kgps_n[Steps].csv
└── summary_[Reactant]_T[Temp]K_P[Press]bar_L[Length]m_D[Diam]mm_M[MassFlow]kgps_n[Steps].dat

fig/
├── temperature_pressure_profiles.png
└── species_profiles.png

heat_flux_profile.json
└── JSON heat flux profile for pyrolysis simulations
```

### CSV Data Export
The CSV files contain:
- Axial position (z)
- Temperature and pressure profiles
- Velocity and density
- Heat flux profile
- All species mass and mole fractions

### Summary Reports
The DAT files contain:
- Simulation metadata
- Key performance indicators
- Conversion and yield calculations
- File references

### Heat Flux Profile
The JSON heat flux profile (`heat_flux_profile.json`) contains:
- **12 data points** distributed from 0.0m to 5.0m reactor length
- **Realistic heat flux**: 47,516 W/m² (corresponds to 900°C wall temperature)
- Position-based heat flux data points (position in meters, heat flux in W/m²)
- Helpful comments for each heating zone
- Simple structure with essential data and documentation
- Linear interpolation between data points
- Optimized for steam cracking pyrolysis conditions

## Project Structure

```
HydrAI/
├── Main_GeneralizedPFR.py          # Main simulation script
├── reactant_database.json          # Reactant definitions and configurations
├── config_template.json            # Configuration template
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
├── CHANGELOG.md                    # Version history
├── README.md                       # This file
├── DIRECTORY_STRUCTURE.md          # Detailed structure analysis
├── examples/
│   └── basic_usage.py              # Usage examples
├── docs/
│   └── API_REFERENCE.md            # Detailed API documentation
├── mechanism/                      # Chemical kinetic mechanisms
│   ├── Ethane_Kinetic-Model_species_35.yaml
│   ├── Propane_Kinetic-Model_species_53.yaml
│   ├── Naphtha_Kinetic-Model_species_1951.yaml
│   └── n-Hexane_Kinetic-Model_species_153.yaml
├── results/                        # Simulation results (auto-generated)
│   ├── results_*.csv               # Detailed simulation data
│   └── summary_*.dat               # Simulation summaries
├── fig/                           # Generated plots (auto-generated)
│   ├── temperature_profile.png
│   ├── pressure_profile.png
│   ├── velocity_profile.png
│   ├── density_profile.png
│   ├── heat_flux_profile.png
│   ├── molecular_weight_profile.png
│   ├── reactant_conversion.png
│   ├── product_mass_fractions.png
│   ├── product_mole_fractions.png
│   ├── mass_flow_conservation.png
│   ├── residence_time.png
│   └── heat_capacity_ratio.png
├── heat_flux_profile.json         # JSON heat flux profile for pyrolysis
├── run_simulation.sh              # Convenience script for running simulations
└── show_structure.sh              # Script to display clean project structure
```

### Key Files
- **`run_simulation.sh`**: Convenience script that automatically activates the virtual environment and runs simulations
- **`show_structure.sh`**: Displays the clean project structure excluding generated files
- **`DIRECTORY_STRUCTURE.md`**: Comprehensive analysis of the project structure and compatibility

## Adding New Reactants

### Step 1: Add Mechanism File
Place your mechanism file in the `mechanism/` directory following the naming convention:
```
mechanism/[Reactant]_Kinetic-Model_species_[Count].yaml
```

**Note:** The `[Count]` should be the number of species in your mechanism (e.g., `Ethane_Kinetic-Model_species_35.yaml` for a mechanism with 35 species).

### Step 2: Update Database
Add your reactant to `reactant_database.json`:

```json
{
    "reactants": {
        "your-reactant": {
            "name": "Your Reactant Name",
            "formula": "Chemical Formula",
            "mechanism_file": "mechanism/YourReactant_Kinetic-Model_species_X.yaml",
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
python Main_GeneralizedPFR.py --reactant your-reactant
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
| **Virtual Environment Not Activated** | `ModuleNotFoundError: No module named 'scipy'` | Use convenience script: `./run_simulation.sh --list` or activate manually: `source /Users/nikolaskarefyllidis/ct-env/bin/activate` |
| **Species Not Found** | `Error: Unknown species 'C2H6'` | Check species names in mechanism file and update database accordingly |
| **Mechanism File Missing** | `Error: Could not load mechanism file` | Ensure mechanism file exists and path in database is correct |
| **Convergence Issues** | `Warning: Solver convergence warnings` | Normal for complex mechanisms; simulation continues and produces valid results |
| **Permission Denied** | `Permission denied: ./run_simulation.sh` | Make scripts executable: `chmod +x run_simulation.sh show_structure.sh` |

### Quick Fixes

```bash
# Fix permission issues
chmod +x run_simulation.sh show_structure.sh

# Activate virtual environment
source /Users/nikolaskarefyllidis/ct-env/bin/activate

# Verify installation
./run_simulation.sh --list

# Check project structure
./show_structure.sh
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
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 *.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:

- **Repository**: [GitHub Repository](https://github.com/karefyllidis/HydrAI)
- **Issues**: [GitHub Issues](https://github.com/karefyllidis/HydrAI/issues)
- **Documentation**: [API Reference](docs/API_REFERENCE.md)

## Acknowledgments

- **Cantera Team** for the excellent chemical kinetics library
- **RMG Team at MIT** for the [Reaction Mechanism Generator](https://reactionmechanismgenerator.github.io/RMG-Py/) used to generate the kinetic mechanisms
- **Chemical Engineering Community** for feedback and testing
- **Contributors** who have helped improve this system

---

**Version:** 2.1  
**Last Updated:** January 15, 2025  
**Maintainer:** Chemical Engineering Simulation Team

### Recent Updates (v2.1)
- **Added convenience scripts** (`run_simulation.sh`, `show_structure.sh`)
- **Improved virtual environment handling** with clear activation instructions
- **Enhanced project structure documentation** with clean visual presentation
- **Updated installation and usage instructions** with step-by-step guidance
- **Added comprehensive troubleshooting guide** with quick fixes

