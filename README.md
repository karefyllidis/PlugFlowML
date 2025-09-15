# Generalized Plug Flow Reactor (PFR) Simulation System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Cantera](https://img.shields.io/badge/Cantera-3.1.0%2B-green.svg)](https://cantera.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive simulation framework for modeling steam cracking reactions in plug flow reactors using Cantera. This system supports multiple feedstocks with automatic configuration generation, species name standardization, and comprehensive result analysis.

## Features

- **Multi-Reactant Support**: Ethane, propane, naphtha, and n-hexane
- **Automatic Configuration**: Dynamic configuration generation for each reactant
- **Species Name Handling**: Intelligent handling of different naming conventions across mechanisms
- **Professional Output**: Systematic file naming and comprehensive result export
- **Flexible Visualization**: Customizable plots and data export formats
- **Extensible Design**: Easy addition of new reactants and mechanisms
- **Error Handling**: Robust error handling with informative messages

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Reactants](#available-reactants)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Adding New Reactants](#adding-new-reactants)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- Cantera 3.1.0 or higher
- Required Python packages (see requirements.txt)

### Setup

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd naphtha-pyrolisi-pfr
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python Main_GeneralizedPFR.py --list
   ```

### Dependencies

```
cantera >= 3.1.0
numpy >= 1.20.0
matplotlib >= 3.5.0
pandas >= 1.3.0
scipy >= 1.7.0
```

## Quick Start

### List Available Reactants
```bash
python Main_GeneralizedPFR.py --list
```

### Run a Simulation
```bash
# Ethane cracking
python Main_GeneralizedPFR.py --reactant ethane

# Propane cracking
python Main_GeneralizedPFR.py --reactant propane

# Naphtha cracking
python Main_GeneralizedPFR.py --reactant naphtha

# n-Hexane cracking
python Main_GeneralizedPFR.py --reactant n-hexane
```

## Available Reactants

| Reactant | Formula | Mechanism File | Description |
|----------|---------|----------------|-------------|
| **ethane** | C₂H₆ | `Ethane_Kinetic-Model.yaml` | Ethane steam cracking for ethylene production |
| **propane** | C₃H₈ | `Propane_Kinetic-Model.yaml` | Propane steam cracking for olefins production |
| **naphtha** | Mixed C₅-C₁₂ | `Naphtha_Kinetic-Model.yaml` | Naphtha steam cracking for petrochemicals |
| **n-hexane** | C₆H₁₄ | `n-Hexane_Kinetic-Model.yaml` | n-Hexane steam cracking for olefins |

## Usage Examples

### Basic Simulation
```bash
# Run ethane cracking simulation
python Main_GeneralizedPFR.py --reactant ethane

# Expected output:
# - Temperature and pressure profiles
# - Species concentration profiles
# - Conversion and yield calculations
# - CSV data export
# - Summary report
```

### Batch Processing
```bash
# Run multiple simulations
for reactant in ethane propane n-hexane; do
    python Main_GeneralizedPFR.py --reactant $reactant
done
```

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
            "mechanism_file": "mechanism/Ethane_Kinetic-Model.yaml",
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
- **Pressure**: Initial reactor pressure (Pa)
- **Reactor Length**: Total reactor length (m)
- **Reactor Diameter**: Reactor diameter (m)
- **Mass Flow Rate**: Feed mass flow rate (kg/s)
- **Number of Steps**: Simulation resolution
- **Heat Flux Profile**: External heating profile

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
- Two-section heating profile: inlet heating (0.0m) and outlet cooling (10.0m)
- Position-based heat flux data points (position in meters, heat flux in W/m²)
- Helpful comments for each heating zone
- Simple structure with essential data and documentation
- Linear interpolation between data points
- Optimized for steam cracking pyrolysis conditions

## Project Structure

```
naphtha-pyrolisi-pfr/
├── Main_GeneralizedPFR.py          # Main simulation script
├── reactant_database.json          # Reactant definitions and configurations
├── config_template.json            # Configuration template
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
├── CHANGELOG.md                    # Version history
├── README.md                       # This file
├── examples/
│   └── basic_usage.py              # Usage examples
├── docs/
│   └── API_REFERENCE.md            # Detailed API documentation
├── mechanism/                      # Chemical kinetic mechanisms
│   ├── Ethane_Kinetic-Model.yaml
│   ├── Propane_Kinetic-Model.yaml
│   ├── Naphtha_Kinetic-Model.yaml
│   └── n-Hexane_Kinetic-Model.yaml
├── results/                        # Simulation results (generated)
│   ├── results_*.csv
│   └── summary_*.dat
├── fig/                           # Generated plots (generated)
│   ├── temperature_pressure_profiles.png
│   └── species_profiles.png
└── heat_flux_profile.json         # JSON heat flux profile for pyrolysis
```

## Adding New Reactants

### Step 1: Add Mechanism File
Place your mechanism file in the `mechanism/` directory following the naming convention:
```
mechanism/[Reactant]_Kinetic-Model.yaml
```

### Step 2: Update Database
Add your reactant to `reactant_database.json`:

```json
{
    "reactants": {
        "your-reactant": {
            "name": "Your Reactant Name",
            "formula": "Chemical Formula",
            "mechanism_file": "mechanism/YourReactant_Kinetic-Model.yaml",
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

#### 1. Species Not Found
```
Error: Unknown species 'C2H6'
```
**Solution:** Check species names in your mechanism file and update the database accordingly.

#### 2. Mechanism File Missing
```
Error: Could not load mechanism file
```
**Solution:** Ensure the mechanism file exists and the path in the database is correct.

#### 3. Convergence Issues
```
Warning: Solver convergence warnings
```
**Solution:** These are normal for complex mechanisms. The simulation will continue and produce valid results.

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

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Email**: your-email@domain.com

## Acknowledgments

- **Cantera Team** for the excellent chemical kinetics library
- **Chemical Engineering Community** for feedback and testing
- **Contributors** who have helped improve this system

---

**Version:** 2.0  
**Last Updated:** January 15, 2025  
**Maintainer:** Chemical Engineering Simulation Team

