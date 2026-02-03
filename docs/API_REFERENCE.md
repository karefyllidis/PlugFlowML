# API Reference - Generalized PFR Simulation System

This document provides comprehensive API documentation for all functions, classes, and modules in the Generalized PFR Simulation System.

## Table of Contents

- [Core Modules](#core-modules)
- [Configuration Functions](#configuration-functions)
- [Simulation Functions](#simulation-functions)
- [Analysis Functions](#analysis-functions)
- [Utility Functions](#utility-functions)
- [Data Structures](#data-structures)
- [Error Handling](#error-handling)

## Core Modules

### src/cantera/pfr_simulator.py

The main simulation module containing all core PFR functionality. The interactive entry point is **`notebooks/Main_run_pfr.ipynb`**, which imports and uses this module.

#### Module-level Constants

```python
# Version information
__version__ = "2.1.0"
__author__ = "Nikolas Karefyllidis, PhD"
__date__ = "2025-09-20"
```

## Configuration Functions

### `load_reactant_database()`

Loads the reactant database containing all available feedstocks and their properties.

**Signature:**
```python
def load_reactant_database() -> dict
```

**Returns:**
- `dict`: Dictionary containing reactant information

**Structure:**
```python
{
    'export_controls': {
        'if_csv_out': int,         # 1 = enable CSV export, 0 = disable
        'if_plot_out': int,        # 1 = enable plot generation, 0 = disable
        '_comment': str            # Documentation comment
    },
    'reactants': {
        'reactant_key': {
            'name': str,           # Human-readable name
            'formula': str,        # Chemical formula
            'mechanism_file': str, # Path to mechanism file
            'feed_species': str,   # Species name in mechanism
            'diluent': str,        # Diluent species name
            'diluent_ratio': float,# Diluent to reactant ratio
            'target_products': list, # List of product species
            'product_names': list,   # Human-readable product names
            'description': str      # Description of the process
        }
    },
    'default_reactant': str,       # Default reactant key
    'simulation_settings': dict    # Default simulation parameters
}
```

**Raises:**
- `FileNotFoundError`: If reactant_database.json is not found
- `json.JSONDecodeError`: If the JSON file is malformed

**Example:**
```python
database = load_reactant_database()
print(f"Available reactants: {list(database['reactants'].keys())}")
```

### `generate_config_for_reactant(reactant_key, database)`

Generates a complete simulation configuration for a specific reactant.

**Signature:**
```python
def generate_config_for_reactant(reactant_key: str, database: dict) -> dict
```

**Parameters:**
- `reactant_key` (str): The key identifying the reactant in the database
- `database` (dict): The loaded reactant database

**Returns:**
- `dict`: Complete configuration dictionary with all placeholders replaced

**Configuration Structure:**
```python
{
    'simulation_info': {
        'title': str,
        'description': str,
        'version': str,
        'date': str,
        'author': str,
        'reactant_type': str
    },
    'mechanism': {
        'reaction_mechanism_file': str,
        'heat_flux_file': str,
        'n_species': int,
        'n_reactions': int
    },
    'initial_conditions': {
        'composition': str,
        'temperature_K': float,
        'pressure_Pa': float,
        'pressure_bar': float,
        'initial_velocity_ms': float
    },
    'reactor_geometry': {
        'length_m': float,
        'diameter_m': float,
        'diameter_mm': float,
        'cross_sectional_area_m2': float,
        'roughness_m': float
    },
    'operating_conditions': {
        'mass_flow_rate_kgps': float,
        'residence_time_s': float
    },
    'simulation_settings': {
        'n_steps': int
    }
}
```

**Raises:**
- `ValueError`: If the reactant_key is not found in the database
- `FileNotFoundError`: If config_template.json is not found
- `cantera.CanteraError`: If the mechanism file cannot be loaded

**Example:**
```python
database = load_reactant_database()
config = generate_config_for_reactant('ethane', database)
print(f"Temperature: {config['initial_conditions']['temperature_K']} K")
```

### `load_configuration(reactant_key=None)`

Loads configuration for specified reactant or from existing config.json.

**Signature:**
```python
def load_configuration(reactant_key: str = None) -> tuple[dict, dict]
```

**Parameters:**
- `reactant_key` (str, optional): Reactant key to generate configuration for

**Returns:**
- `tuple[dict, dict]`: (configuration, reactant_info)

**Example:**
```python
config, reactant_info = load_configuration('ethane')
print(f"Reactant: {reactant_info['name']}")
```

## Simulation Functions

### `setup_mechanism(config)`

Sets up the gas phase model with kinetics, thermodynamics, and transport.

**Signature:**
```python
def setup_mechanism(config: dict) -> ct.Solution
```

**Parameters:**
- `config` (dict): Simulation configuration

**Returns:**
- `ct.Solution`: Cantera gas solution object

**Example:**
```python
gas = setup_mechanism(config)
print(f"Species: {gas.n_species}, Reactions: {gas.n_reactions}")
```

### `setup_initial_conditions(gas, config)`

Sets up initial gas state and reactor parameters.

**Signature:**
```python
def setup_initial_conditions(gas: ct.Solution, config: dict) -> tuple
```

**Parameters:**
- `gas` (ct.Solution): Cantera gas solution object
- `config` (dict): Simulation configuration

**Returns:**
- `tuple`: (T_0, p_0, composition_0, length, diam, area, roughness, mass_flow_rate, u_0)

**Example:**
```python
T_0, p_0, comp, length, diam, area, rough, mdot, u_0 = setup_initial_conditions(gas, config)
```

### `setup_heat_flux(config)`

Sets up heat flux profile from file. The heat flux profile uses relative positions (0.0 = inlet, 1.0 = outlet) which are automatically converted to absolute positions based on the reactor length specified in the configuration.

**Signature:**
```python
def setup_heat_flux(config: dict) -> tuple[ct.Func1, np.ndarray, np.ndarray]
```

**Parameters:**
- `config` (dict): Simulation configuration containing reactor geometry

**Returns:**
- `tuple`: (heat_flux_function, z_profile_absolute, heatflux_profile)
  - `heat_flux_function`: Cantera function for heat flux interpolation
  - `z_profile_absolute`: Absolute position array in meters
  - `heatflux_profile`: Heat flux values array in W/m²

**Example:**
```python
hf, z_profile, heatflux_profile = setup_heat_flux(config)
# hf(z) returns heat flux at position z (in meters)
```

**Interpolation Methods:**
The heat flux profile JSON file supports two interpolation methods:
- `"linear"` (default): Linear interpolation between data points
- `"step"`: Step-wise interpolation - heat flux remains constant between data points

**Note:** The heat flux profile JSON file should contain relative positions (0.0 to 1.0) that are automatically scaled to the reactor length specified in `config['reactor_geometry']['length_m']`.

### `run_simulation(gas, config, reactant_info, hf, T_0, p_0, length, diam, area, roughness, mass_flow_rate, u_0)`

Runs the main simulation loop for the PFR.

**Note:** Step size, reactor volume, and wall surface area are automatically calculated:
- Step size = reactor length ÷ number of steps
- Volume per step = cross-sectional area × step size
- Wall surface area per step = perimeter × step size

**Signature:**
```python
def run_simulation(
    gas: ct.Solution,
    config: dict,
    reactant_info: dict,
    hf: ct.Func1,
    T_0: float,
    p_0: float,
    length: float,
    diam: float,
    area: float,
    roughness: float,
    mass_flow_rate: float,
    u_0: float
) -> ct.SolutionArray
```

**Parameters:**
- `gas` (ct.Solution): Gas phase model
- `config` (dict): Simulation configuration
- `reactant_info` (dict): Reactant-specific information
- `hf` (ct.Func1): Heat flux function
- `T_0` (float): Initial temperature [K]
- `p_0` (float): Initial pressure [Pa]
- `length` (float): Reactor length [m]
- `diam` (float): Reactor diameter [m]
- `area` (float): Cross-sectional area [m²]
- `roughness` (float): Wall roughness [m]
- `mass_flow_rate` (float): Mass flow rate [kg/s]
- `u_0` (float): Initial velocity [m/s]

**Returns:**
- `ct.SolutionArray`: Simulation results

**Example:**
```python
states = run_simulation(gas, config, reactant_info, hf, T_0, p_0, 
                       length, diam, area, roughness, mass_flow_rate, u_0)
```

## Analysis Functions

### `export_results(gas, states1, config, reactant_info, conversion, yields, T_0, p_0, u_0, hf)`

Exports comprehensive simulation results to CSV and summary files.

**Signature:**
```python
def export_results(gas, states1, config, reactant_info, conversion, yields, T_0, p_0, u_0, hf) -> None
```

**Parameters:**
- `gas` (ct.Solution): Cantera gas solution object
- `states1` (ct.SolutionArray): Solution array containing reactor states
- `config` (dict): Simulation configuration dictionary
- `reactant_info` (dict): Reactant information dictionary
- `conversion` (float): Reactant conversion percentage
- `yields` (dict): Product yield dictionary
- `T_0` (float): Initial temperature
- `p_0` (float): Initial pressure
- `u_0` (float): Initial velocity
- `hf` (ct.Func1): Heat flux function

**Exported Data (245 columns total):**
- **Basic Properties (7):** Position, temperature, pressure, velocity, density, heat flux
- **Thermodynamic Properties (8):** Heat capacity (Cp, Cv), enthalpy, entropy, internal energy, Gibbs free energy, molecular weight
- **Transport Properties (2):** Viscosity, thermal conductivity
- **Composition Data (228):** Mass and mole fractions for all species

**Output Files:**
- `results/results_[Reactant]_T[Temp]K_P[Press]bar_L[Length]m_D[Diam]mm_M[MassFlow]kgps_n[Steps].csv`
- `results/summary_[Reactant]_T[Temp]K_P[Press]bar_L[Length]m_D[Diam]mm_M[MassFlow]kgps_n[Steps].dat`

### `create_visualizations(gas, states1, config, reactant_info, hf, conversion, yields)`

Creates comprehensive visualization plots for the simulation results.

**Signature:**
```python
def create_visualizations(gas, states1, config, reactant_info, hf, conversion, yields) -> None
```

**Parameters:**
- `gas` (ct.Solution): Cantera gas solution object
- `states1` (ct.SolutionArray): Solution array containing reactor states
- `config` (dict): Simulation configuration dictionary
- `reactant_info` (dict): Reactant information dictionary
- `hf` (ct.Func1): Heat flux function
- `conversion` (float): Reactant conversion percentage
- `yields` (dict): Product yield dictionary

**Generated Plots (18 total):**
- Temperature, pressure, velocity, density profiles
- Heat flux profiles (absolute and relative position)
- Thermodynamic properties (Cp, Cv, enthalpy, entropy, molecular weight)
- Transport properties (viscosity, thermal conductivity)
- Process analysis (residence time, conversion, product fractions)

**Output Directory:** `fig/`

**Export Control:**
The function respects the `if_plot_out` configuration flag:
- `if_plot_out: 1` (default): Generates all plots
- `if_plot_out: 0`: Skips plot generation

### `export_results(gas, states1, config, reactant_info, conversion, yields, T_0, p_0, u_0, hf)`

**Export Control:**
The function respects the `if_csv_out` configuration flag:
- `if_csv_out: 1` (default): Exports CSV and summary files
- `if_csv_out: 0`: Skips CSV export

### `calculate_conversion(gas, states, reactant_info)`

Calculates conversion for the specified reactant.

**Signature:**
```python
def calculate_conversion(gas: ct.Solution, states: ct.SolutionArray, reactant_info: dict) -> tuple[float, str]
```

**Parameters:**
- `gas` (ct.Solution): Gas phase model
- `states` (ct.SolutionArray): Simulation results
- `reactant_info` (dict): Reactant information

**Returns:**
- `tuple[float, str]`: (conversion_percentage, species_name)

**Example:**
```python
conversion, species_name = calculate_conversion(gas, states, reactant_info)
print(f"Conversion: {conversion:.1f}%")
```

### `calculate_product_yields(gas, states, reactant_info)`

Calculates yields for target products.

**Signature:**
```python
def calculate_product_yields(gas: ct.Solution, states: ct.SolutionArray, reactant_info: dict) -> dict
```

**Parameters:**
- `gas` (ct.Solution): Gas phase model
- `states` (ct.SolutionArray): Simulation results
- `reactant_info` (dict): Reactant information

**Returns:**
- `dict`: Dictionary mapping product names to yield percentages

**Example:**
```python
yields = calculate_product_yields(gas, states, reactant_info)
for product, yield_val in yields.items():
    print(f"{product}: {yield_val:.2f}%")
```

### `process_and_visualize_results(gas, states1, config, reactant_info, hf, T_0, p_0, u_0)`

Processes results and creates visualizations.

**Signature:**
```python
def process_and_visualize_results(
    gas: ct.Solution,
    states1: ct.SolutionArray,
    config: dict,
    reactant_info: dict,
    hf: ct.Func1,
    T_0: float,
    p_0: float,
    u_0: float
) -> tuple[float, dict]
```

**Parameters:**
- `gas` (ct.Solution): Gas phase model
- `states1` (ct.SolutionArray): Simulation results
- `config` (dict): Simulation configuration
- `reactant_info` (dict): Reactant information
- `hf` (ct.Func1): Heat flux function
- `T_0` (float): Initial temperature [K]
- `p_0` (float): Initial pressure [Pa]
- `u_0` (float): Initial velocity [m/s]

**Returns:**
- `tuple[float, dict]`: (conversion, yields)

**Example:**
```python
conversion, yields = process_and_visualize_results(gas, states, config, 
                                                 reactant_info, hf, T_0, p_0, u_0)
```

## Utility Functions

### `dp_churchill(thermo, mass_flow_rate, area, diam, roughness)`

Calculates pressure drop using Churchill correlation for friction factor.

**Signature:**
```python
def dp_churchill(thermo: ct.ThermoPhase, mass_flow_rate: float, area: float, 
                diam: float, roughness: float) -> tuple[float, float]
```

**Parameters:**
- `thermo` (ct.ThermoPhase): Thermodynamic state
- `mass_flow_rate` (float): Mass flow rate [kg/s]
- `area` (float): Cross-sectional area [m²]
- `diam` (float): Diameter [m]
- `roughness` (float): Wall roughness [m]

**Returns:**
- `tuple[float, float]`: (velocity, pressure_drop_per_length)

**Example:**
```python
u, dpdz = dp_churchill(thermo, mass_flow_rate, area, diam, roughness)
```

## Data Structures

### Reactant Database Structure

```python
{
    "reactants": {
        "reactant_key": {
            "name": str,                    # Human-readable name
            "formula": str,                 # Chemical formula
            "mechanism_file": str,          # Path to mechanism file
            "feed_species": str,            # Species name in mechanism
            "diluent": str,                 # Diluent species name
            "diluent_ratio": float,         # Diluent to reactant ratio
            "target_products": list[str],   # List of product species
            "product_names": list[str],     # Human-readable product names
            "description": str              # Description of the process
        }
    },
    "default_reactant": str,                # Default reactant key
    "simulation_settings": {                # Default simulation parameters
        "temperature_range": list[float],
        "pressure_range": list[float],
        "residence_time_range": list[float]
    }
}
```

### Configuration Structure

```python
{
    "simulation_info": {
        "title": str,
        "description": str,
        "version": str,
        "date": str,
        "author": str,
        "reactant_type": str
    },
    "mechanism": {
        "reaction_mechanism_file": str,
        "heat_flux_file": str,
        "n_species": int,
        "n_reactions": int
    },
    "initial_conditions": {
        "composition": str,
        "temperature_K": float,
        "pressure_Pa": float,
        "pressure_bar": float,
        "initial_velocity_ms": float
    },
    "reactor_geometry": {
        "length_m": float,
        "diameter_m": float,
        "diameter_mm": float,
        "cross_sectional_area_m2": float,
        "roughness_m": float
    },
    "operating_conditions": {
        "mass_flow_rate_kgps": float,
        "residence_time_s": float
    },
    "simulation_settings": {
        "n_steps": int,
        "step_size_m": float
    },
    "heat_flux_profile": {
        "n_points": int,
        "z_range_m": list[float],
        "heat_flux_range_Wm2": list[float],
        "peak_heat_flux_Wm2": float
    },
    "pressure_drop": {
        "method": str,
        "roughness_factor": float
    }
}
```

## Error Handling

### Common Exceptions

#### `ValueError`
Raised when:
- Reactant key is not found in database
- Invalid configuration parameters
- Species not found in mechanism

#### `FileNotFoundError`
Raised when:
- Reactant database file is missing
- Configuration template is missing
- Mechanism file is missing
- Heat flux file is missing

#### `cantera.CanteraError`
Raised when:
- Mechanism file cannot be loaded
- Invalid species composition
- Thermodynamic state errors

#### `json.JSONDecodeError`
Raised when:
- Malformed JSON in configuration files
- Invalid database structure

### Error Handling Best Practices

```python
try:
    database = load_reactant_database()
    config = generate_config_for_reactant('ethane', database)
    gas = setup_mechanism(config)
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Mechanism Generation
All kinetic mechanisms in this system were generated using [**Reaction Mechanism Generator (RMG)**](https://reactionmechanismgenerator.github.io/RMG-Py/), an automatic chemical reaction mechanism generator developed by the RMG team at MIT. RMG constructs kinetic models composed of elementary chemical reaction steps using a general understanding of how molecules react.

### Mechanism Complexity Impact

| Mechanism | Species | Reactions | Relative Speed | Memory Usage |
|-----------|---------|-----------|----------------|--------------|
| Ethane | 35 | 135 | Very Fast | Low |
| Propane | 53 | 325 | Fast | Low |
| n-Hexane | 153 | 2,146 | Moderate | Medium |
| Naphtha | 1,951 | 82,557 | **Very Slow** | High |

### Memory Usage
- Large mechanisms (>1000 species) may require significant memory
- Solution arrays scale with number of species and simulation steps
- CSV export scales with species count (245+ columns for propane, 1000+ for naphtha)
- Consider reducing step count for memory-constrained systems
- **Naphtha mechanism requires ~4GB RAM** for full simulation

### Computational Performance
- Simulation time scales **exponentially** with number of reactions and species
- Complex mechanisms may require longer convergence times
- **Naphtha simulations can take 10-30 minutes** vs. seconds for simpler mechanisms
- Consider mechanism reduction for faster simulations

### Optimization Tips
- **For naphtha**: Reduce `n_steps` to 50-100 for faster computation
- Use appropriate step sizes for convergence vs. speed trade-off
- Monitor solver warnings for convergence issues
- Consider parallel processing for batch simulations
- **Memory warning**: Naphtha mechanism may cause memory issues on systems with <8GB RAM

---

**Last Updated:** January 15, 2025  
**Version:** 2.0.0
