# API Reference - Generalized PFR Simulation System

This document provides comprehensive API documentation for all functions, classes, and modules in the Generalized PFR Simulation System.

## Table of Contents

- [Core Modules](#core-modules)
- [Configuration Functions](#configuration-functions)
- [Simulation Functions](#simulation-functions)
- [Analysis Functions](#analysis-functions)
- [Utility Functions](#utility-functions)
- [Plot Utilities (`src/utils/`)](#plot-utilities-srcutils)
- [Data Structures](#data-structures)
- [Error Handling](#error-handling)

## Core Modules

### src/cantera/pfr_simulator.py

The main simulation module containing all core PFR functionality. The interactive entry point is **`notebooks/Main_1_run_pfr.ipynb`**, which imports and uses this module.

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
- `FileNotFoundError`: If main1_reactant_database.json is not found
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
- `FileNotFoundError`: If main1_pfr_run_config_template.json is not found
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

## Plot Utilities (`src/utils/`)

Reusable matplotlib helpers shared across the `Main_*` notebooks. All helpers
respect the global rcParams set by `setup_matplotlib()` and are designed so that
the caller owns figure saving (`fig.savefig(...)`) — none of them write to disk.

### `setup_matplotlib(ax=None)` — `src/utils/plot_style.py`

Apply the project's standard matplotlib style globally and, optionally, to one
or more Axes. Call once per notebook/script at import time with no arguments
to set rcParams; pass `ax` after `plt.subplots(...)` to apply per-axis
finishing touches (minor ticks, in-pointing tick direction, hidden top/right
spines).

**Signature:**
```python
def setup_matplotlib(ax: None | Axes | np.ndarray = None) -> None
```

**Parameters:**
- `ax` (None | Axes | ndarray of Axes): If `None`, only global rcParams are
  updated. If one or many Axes, per-axis styling is also applied.

**Example:**
```python
from src.utils.plot_style import setup_matplotlib
setup_matplotlib()                              # global rcParams
fig, axes = plt.subplots(1, 3)
setup_matplotlib(axes)                          # per-axis polish
```

### JSON-driven aesthetics helpers — `src/utils/plot_style.py`

Lower-level helpers that read `configs/style/figure_aesthetics.json`
(legacy fallbacks: flat `configs/figure_aesthetics.json`,
`styles/figure_aesthetics.json`). Useful when a notebook or script needs
profile-specific colors/labels (e.g. temperature vs. pressure curves).

| Function | Purpose |
|---|---|
| `load_aesthetics(config_file=None)` | Load and return the aesthetics dictionary. |
| `apply_style(aesthetics=None, config_file=None)` | Apply global rcParams from the JSON. |
| `create_figure(aesthetics=None, figsize=None, ...)` | Create a `Figure` with aesthetics-driven defaults. |
| `setup_axes(ax, aesthetics=None, ...)` | Apply grid/spine settings to an `Axes`. |
| `setup_legend(ax, aesthetics=None, **kwargs)` | Render a legend with aesthetics defaults; returns the `Legend`. |
| `get_profile_style(profile_name, ...)` | Return a per-profile style dict (color, linestyle, label, ylabel, title). |
| `get_color(name, ...)` | Look up a named color from the JSON palette. |
| `save_figure(fig, filename, ...)` | Save with aesthetics-driven format/DPI/`bbox_inches`. |
| `plot_profile(x, y, profile_name, ...)` | One-call profile plot using `get_profile_style`. Returns `(fig, ax)`. |

**Example — profile plot from JSON:**
```python
from src.utils.plot_style import plot_profile
fig, ax = plot_profile(z, T, 'temperature', xlabel='$z$ [m]')
```

### `plot_parallel_coordinates(...)` — `src/utils/plot_parallel.py`

Inselberg-style parallel coordinates plot for continuous multidimensional
data. Each row becomes a polyline across `len(dims)` parallel vertical axes;
each axis is independently min-max normalized so all values share a single
[0, 1] display range. Rendered with a `LineCollection` for speed.

**Signature:**
```python
def plot_parallel_coordinates(
    df: pd.DataFrame,
    dims: Sequence[str],
    color_by: Optional[str] = None,
    *,
    axis_labels: Optional[Sequence[str]] = None,
    color_label: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "magma",
    alpha: float = 0.35,
    linewidth: float = 0.8,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (12.0, 4.5),
    sort_by_color: bool = True,
) -> tuple[plt.Figure, plt.Axes]
```

**Parameters:**
- `df` (DataFrame): Source data; must contain every column in `dims` (and `color_by` if given).
- `dims` (sequence of str): Column names in the order they should appear on the x-axis. At least 2.
- `color_by` (str, optional): Continuous column whose value colors each polyline via `cmap`. If `None`, lines are drawn in neutral gray and no colorbar is added.
- `axis_labels` (sequence of str, optional): Display labels for each axis; defaults to `dims`.
- `color_label` (str, optional): Colorbar label; defaults to `color_by`.
- `title` (str, optional): Axes title.
- `cmap` (str): Matplotlib colormap name. Defaults to `"magma"`.
- `alpha`, `linewidth` (float): Polyline transparency and width. Lower `alpha` for dense datasets.
- `ax` (Axes, optional): Existing axes to draw into; otherwise a new figure is created.
- `figsize` ((float, float)): Figure size when `ax` is `None`.
- `sort_by_color` (bool): If `True` and `color_by` is given, draw rows from low to high color value so high-value lines appear on top.

**Returns:** `(fig, ax)` — caller saves with `fig.savefig(...)`.

**Raises:** `ValueError` if `len(dims) < 2` or if no rows remain after dropping NaNs in `dims + [color_by]`.

**Example:**
```python
from src.utils.plot_parallel import plot_parallel_coordinates
fig, ax = plot_parallel_coordinates(
    df_design,
    dims=['initial_temperature_K', 'pressure_bar', 'reactor_length_m',
          'diameter_mm', 'mass_flow_rate_kgps', 'heat_flux_kWm2'],
    color_by='nhexane_conversion_pct',
    color_label='n-hexane conversion (%)',
    title='Training space: parallel coordinates',
)
fig.savefig('outputs/figures/.../parallel_coordinates_design_space.png',
            dpi=200, bbox_inches='tight')
```

### `plot_parallel_sets(...)` — `src/utils/plot_parallel.py`

Kosara-style Parallel Sets for binned multidimensional data. Each column in
`dims` is binned into `n_bins` categories; for every adjacent pair of axes,
cubic-Bézier ribbons are drawn whose width is proportional to the joint count
of (left-bin, right-bin) and whose color is the mean of `color_by` over rows
in that joint bin (or a single `base_color` if `color_by` is `None`).

**Signature:**
```python
def plot_parallel_sets(
    df: pd.DataFrame,
    dims: Sequence[str],
    color_by: Optional[str] = None,
    *,
    n_bins: int = 5,
    bin_strategy: str = "equal_width",          # or "quantile"
    axis_labels: Optional[Sequence[str]] = None,
    color_label: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "magma",
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    base_color: str = "#4477AA",
    ribbon_alpha: float = 0.55,
    gap_frac: float = 0.015,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (13.0, 5.5),
    bin_label_fontsize: int = 7,
    show_bin_labels: bool = True,
) -> tuple[plt.Figure, plt.Axes]
```

**Parameters:**
- `df`, `dims`, `color_by`, `axis_labels`, `color_label`, `title`, `cmap`, `ax`, `figsize`: same semantics as `plot_parallel_coordinates`.
- `n_bins` (int): Number of bins per axis (categories). Note: a constant column collapses to fewer effective bins.
- `bin_strategy` ({`'equal_width'`, `'quantile'`}): How to bin continuous columns.
- `color_vmin`, `color_vmax` (float, optional): Manual color-scale bounds; default to the data range of `color_by`.
- `base_color` (str): Ribbon color when `color_by` is `None`.
- `ribbon_alpha` (float): Ribbon transparency.
- `gap_frac` (float): Gap between bin segments on each axis, as a fraction of axis height.
- `bin_label_fontsize` (int): Font size for per-bin range labels written to the right of each axis.
- `show_bin_labels` (bool): If `True`, annotate each bin segment with its value range.

**Returns:** `(fig, ax)` — caller saves with `fig.savefig(...)`.

**Raises:** `ValueError` if `len(dims) < 2`, if no rows remain after dropping NaNs, or if `bin_strategy` is unrecognized.

**Example:**
```python
from src.utils.plot_parallel import plot_parallel_sets
fig, ax = plot_parallel_sets(
    df_design,
    dims=['initial_temperature_K', 'pressure_bar', 'reactor_length_m',
          'diameter_mm', 'mass_flow_rate_kgps', 'heat_flux_kWm2'],
    color_by='nhexane_conversion_pct',
    n_bins=5, bin_strategy='equal_width',
    color_label='mean n-hexane conversion (%)',
    title='Training space: parallel sets (5 equal-width bins/axis)',
)
fig.savefig('outputs/figures/.../parallel_sets_design_space.png',
            dpi=200, bbox_inches='tight')
```

**Implementation notes:**
- Ribbons are stacked within each bin in ascending-right-bin order on the left side and ascending-left-bin order on the right side, which keeps the visual stable across reruns.
- Joint-bin means used for color are computed via `np.add.at`, so the cost is `O(N_rows + n_bins**2 * (n_dims - 1))` — fast even for tens of thousands of rows.
- Bin range labels use the same compact tick formatter as the parallel-coordinates axes (auto-switches between fixed and scientific notation).

### `start_run_log(...)` / `stop_run_log()` — `src/utils/run_log.py`

Tee `stdout` and `stderr` to `outputs/reports/<safe_notebook_name>.txt` while still
echoing to the notebook display. The log file is opened in **overwrite** mode each
time `start_run_log` runs, so the latest full run always replaces the previous file.
If logging is already active, a second call closes the previous tee and starts a
fresh file at the same path (for example after re-running the setup cell).
`stop_run_log()` is optional — only call it if you want to close the file
mid-session.

**Signature:**
```python
def start_run_log(notebook_name: str,
                  reports_dir: str | pathlib.Path = 'outputs/reports') -> pathlib.Path
def stop_run_log() -> None
```

**Returns:** `start_run_log` returns the `Path` to the active log file.

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
