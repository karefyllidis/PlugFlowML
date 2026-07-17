#!/usr/bin/env python3
"""
Generalized Plug Flow Reactor (PFR) Simulation Workflow
=======================================================

A comprehensive simulation framework for modeling steam cracking reactions
in plug flow reactors using Cantera. Supports multiple feedstocks including
ethane, propane, naphtha, and n-hexane with automatic configuration generation
and species name handling.

Author: Nikolas Karefyllidis, PhD
Development Started: 2025-09-15
License: MIT

Features:
---------
- Multi-reactant support (ethane, propane, naphtha, n-hexane)
- Automatic configuration generation
- Species name standardization across different mechanisms
- Comprehensive result visualization and export
- Flexible product tracking and yield calculations
- Professional output formatting and file naming

Dependencies:
-------------
- cantera >= 3.1.0
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- scipy >= 1.7.0

Usage:
------
    python Main_GeneralizedPFR.py --reactant ethane
    python Main_GeneralizedPFR.py --list
    python run_simulation.py ethane

For more information, see README.md
"""

# =============================================================================
# IMPORTS
# =============================================================================
import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import os
import time
import pandas as pd
import json
import argparse
import warnings

# numpy >=2.0 renamed trapz -> trapezoid and removed trapz outright in >=2.4;
# a ternary picks the right one lazily (an eager getattr(np, ..., np.trapz)
# default would itself raise AttributeError once np.trapz no longer exists).
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import sys
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

# Suppress all Cantera/SUNDIALS solver warnings and messages
# These are common and don't affect simulation results
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*rank.*')
warnings.filterwarnings('ignore', message='.*CVode.*')
warnings.filterwarnings('ignore', message='.*SUNDIALS.*')
warnings.filterwarnings('ignore', message='.*WARNING.*')

# Suppress SUNDIALS rank messages (MPI process rank, always 0 for single-threaded)
# These messages like "[rank 0]" come from the underlying CVode solver
# They're informational and can clutter output during batch data generation
class SuppressRankMessages:
    """Context manager to suppress SUNDIALS rank messages from stderr."""
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stderr = self
        
    def __exit__(self, *args):
        sys.stderr = self.original_stderr
        
    def write(self, message):
        # Filter out rank messages
        if '[rank' not in message.lower() and 'cvode' not in message.lower():
            self.original_stderr.write(message)
        
    def flush(self):
        self.original_stderr.flush()

# Import plot style utilities
from src.utils.plot_style import (
    load_aesthetics, apply_style, create_figure, setup_axes,
    get_profile_style, get_color, save_figure, setup_legend, plot_profile
)

# =============================================================================
# PATH UTILITIES
# =============================================================================
def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_config_path(*parts):
    """Path under ``configs/`` (e.g. ``get_config_path('simulation', 'main1_reactant_database.json')``)."""
    return os.path.join(get_project_root(), 'configs', *parts)


def resolve_heat_flux_file_path(heat_flux_file: str) -> str:
    """
    Resolve ``mechanism.heat_flux_file`` to an absolute path.

    Supports repo-relative ``configs/...`` paths, bare ``main1_heat_flux_profile.json``,
    and legacy flat ``configs/heat_flux_profile.json`` (under ``configs/simulation/``).
    """
    root = get_project_root()
    if os.path.isabs(heat_flux_file):
        return heat_flux_file
    if heat_flux_file in ('heat_flux_profile.json', 'main1_heat_flux_profile.json'):
        return get_config_path('simulation', 'main1_heat_flux_profile.json')
    if heat_flux_file.startswith('configs/'):
        path = os.path.normpath(os.path.join(root, heat_flux_file.replace('/', os.sep)))
        if os.path.isfile(path):
            return path
        if heat_flux_file in ('configs/heat_flux_profile.json', 'configs/main1_heat_flux_profile.json'):
            return get_config_path('simulation', 'main1_heat_flux_profile.json')
        return path
    return os.path.join(root, heat_flux_file)

def get_output_path(subdir, filename):
    """Get path to an output file."""
    return os.path.join(get_project_root(), 'outputs', subdir, filename)

# =============================================================================
# REACTANT DATABASE AND CONFIGURATION LOADING
# =============================================================================
def load_reactant_database():
    """
    Load the reactant database containing all available feedstocks and their properties.
    
    Returns:
    --------
    dict
        Dictionary containing reactant information with the following structure:
        {
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
    
    Raises:
    -------
    FileNotFoundError
        If main1_reactant_database.json is not found
    json.JSONDecodeError
        If the JSON file is malformed
    """
    config_path = get_config_path('simulation', 'main1_reactant_database.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def generate_config_for_reactant(reactant_key: str, database: dict) -> dict:
    """
    Generate a complete simulation configuration for a specific reactant.
    
    This function takes a reactant key and generates a fully configured
    simulation setup by replacing placeholders in the configuration template
    with reactant-specific values.
    
    Parameters:
    -----------
    reactant_key : str
        The key identifying the reactant in the database (e.g., 'ethane', 'propane')
    database : dict
        The loaded reactant database containing all reactant information
    
    Returns:
    --------
    dict
        Complete configuration dictionary with all placeholders replaced:
        - Simulation metadata (title, description, version, date)
        - Mechanism information (file path, species count, reaction count)
        - Initial conditions (composition, temperature, pressure)
        - Reactor geometry (length, diameter, cross-sectional area)
        - Operating conditions (mass flow rate, residence time)
        - Simulation settings (number of steps, step size)
        - Heat flux profile information
        - Pressure drop calculation method
    
    Raises:
    -------
    ValueError
        If the reactant_key is not found in the database
    FileNotFoundError
        If main1_pfr_run_config_template.json is not found
    cantera.CanteraError
        If the mechanism file cannot be loaded (non-critical, uses fallback values)
    
    Notes:
    ------
    - The function automatically determines species and reaction counts by loading
      the mechanism file
    - If mechanism loading fails, 'Unknown' is used as a fallback
    - Feed composition is automatically generated from reactant and diluent information
    - All timestamps are set to the current date and time
    """
    if reactant_key not in database['reactants']:
        raise ValueError(f"Reactant '{reactant_key}' not found in database. Available: {list(database['reactants'].keys())}")
    
    reactant_info = database['reactants'][reactant_key]
    
    # Load configuration template
    config_path = get_config_path('simulation', 'main1_pfr_run_config_template.json')
    with open(config_path, 'r') as f:
        config_template = json.load(f)
    
    # Replace placeholders with reactant-specific values
    config_str = json.dumps(config_template)
    config_str = config_str.replace('{{REACTANT_NAME}}', reactant_info['name'])
    config_str = config_str.replace('{{REACTANT_DESCRIPTION}}', reactant_info['description'])
    config_str = config_str.replace('{{REACTANT_KEY}}', reactant_key)
    config_str = config_str.replace('{{MECHANISM_FILE}}', reactant_info['mechanism_file'])
    config_str = config_str.replace('{{DATE}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Create feed composition string
    feed_composition = f"{reactant_info['feed_species']}, {reactant_info['diluent']}:{reactant_info['diluent_ratio']}"
    config_str = config_str.replace('{{FEED_COMPOSITION}}', feed_composition)
    
    # Load mechanism to get species and reaction counts (resolve path relative to project root)
    mechanism_path = reactant_info['mechanism_file']
    if not os.path.isabs(mechanism_path):
        mechanism_path = os.path.join(get_project_root(), mechanism_path)
    if not os.path.isfile(mechanism_path):
        raise FileNotFoundError(
            f"Mechanism file not found: {mechanism_path}\n"
            "Add your mechanism YAML files to the mechanisms/ directory (see README: Adding New Reactants)."
        )
    try:
        gas_temp = ct.Solution(mechanism_path)
        config_str = config_str.replace('{{N_SPECIES}}', str(gas_temp.n_species))
        config_str = config_str.replace('{{N_REACTIONS}}', str(gas_temp.n_reactions))
    except Exception as e:
        print(f"Warning: Could not load mechanism to get species count: {e}")
        config_str = config_str.replace('{{N_SPECIES}}', 'Unknown')
        config_str = config_str.replace('{{N_REACTIONS}}', 'Unknown')
    
    return json.loads(config_str)

def load_configuration(reactant_key: str = None):
    """Load configuration for specified reactant or from existing config.json."""
    if reactant_key:
        # Generate new configuration for specified reactant
        database = load_reactant_database()
        config = generate_config_for_reactant(reactant_key, database)
        reactant_info = database['reactants'][reactant_key]
        return config, reactant_info
    else:
        # Load existing configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Try to get reactant info from config
        reactant_key = config.get('simulation_info', {}).get('reactant_type', 'unknown')
        database = load_reactant_database()
        reactant_info = database['reactants'].get(reactant_key, {})
        
        return config, reactant_info

# =============================================================================
# REACTION MECHANISM SETUP
# =============================================================================
def setup_mechanism(config: dict):
    """Setup the gas phase model with kinetics, thermodynamics, and transport."""
    reaction_mechanism = config['mechanism']['reaction_mechanism_file']
    # Resolve mechanism path relative to project root
    if not os.path.isabs(reaction_mechanism):
        reaction_mechanism = os.path.join(get_project_root(), reaction_mechanism)
    gas = ct.Solution(reaction_mechanism)
    
    print('Gas mechanism contains {} species and {} reactions'.format(gas.n_species, gas.n_reactions))
    return gas

# =============================================================================
# INITIAL CONDITIONS AND REACTOR PARAMETERS
# =============================================================================
def setup_initial_conditions(gas: ct.Solution, config: dict):
    """Setup initial gas state and reactor parameters."""
    composition_0 = config['initial_conditions']['composition']
    T_0 = config['initial_conditions']['temperature_K']
    p_0 = config['initial_conditions']['pressure_Pa']
    
    # Set initial gas state
    gas.TPY = T_0, p_0, composition_0
    
    # Extract reactor geometry
    length = config['reactor_geometry']['length_m']
    diameter = config['reactor_geometry']['diameter_m']
    # Calculate cross-sectional area from diameter
    area = np.pi * (diameter / 2) ** 2
    roughness = config['reactor_geometry']['roughness_m']
    
    # Extract operating conditions
    mass_flow_rate = config['operating_conditions']['mass_flow_rate_kgps']
    u_0 = mass_flow_rate/gas.density/area  # Initial velocity [m/s]
    
    return T_0, p_0, composition_0, length, diameter, area, roughness, mass_flow_rate, u_0

# =============================================================================
# HEAT FLUX PROFILE
# =============================================================================
def setup_heat_flux(config: dict):
    """Setup heat flux profile from JSON file."""
    import json
    
    heat_flux_file = resolve_heat_flux_file_path(config['mechanism']['heat_flux_file'])
    reactor_length = config['reactor_geometry']['length_m']
    
    # Load heat flux profile from JSON
    with open(heat_flux_file, 'r') as f:
        heat_flux_data = json.load(f)
    
    # Extract data points and interpolation method
    data_points = heat_flux_data['heat_flux_profile']['data_points']
    interpolation_method = heat_flux_data['heat_flux_profile'].get('interpolation_method', 'linear')
    
    z_profile_relative = np.array([point['position'] for point in data_points])
    heatflux_profile = np.array([point['heat_flux'] for point in data_points])
    
    # Convert relative positions (0.0-1.0) to absolute positions (0 to reactor_length)
    z_profile_absolute = z_profile_relative * reactor_length
    
    # Create Cantera function for heat flux interpolation
    if interpolation_method == 'step':
        # Step-wise interpolation: use the value from the previous data point
        def step_interp(z):
            # Find the index where z would be inserted to maintain order
            idx = np.searchsorted(z_profile_absolute, z, side='right')
            # Clamp index to valid range
            idx = max(0, min(idx - 1, len(heatflux_profile) - 1))
            return heatflux_profile[idx]
        
        hf = ct.Func1(step_interp)
    else:  # Default to linear interpolation
        hf = ct.Func1(lambda z: np.interp(z, z_profile_absolute, heatflux_profile))
    
    return hf, z_profile_absolute, heatflux_profile

# =============================================================================
# PRESSURE DROP CALCULATION
# =============================================================================
def dp_churchill(thermo, mass_flow_rate, area, diam, roughness):
    """Calculate pressure drop using Churchill correlation for friction factor."""
    rho = thermo.density      # Density [kg/m³]
    u = mass_flow_rate/rho/area  # Velocity [m/s]
    try:
        mu = thermo.viscosity     # Dynamic viscosity [Pa·s]
        Re = rho*diam*u/mu        # Reynolds number [-]
        
        # Churchill correlation for friction factor
        psi1 = (-2.457*np.log((7/Re)**0.9+0.27*roughness/diam))**16
        psi2 = (37530/Re)**16
        fD = 8*((8/Re)**12+1/(psi1+psi2)**1.5)**(1.0/12.0)
        
        # Pressure drop per unit length [Pa/m]
        dpdz = fD/diam*rho/2*u**2
    except (NotImplementedError, AttributeError):
        # Transport properties not available, assume no pressure drop
        dpdz = 0.0   
    
    return u, dpdz

# =============================================================================
# CONVERSION CALCULATIONS
# =============================================================================
def calculate_conversion(gas: ct.Solution, states, reactant_info: dict):
    """Calculate conversion for the specified reactant."""
    feed_species = reactant_info.get('feed_species', '')
    
    # Handle different species name formats
    if '(' in feed_species:
        # Format like "ethane(1)" or "C2H4(8)"
        species_name = feed_species
    else:
        # Format like "C3H8" - try to find the species
        species_name = feed_species
    
    try:
        species_idx = gas.species_index(species_name)
        initial = states.Y[0, species_idx]
        final = states.Y[-1, species_idx]
        conversion = (1 - final / initial) * 100
        return conversion, species_name
    except:
        # Fallback: try to find any species that might be the reactant
        for species in gas.species_names:
            if reactant_info['name'].lower() in species.lower() or reactant_info['formula'] in species:
                try:
                    species_idx = gas.species_index(species)
                    initial = states.Y[0, species_idx]
                    final = states.Y[-1, species_idx]
                    conversion = (1 - final / initial) * 100
                    return conversion, species
                except:
                    continue
        
        print(f"Warning: Could not find reactant species for {reactant_info['name']}")
        return 0.0, "Unknown"

# =============================================================================
# PRODUCT YIELD CALCULATIONS
# =============================================================================
def calculate_product_yields(gas: ct.Solution, states, reactant_info: dict):
    """Calculate yields for target products."""
    yields = {}
    target_products = reactant_info.get('target_products', [])
    product_names = reactant_info.get('product_names', [])
    
    for i, product in enumerate(target_products):
        try:
            product_idx = gas.species_index(product)
            yield_value = states.Y[-1, product_idx] * 100
            product_name = product_names[i] if i < len(product_names) else product
            yields[product_name] = yield_value
        except:
            yields[product] = 0.0
    
    return yields

# =============================================================================
# SIMULATION LOOP
# =============================================================================
def run_simulation(gas, config, reactant_info, hf, T_0, p_0, length, diameter, area, roughness, mass_flow_rate, u_0):
    """Run the main simulation loop."""
    n_steps = config['simulation_settings']['n_steps']
    # Calculate step size from reactor length and number of steps
    dz = length / n_steps
    # Calculate reactor volume and wall surface area for a single step
    # Volume per step = cross-sectional area × step size
    r_vol = area * dz
    # Wall surface area per step = perimeter × step size  
    r_area = np.pi * diameter * dz
    
    # Create reactor network components
    upstream = ct.Reservoir(gas, name='upstream')
    furnace = ct.Reservoir(gas)
    r1 = ct.IdealGasReactor(gas, volume=r_vol)
    w1 = ct.Wall(furnace, r1, A=r_area, Q=hf(0.0), U=0.0, K=0.0)
    downstream = ct.Reservoir(gas, name='downstream')
    
    # Flow controllers
    m = ct.MassFlowController(upstream, r1, mdot=mass_flow_rate)
    v = ct.PressureController(r1, downstream, K=1e-5)
    v.primary = m
    
    # Create reactor network solver
    sim1 = ct.ReactorNet([r1])
    
    print(f"Starting simulation with {n_steps} steps...")
    
    # Initialize solution array
    states1 = ct.SolutionArray(r1.thermo, 1, extra={'z': [0.0], 'velocity':[u_0]})
    
    # Initialize simulation variables
    dist = 0.0
    p = p_0
    
    # Main simulation loop
    for n in range(n_steps):
        if n % (n_steps // 10) == 0:
            print(f"Progress: {n}/{n_steps} ({100*n/n_steps:.0f}%)")
        
        dist = dist + dz
        
        # Stop if we've reached the reactor length
        if dist >= length:
            dist = length  # Set to exact length
            break
        
        u, dpdz = dp_churchill(r1.thermo, mass_flow_rate, area, diameter, roughness)
        p = p - dpdz*dz
        
        gas.TPY = r1.thermo.T, p, r1.thermo.Y
        w1.heat_flux = hf(dist)
        
        upstream.syncState()
        r1.syncState()
        downstream.syncState()
        
        sim1.reinitialize()
        # Suppress SUNDIALS rank messages during solver advance
        with SuppressRankMessages():
            sim1.advance_to_steady_state()
        
        states1.append(r1.thermo.state, z=dist, velocity=u)
    
    return states1

# =============================================================================
# RESULTS PROCESSING AND VISUALIZATION
# =============================================================================
def process_and_visualize_results(gas, states1, config, reactant_info, hf, T_0, p_0, u_0):
    """Process results and create visualizations."""
    print("Simulation completed!")
    print(f"Final temperature: {states1.T[-1]:.1f} K")
    print(f"Final pressure: {states1.P[-1]/1e5:.2f} bar")
    
    # Calculate conversion
    conversion, species_name = calculate_conversion(gas, states1, reactant_info)
    print(f"{reactant_info['name']} conversion: {conversion:.1f}%")
    
    # Calculate product yields
    yields = calculate_product_yields(gas, states1, reactant_info)
    print("Product yields:")
    for product, yield_val in yields.items():
        print(f"  {product}: {yield_val:.2f}% (mass)")
    
    # Create visualizations (if enabled)
    if config.get('export_controls', {}).get('if_plot_out', 1):
        create_visualizations(gas, states1, config, reactant_info, hf, conversion, yields)
    else:
        print("Plot generation disabled by configuration")
    
    # Export data (if enabled)
    if config.get('export_controls', {}).get('if_csv_out', 1):
        export_results(gas, states1, config, reactant_info, conversion, yields, T_0, p_0, u_0, hf)
    else:
        print("CSV export disabled by configuration")
    
    return conversion, yields

def create_visualizations(gas, states1, config, reactant_info, hf, conversion, yields):
    """Create all visualization plots using ``configs/style/figure_aesthetics.json``."""
    aesthetics = load_aesthetics()
    apply_style(aesthetics)

    def _save_profile(profile_key: str, y, filename: str, **plot_kw):
        path = get_output_path('figures', filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig, _ = plot_profile(
            states1.z, y, profile_key,
            aesthetics=aesthetics,
            output_path=path,
            **plot_kw,
        )
        plt.close(fig)
        print(f"Saved {filename} to {path}")

    _save_profile('temperature', states1.T, 'temperature_profile.png')
    _save_profile('pressure', states1.P / 1e5, 'pressure_profile.png')
    _save_profile('velocity', states1.velocity, 'velocity_profile.png')
    _save_profile('density', states1.density, 'density_profile.png')
    heat_flux_values = [hf(z) for z in states1.z]
    _save_profile('heat_flux', heat_flux_values, 'heat_flux_profile.png')
    _save_profile('molecular_weight', states1.mean_molecular_weight, 'molecular_weight_profile.png')

    create_heat_flux_relative_figure(config, hf, aesthetics)
    create_species_plots(gas, states1, config, reactant_info, aesthetics)

def create_heat_flux_relative_figure(config, hf, aesthetics=None):
    """Create heat flux vs relative position figure (uses centralized aesthetics)."""
    import json

    if aesthetics is None:
        aesthetics = load_aesthetics()
        apply_style(aesthetics)

    heat_flux_file = resolve_heat_flux_file_path(config['mechanism']['heat_flux_file'])
    with open(heat_flux_file, 'r') as f:
        heat_flux_data = json.load(f)

    data_points = heat_flux_data['heat_flux_profile']['data_points']
    interpolation_method = heat_flux_data['heat_flux_profile'].get('interpolation_method', 'linear')

    z_profile_relative = np.array([point['position'] for point in data_points])
    heatflux_profile = np.array([point['heat_flux'] for point in data_points])

    z_fine = np.linspace(0.0, 1.0, 1000)

    if interpolation_method == 'step':
        heat_flux_fine = np.zeros_like(z_fine)
        for i, z in enumerate(z_fine):
            idx = np.searchsorted(z_profile_relative, z, side='right')
            idx = max(0, min(idx - 1, len(heatflux_profile) - 1))
            heat_flux_fine[i] = heatflux_profile[idx]
    else:
        heat_flux_fine = np.interp(z_fine, z_profile_relative, heatflux_profile)

    line_st = get_profile_style('heat_flux', aesthetics)
    font = aesthetics.get('font', {})
    fig = create_figure(aesthetics)
    ax = fig.add_subplot(111)

    ax.plot(
        z_fine, heat_flux_fine,
        color=line_st['color'],
        linewidth=line_st['linewidth'],
        linestyle=line_st['linestyle'],
        label=f"{interpolation_method.capitalize()} interpolation",
    )
    ax.plot(
        z_profile_relative, heatflux_profile,
        's', color=get_color('quaternary', aesthetics), markersize=8,
        markeredgecolor=get_color('tertiary', aesthetics), markeredgewidth=1,
        label='Data points',
    )

    ax.set_xlabel('Relative position', fontsize=font.get('label_size', 12))
    ax.set_ylabel('Heat flux [W/m²]', fontsize=font.get('label_size', 12))
    ax.set_title(
        'Heat flux profile vs relative position',
        fontsize=font.get('title_size', 14),
        fontweight=font.get('title_weight', 'normal'),
    )
    axes_cfg = aesthetics.get('axes', {})
    if axes_cfg.get('grid', True):
        ax.grid(True, alpha=axes_cfg.get('grid_alpha', 0.3), linestyle=axes_cfg.get('grid_style', '-'),
                color=axes_cfg.get('grid_color', 'gray'))
    if axes_cfg.get('spines_top', False) is False:
        ax.spines['top'].set_visible(False)
    if axes_cfg.get('spines_right', False) is False:
        ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', labelsize=font.get('tick_size', 10))
    setup_legend(ax, aesthetics)
    textstr = f'Interpolation: {interpolation_method.capitalize()}\nData points: {len(data_points)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=font.get('tick_size', 10),
            verticalalignment='top', bbox=props)

    layout_config = aesthetics.get('layout', {})
    if layout_config.get('tight_layout', True):
        plt.tight_layout(pad=layout_config.get('pad', 1.08))

    output_path = get_output_path('figures', 'heat_flux_vs_relative_position.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_figure(fig, output_path, aesthetics)
    plt.close(fig)
    print(f"Saved heat_flux_vs_relative_position.png to {output_path}")

def create_species_plots(gas, states1, config, reactant_info, aesthetics):
    """Create species and property plots using ``configs/style/figure_aesthetics.json``."""
    line_cfg = aesthetics.get('line', {})
    lw = line_cfg.get('width', 2)
    layout_cfg = aesthetics.get('layout', {})

    feed_species = reactant_info.get('feed_species', '')
    if ':' in feed_species:
        species_name = feed_species.split(':')[0]
    else:
        species_name = feed_species

    out_conv = get_output_path('figures', 'reactant_conversion.png')
    os.makedirs(os.path.dirname(out_conv), exist_ok=True)
    try:
        species_idx = gas.species_index(species_name)
        fig, _ = plot_profile(
            states1.z, states1.Y[:, species_idx], 'conversion',
            aesthetics=aesthetics,
            output_path=out_conv,
            ylabel=f'$Y_{{{reactant_info["name"]}}}$ [-]',
            title=f'{reactant_info["name"]} conversion',
            label=reactant_info['name'],
        )
        plt.close(fig)
    except Exception:
        fig = create_figure(aesthetics)
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, f'Could not plot {reactant_info["name"]} conversion',
            transform=ax.transAxes, ha='center',
        )
        if layout_cfg.get('tight_layout', True):
            plt.tight_layout(pad=layout_cfg.get('pad', 1.08))
        save_figure(fig, out_conv, aesthetics)
        plt.close(fig)
    print(f"Saved reactant_conversion.png to {out_conv}")

    target_products = reactant_info.get('target_products', [])
    product_names = reactant_info.get('product_names', [])
    prod_style = get_profile_style('products', aesthetics)
    colors = prod_style.get('colors') or [
        get_color('primary', aesthetics), get_color('secondary', aesthetics),
    ]

    def _save_product_fractions(use_mole: bool, filename: str):
        fig = create_figure(aesthetics)
        ax = fig.add_subplot(111)
        arr = states1.X if use_mole else states1.Y
        for i, product in enumerate(target_products):
            try:
                product_idx = gas.species_index(product)
                product_name = product_names[i] if i < len(product_names) else product
                ax.plot(
                    states1.z, arr[:, product_idx],
                    color=colors[i % len(colors)], linewidth=lw, label=product_name,
                )
            except Exception:
                continue
        ax.set_xlabel('$z$ [m]')
        if use_mole:
            ax.set_ylabel('$X_k$ [-]')
            ax.set_title('Major product formation (mole fractions)')
        else:
            ax.set_ylabel(prod_style['ylabel'])
            ax.set_title(prod_style['title'])
        setup_axes(ax, aesthetics)
        setup_legend(ax, aesthetics, loc='best', bbox_to_anchor=(1.05, 1))
        if layout_cfg.get('tight_layout', True):
            plt.tight_layout(pad=layout_cfg.get('pad', 1.08))
        path = get_output_path('figures', filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_figure(fig, path, aesthetics)
        plt.close(fig)
        print(f"Saved {filename} to {path}")

    _save_product_fractions(False, 'product_mass_fractions.png')
    _save_product_fractions(True, 'product_mole_fractions.png')

    def _save_z_profile(profile_key: str, y, filename: str, **kw):
        path = get_output_path('figures', filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig, _ = plot_profile(states1.z, y, profile_key, aesthetics=aesthetics, output_path=path, **kw)
        plt.close(fig)
        print(f"Saved {filename} to {path}")

    _save_z_profile('heat_capacity_cp', states1.cp, 'heat_capacity_cp.png')
    _save_z_profile('heat_capacity_cv', states1.cv, 'heat_capacity_cv.png')

    dz = np.diff(states1.z)
    dt = dz / states1.velocity[1:]
    residence_time = np.cumsum(np.concatenate([[0], dt]))
    _save_z_profile('residence_time', residence_time, 'residence_time.png')

    ratio_path = get_output_path('figures', 'heat_capacity_ratio.png')
    os.makedirs(os.path.dirname(ratio_path), exist_ok=True)
    try:
        cp_cv_ratio = states1.cp / states1.cv
        fig, _ = plot_profile(
            states1.z, cp_cv_ratio, 'heat_capacity_ratio',
            aesthetics=aesthetics, output_path=ratio_path,
            label='$C_p/C_v$',
        )
        plt.close(fig)
    except Exception:
        fig = create_figure(aesthetics)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Heat capacity ratio not available', transform=ax.transAxes, ha='center')
        if layout_cfg.get('tight_layout', True):
            plt.tight_layout(pad=layout_cfg.get('pad', 1.08))
        save_figure(fig, ratio_path, aesthetics)
        plt.close(fig)
    print(f"Saved heat_capacity_ratio.png to {ratio_path}")

    _save_z_profile('enthalpy', states1.h, 'enthalpy_profile.png')
    _save_z_profile('entropy', states1.s, 'entropy_profile.png')
    _save_z_profile('viscosity', states1.viscosity, 'viscosity_profile.png')
    _save_z_profile('thermal_conductivity', states1.thermal_conductivity, 'thermal_conductivity_profile.png')

    # Note: compressibility_factor not available in this Cantera version

def export_results(gas, states1, config, reactant_info, conversion, yields, T_0, p_0, u_0, hf):
    """Export results to CSV and summary files."""
    print("\nExporting results to CSV...")
    
    # Create results DataFrame with comprehensive Cantera data
    results_data = {
        # Basic properties
        'z_position_m': states1.z,
        'temperature_K': states1.T,
        'pressure_Pa': states1.P,
        'pressure_bar': states1.P / 1e5,
        'velocity_ms': states1.velocity,
        'density_kgm3': states1.density,
        'heat_flux_Wm2': [hf(z) for z in states1.z],
        
        # Thermodynamic properties
        'heat_capacity_cp_JkgK': states1.cp,
        'heat_capacity_cv_JkgK': states1.cv,
        'heat_capacity_ratio_cp_cv': states1.cp / states1.cv,
        'mean_molecular_weight_kgkmol': states1.mean_molecular_weight,
        'enthalpy_Jkg': states1.h,
        'entropy_JkgK': states1.s,
        'internal_energy_Jkg': states1.u,
        'gibbs_free_energy_Jkg': states1.g,
        
        # Transport properties
        'viscosity_Pas': states1.viscosity,
        'thermal_conductivity_WmK': states1.thermal_conductivity
    }
    
    # Note: compressibility_factor not available in this Cantera version
    
    # Note: Advanced reaction and transport properties not available in this Cantera version
    # Only basic properties are exported to ensure compatibility
    
    # Add all species mass fractions
    for i, species in enumerate(gas.species_names):
        results_data[f'Y_{species}'] = states1.Y[:, i]
        results_data[f'X_{species}'] = states1.X[:, i]
    
    df = pd.DataFrame(results_data)
    
    # Create systematic filename
    reactant_name = reactant_info['name'].replace(' ', '').replace('-', '')
    temp_str = f"{T_0:.0f}"
    pressure_str = f"{p_0/1e5:.1f}"
    length_str = f"{config['reactor_geometry']['length_m']:.1f}"
    diameter_str = f"{config['reactor_geometry']['diameter_m']*1000:.1f}"
    massflow_str = f"{config['operating_conditions']['mass_flow_rate_kgps']:.4f}"
    steps_str = f"{config['simulation_settings']['n_steps']}"
    
    csv_filename = get_output_path('results', f'results_{reactant_name}_T{temp_str}K_P{pressure_str}bar_L{length_str}m_D{diameter_str}mm_M{massflow_str}kgps_n{steps_str}.csv')
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df.to_csv(csv_filename, index=False)
    print(f"Saved complete results to {csv_filename}")
    
    # Create summary
    # Create summary file (always created when CSV export is enabled)
    create_summary_file(config, reactant_info, conversion, yields, states1, T_0, p_0, u_0, csv_filename, hf)

def create_summary_file(config, reactant_info, conversion, yields, states1, T_0, p_0, u_0, csv_filename, hf):
    """Create summary file with key results."""
    reactant_name = reactant_info['name'].replace(' ', '').replace('-', '')
    temp_str = f"{T_0:.0f}"
    pressure_str = f"{p_0/1e5:.1f}"
    length_str = f"{config['reactor_geometry']['length_m']:.1f}"
    diameter_str = f"{config['reactor_geometry']['diameter_m']*1000:.1f}"
    massflow_str = f"{config['operating_conditions']['mass_flow_rate_kgps']:.4f}"
    steps_str = f"{config['simulation_settings']['n_steps']}"
    
    summary_filename = get_output_path('results', f'summary_{reactant_name}_T{temp_str}K_P{pressure_str}bar_L{length_str}m_D{diameter_str}mm_M{massflow_str}kgps_n{steps_str}.dat')
    os.makedirs(os.path.dirname(summary_filename), exist_ok=True)
    
    with open(summary_filename, 'w') as f:
        f.write(f"# {reactant_info['name']} Pyrolysis PFR Simulation Summary\n")
        f.write("# =========================================\n")
        f.write(f"# Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Reactant: {reactant_info['name']} ({reactant_info['formula']})\n")
        f.write(f"# Mechanism File: {config['mechanism']['reaction_mechanism_file']}\n")
        f.write(f"# Description: {reactant_info['description']}\n")
        f.write("#\n")
        f.write("# SIMULATION CONFIGURATION\n")
        f.write("# ========================\n")
        f.write(f"# Initial Temperature: {T_0:.1f} K\n")
        f.write(f"# Initial Pressure: {p_0/1e5:.1f} bar\n")
        f.write(f"# Feed Composition: {config['initial_conditions']['composition']}\n")
        f.write(f"# Reactor Length: {config['reactor_geometry']['length_m']:.1f} m\n")
        f.write(f"# Reactor Diameter: {config['reactor_geometry']['diameter_m']*1000:.1f} mm\n")
        f.write(f"# Mass Flow Rate: {config['operating_conditions']['mass_flow_rate_kgps']:.4f} kg/s\n")
        f.write(f"# Number of Integration Steps: {config['simulation_settings']['n_steps']}\n")
        f.write(f"# Solver Tolerance: {config['simulation_settings'].get('solver_tolerance', 1e-6):.2e}\n")
        f.write(f"# Heat Flux File: {config['mechanism']['heat_flux_file']}\n")
        f.write(f"# Initial Heat Flux: {hf(0.0):.0f} W/m²\n")
        f.write(f"# Final Heat Flux: {hf(states1.z[-1]):.0f} W/m²\n")
        f.write(f"# Average Heat Flux: {np.mean([hf(z) for z in states1.z]):.0f} W/m²\n")
        f.write("#\n")
        f.write("# SIMULATION RESULTS\n")
        f.write("# ==================\n")
        f.write(f"# Final Temperature: {states1.T[-1]:.1f} K\n")
        f.write(f"# Temperature Rise: {states1.T[-1] - T_0:.1f} K\n")
        f.write(f"# Final Pressure: {states1.P[-1]/1e5:.2f} bar\n")
        f.write(f"# Pressure Drop: {(p_0 - states1.P[-1])/1e5:.2f} bar\n")
        f.write(f"# Residence Time: {_trapz(1/states1.velocity, states1.z):.3f} s\n")
        f.write("#\n")
        f.write("# CONVERSION AND YIELDS\n")
        f.write("# =====================\n")
        f.write(f"# {reactant_info['name']} Conversion: {conversion:.1f}%\n")
        
        for product, yield_val in yields.items():
            f.write(f"# {product} Yield: {yield_val:.2f}% (mass)\n")
        
        f.write("#\n")
        f.write("# OUTPUT FILES\n")
        f.write("# ============\n")
        f.write(f"# Main Results CSV: {csv_filename}\n")
        f.write(f"# Summary DAT: {summary_filename}\n")
        f.write("#\n")
        f.write("# END OF SUMMARY\n")
        f.write("# ==============\n")
    
    print(f"Saved simulation summary to {summary_filename}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generalized PFR Simulation for Multiple Reactants')
    parser.add_argument('--reactant', '-r', type=str, 
                       help='Reactant type (ethane, propane, naphtha, n-hexane)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available reactants')
    
    args = parser.parse_args()
    
    # Load reactant database
    database = load_reactant_database()
    
    if args.list:
        print("Available reactants:")
        for key, info in database['reactants'].items():
            print(f"  {key}: {info['name']} - {info['description']}")
        return
    
    # Determine reactant
    reactant_key = args.reactant or database['default_reactant']
    
    if reactant_key not in database['reactants']:
        print(f"Error: Reactant '{reactant_key}' not found.")
        print("Available reactants:", list(database['reactants'].keys()))
        return
    
    print(f"Running simulation for: {database['reactants'][reactant_key]['name']}")
    
    # Load configuration
    config, reactant_info = load_configuration(reactant_key)
    
    print(f"Loaded configuration for {config['simulation_info']['title']}")
    print(f"Version: {config['simulation_info']['version']}")
    
    # Setup mechanism
    gas = setup_mechanism(config)
    
    # Setup initial conditions
    T_0, p_0, composition_0, length, diameter, area, roughness, mass_flow_rate, u_0 = setup_initial_conditions(gas, config)
    
    # Setup heat flux
    hf, z_profile, heatflux_profile = setup_heat_flux(config)
    
    # Configuration validation
    print(f"Configuration validation:")
    print(f"  - Mechanism: {config['mechanism']['reaction_mechanism_file']}")
    print(f"  - Temperature: {T_0} K")
    print(f"  - Pressure: {p_0/1e5:.1f} bar")
    print(f"  - Composition: {composition_0}")
    print(f"  - Reactant: {reactant_info['name']}")
    
    # Run simulation
    states1 = run_simulation(gas, config, reactant_info, hf, T_0, p_0, length, diameter, area, roughness, mass_flow_rate, u_0)
    
    # Process results
    conversion, yields = process_and_visualize_results(gas, states1, config, reactant_info, hf, T_0, p_0, u_0)
    
    # Final summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"[OK] {reactant_info['name']} conversion: {conversion:.1f}%")
    print(f"[OK] Temperature rise: {states1.T[-1] - T_0:.1f} K")
    print(f"[OK] Pressure drop: {(p_0 - states1.P[-1])/1e5:.2f} bar")
    print(f"[OK] Residence time: {_trapz(1/states1.velocity, states1.z):.3f} s")
    print(f"[OK] Files generated:")
    print(f"  - fig/*.png: Visualization plots")
    print(f"  - results/*.csv: Simulation data")
    print(f"  - results/*.dat: Simulation summary")
    print("="*60)

if __name__ == "__main__":
    main()
