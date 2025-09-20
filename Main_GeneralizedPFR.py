#!/usr/bin/env python3
"""
Generalized Plug Flow Reactor (PFR) Simulation Workflow
=======================================================

A comprehensive simulation framework for modeling steam cracking reactions
in plug flow reactors using Cantera. Supports multiple feedstocks including
ethane, propane, naphtha, and n-hexane with automatic configuration generation
and species name handling.

Author: Nikolas Karefyllidis, PhD
Development Started: 2025-09-20
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
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

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
        If reactant_database.json is not found
    json.JSONDecodeError
        If the JSON file is malformed
    """
    with open('reactant_database.json', 'r') as f:
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
        If config_template.json is not found
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
    with open('config_template.json', 'r') as f:
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
    
    # Load mechanism to get species and reaction counts
    try:
        gas_temp = ct.Solution(reactant_info['mechanism_file'])
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
    
    heat_flux_file = config['mechanism']['heat_flux_file']
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
    """Create all visualization plots."""
    # Temperature profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.T, 'r-', linewidth=2, label='Temperature')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$T$ [K]')
    plt.title('Temperature Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/temperature_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved temperature profile to fig/temperature_profile.png")
    
    # Pressure profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.P/1e5, 'b-', linewidth=2, label='Pressure')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$p$ [bar]')
    plt.title('Pressure Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/pressure_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pressure profile to fig/pressure_profile.png")
    
    # Velocity profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.velocity, 'g-', linewidth=2, label='Velocity')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$u$ [m/s]')
    plt.title('Velocity Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/velocity_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved velocity profile to fig/velocity_profile.png")
    
    # Density profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.density, 'm-', linewidth=2, label='Density')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$\\rho$ [kg/m³]')
    plt.title('Density Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/density_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved density profile to fig/density_profile.png")
    
    # Heat flux profile
    plt.figure(figsize=(10, 6))
    heat_flux_values = [hf(z) for z in states1.z]
    plt.plot(states1.z, heat_flux_values, 'red', linewidth=5, label='Heat Flux')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$q$ [W/m²]')
    plt.title('Heat Flux Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/heat_flux_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved heat flux profile to fig/heat_flux_profile.png")
    
    # Molecular weight profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.mean_molecular_weight, 'c-', linewidth=2, label='MW')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$MW$ [kg/kmol]')
    plt.title('Mean Molecular Weight')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/molecular_weight_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved molecular weight profile to fig/molecular_weight_profile.png")
    
    # Heat flux vs relative position
    create_heat_flux_relative_figure(config, hf)
    
    # Species profiles
    create_species_plots(gas, states1, config, reactant_info)

def create_heat_flux_relative_figure(config, hf):
    """Create heat flux vs relative position figure."""
    import json
    
    # Load the heat flux profile to get the original data points
    heat_flux_file = config['mechanism']['heat_flux_file']
    with open(heat_flux_file, 'r') as f:
        heat_flux_data = json.load(f)
    
    # Extract data points and interpolation method
    data_points = heat_flux_data['heat_flux_profile']['data_points']
    interpolation_method = heat_flux_data['heat_flux_profile'].get('interpolation_method', 'linear')
    
    z_profile_relative = np.array([point['position'] for point in data_points])
    heatflux_profile = np.array([point['heat_flux'] for point in data_points])
    
    # Create fine resolution for smooth plotting
    z_fine = np.linspace(0.0, 1.0, 1000)
    
    # Calculate heat flux values for the current interpolation method
    if interpolation_method == 'step':
        # Step-wise interpolation
        heat_flux_fine = np.zeros_like(z_fine)
        for i, z in enumerate(z_fine):
            idx = np.searchsorted(z_profile_relative, z, side='right')
            idx = max(0, min(idx - 1, len(heatflux_profile) - 1))
            heat_flux_fine[i] = heatflux_profile[idx]
    else:
        # Linear interpolation
        heat_flux_fine = np.interp(z_fine, z_profile_relative, heatflux_profile)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot the interpolated curve
    plt.plot(z_fine, heat_flux_fine, 'b-', linewidth=2, 
             label=f'{interpolation_method.capitalize()} interpolation')
    
    # Plot the data points
    plt.plot(z_profile_relative, heatflux_profile, 'rs', markersize=8, 
             label='Data points', markeredgecolor='darkred', markeredgewidth=1)
    
    # Customize the plot
    plt.xlabel('Relative Position', fontsize=14)
    plt.ylabel('Heat Flux [W/m²]', fontsize=14)
    plt.title('Heat Flux Profile vs Relative Position', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set axis limits and ticks
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add text box with interpolation method info
    textstr = f'Interpolation Method: {interpolation_method.capitalize()}\nData Points: {len(data_points)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('fig/heat_flux_vs_relative_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved heat flux vs relative position to fig/heat_flux_vs_relative_position.png")

def create_species_plots(gas, states1, config, reactant_info):
    """Create comprehensive species concentration plots."""
    # Reactant conversion
    plt.figure(figsize=(10, 6))
    feed_species = reactant_info.get('feed_species', '')
    # Extract species name from composition string (remove ratio part)
    if ':' in feed_species:
        species_name = feed_species.split(':')[0]
    else:
        species_name = feed_species
    
    try:
        species_idx = gas.species_index(species_name)
        plt.plot(states1.z, states1.Y[:, species_idx], 'r-', linewidth=2, 
                label=f'{reactant_info["name"]}')
        plt.xlabel('$z$ [m]')
        plt.ylabel(f'$Y_{{{reactant_info["name"]}}}$ [-]')
        plt.title(f'{reactant_info["name"]} Conversion')
        plt.legend(loc=0)
        plt.xlim(left=0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fig/reactant_conversion.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved reactant conversion to fig/reactant_conversion.png")
    except:
        plt.text(0.5, 0.5, f'Could not plot {reactant_info["name"]} conversion', 
                transform=plt.gca().transAxes, ha='center')
        plt.tight_layout()
        plt.savefig('fig/reactant_conversion.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved reactant conversion to fig/reactant_conversion.png")
    
    # Major products (mass fractions)
    plt.figure(figsize=(12, 8))
    target_products = reactant_info.get('target_products', [])
    product_names = reactant_info.get('product_names', [])
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, product in enumerate(target_products):  # Plot all products
        try:
            product_idx = gas.species_index(product)
            product_name = product_names[i] if i < len(product_names) else product
            plt.plot(states1.z, states1.Y[:, product_idx], 
                    color=colors[i % len(colors)], linewidth=2, label=product_name)
        except:
            continue
    
    plt.xlabel('$z$ [m]')
    plt.ylabel('$Y_k$ [-]')
    plt.title('Major Product Formation (Mass Fractions)')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/product_mass_fractions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved product mass fractions to fig/product_mass_fractions.png")
    
    # Major products (mole fractions)
    plt.figure(figsize=(12, 8))
    for i, product in enumerate(target_products):  # Plot all products
        try:
            product_idx = gas.species_index(product)
            product_name = product_names[i] if i < len(product_names) else product
            plt.plot(states1.z, states1.X[:, product_idx], 
                    color=colors[i % len(colors)], linewidth=2, label=product_name)
        except:
            continue
    
    plt.xlabel('$z$ [m]')
    plt.ylabel('$X_k$ [-]')
    plt.title('Major Product Formation (Mole Fractions)')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/product_mole_fractions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved product mole fractions to fig/product_mole_fractions.png")
    
    # Mean heat capacity at constant pressure (Cp)
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.cp, 'purple', linewidth=2, label='Cp [J/kg·K]')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$C_p$ [J/kg·K]')
    plt.title('Mean Heat Capacity at Constant Pressure')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/heat_capacity_cp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved heat capacity Cp to fig/heat_capacity_cp.png")
    
    # Mean heat capacity at constant volume (Cv)
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.cv, 'orange', linewidth=2, label='Cv [J/kg·K]')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$C_v$ [J/kg·K]')
    plt.title('Mean Heat Capacity at Constant Volume')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/heat_capacity_cv.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved heat capacity Cv to fig/heat_capacity_cv.png")
    
    # Residence time
    plt.figure(figsize=(10, 6))
    # Calculate cumulative residence time
    dz = np.diff(states1.z)
    dt = dz / states1.velocity[1:]
    residence_time = np.cumsum(np.concatenate([[0], dt]))
    plt.plot(states1.z, residence_time, 'brown', linewidth=2, label='Residence Time')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$t$ [s]')
    plt.title('Cumulative Residence Time')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/residence_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved residence time to fig/residence_time.png")
    
    # Heat capacity ratio (Cp/Cv)
    plt.figure(figsize=(10, 6))
    try:
        cp_cv_ratio = states1.cp / states1.cv
        plt.plot(states1.z, cp_cv_ratio, 'purple', linewidth=2, label='$C_p/C_v$')
        plt.xlabel('$z$ [m]')
        plt.ylabel('$C_p/C_v$ [-]')
        plt.title('Heat Capacity Ratio')
        plt.legend(loc=0)
        plt.xlim(left=0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fig/heat_capacity_ratio.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved heat capacity ratio to fig/heat_capacity_ratio.png")
    except:
        plt.text(0.5, 0.5, 'Heat capacity ratio not available', 
                transform=plt.gca().transAxes, ha='center')
        plt.tight_layout()
        plt.savefig('fig/heat_capacity_ratio.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved heat capacity ratio to fig/heat_capacity_ratio.png")
    
    # Enthalpy profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.h, 'darkgreen', linewidth=2, label='Enthalpy')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$h$ [J/kg]')
    plt.title('Specific Enthalpy Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/enthalpy_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved enthalpy profile to fig/enthalpy_profile.png")
    
    # Entropy profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.s, 'darkred', linewidth=2, label='Entropy')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$s$ [J/kg·K]')
    plt.title('Specific Entropy Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/entropy_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved entropy profile to fig/entropy_profile.png")
    
    # Viscosity profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.viscosity, 'navy', linewidth=2, label='Viscosity')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$\\mu$ [Pa·s]')
    plt.title('Dynamic Viscosity Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/viscosity_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved viscosity profile to fig/viscosity_profile.png")
    
    # Thermal conductivity profile
    plt.figure(figsize=(10, 6))
    plt.plot(states1.z, states1.thermal_conductivity, 'teal', linewidth=2, label='Thermal Conductivity')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$k$ [W/m·K]')
    plt.title('Thermal Conductivity Profile')
    plt.legend(loc=0)
    plt.xlim(left=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig/thermal_conductivity_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved thermal conductivity profile to fig/thermal_conductivity_profile.png")
    
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
    
    csv_filename = f'results/results_{reactant_name}_T{temp_str}K_P{pressure_str}bar_L{length_str}m_D{diameter_str}mm_M{massflow_str}kgps_n{steps_str}.csv'
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
    
    summary_filename = f'results/summary_{reactant_name}_T{temp_str}K_P{pressure_str}bar_L{length_str}m_D{diameter_str}mm_M{massflow_str}kgps_n{steps_str}.dat'
    
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
        f.write(f"# Residence Time: {np.trapezoid(1/states1.velocity, states1.z):.3f} s\n")
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
    print(f"✓ {reactant_info['name']} conversion: {conversion:.1f}%")
    print(f"✓ Temperature rise: {states1.T[-1] - T_0:.1f} K")
    print(f"✓ Pressure drop: {(p_0 - states1.P[-1])/1e5:.2f} bar")
    print(f"✓ Residence time: {np.trapezoid(1/states1.velocity, states1.z):.3f} s")
    print(f"✓ Files generated:")
    print(f"  - fig/*.png: Visualization plots")
    print(f"  - results/*.csv: Simulation data")
    print(f"  - results/*.dat: Simulation summary")
    print("="*60)

if __name__ == "__main__":
    main()
