#!/usr/bin/env python3
"""
Basic Usage Examples for Generalized PFR Simulation System
==========================================================

This script demonstrates basic usage patterns for the generalized PFR simulation
system. It shows how to run simulations, access results, and perform basic
analysis.

Author: Chemical Engineering Simulation Team
Version: 2.0
Date: 2025-01-15
License: MIT

Usage:
------
    python examples/basic_usage.py

Requirements:
-------------
- All dependencies from requirements.txt
- Main_GeneralizedPFR.py in the parent directory
- reactant_database.json in the parent directory
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Main_GeneralizedPFR import (
    load_reactant_database,
    generate_config_for_reactant,
    setup_mechanism,
    setup_initial_conditions,
    setup_heat_flux,
    run_simulation,
    process_and_visualize_results
)

def example_1_list_reactants():
    """
    Example 1: List all available reactants
    """
    print("=" * 60)
    print("EXAMPLE 1: List Available Reactants")
    print("=" * 60)
    
    try:
        database = load_reactant_database()
        print("Available reactants:")
        print("-" * 50)
        for key, info in database['reactants'].items():
            print(f"{key:15} : {info['name']:20} - {info['description']}")
        print("-" * 50)
        print(f"Default reactant: {database['default_reactant']}")
        
    except Exception as e:
        print(f"Error loading database: {e}")

def example_2_generate_config():
    """
    Example 2: Generate configuration for a specific reactant
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Generate Configuration")
    print("=" * 60)
    
    try:
        database = load_reactant_database()
        config = generate_config_for_reactant('ethane', database)
        
        print("Generated configuration for ethane:")
        print(f"  Title: {config['simulation_info']['title']}")
        print(f"  Mechanism: {config['mechanism']['reaction_mechanism_file']}")
        print(f"  Temperature: {config['initial_conditions']['temperature_K']} K")
        print(f"  Pressure: {config['initial_conditions']['pressure_bar']} bar")
        print(f"  Composition: {config['initial_conditions']['composition']}")
        print(f"  Reactor Length: {config['reactor_geometry']['length_m']} m")
        print(f"  Reactor Diameter: {config['reactor_geometry']['diameter_mm']} mm")
        
    except Exception as e:
        print(f"Error generating configuration: {e}")

def example_3_run_simulation():
    """
    Example 3: Run a complete simulation
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Run Complete Simulation")
    print("=" * 60)
    
    try:
        # Load database and generate configuration
        database = load_reactant_database()
        config = generate_config_for_reactant('ethane', database)
        reactant_info = database['reactants']['ethane']
        
        # Setup mechanism and initial conditions
        gas = setup_mechanism(config)
        T_0, p_0, composition_0, length, diam, area, roughness, mass_flow_rate, u_0 = setup_initial_conditions(gas, config)
        hf, z_profile, heatflux_profile = setup_heat_flux(config)
        
        print(f"Setup complete for {reactant_info['name']} simulation:")
        print(f"  Species: {gas.n_species}")
        print(f"  Reactions: {gas.n_reactions}")
        print(f"  Initial temperature: {T_0} K")
        print(f"  Initial pressure: {p_0/1e5:.1f} bar")
        print(f"  Reactor length: {length} m")
        print(f"  Mass flow rate: {mass_flow_rate:.4f} kg/s")
        
        # Note: Full simulation would be run here
        print("  (Simulation execution skipped in this example)")
        
    except Exception as e:
        print(f"Error in simulation setup: {e}")

def example_4_analyze_results():
    """
    Example 4: Analyze simulation results (if available)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Analyze Results")
    print("=" * 60)
    
    # Look for existing result files
    results_dir = "../results"
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
        
        if csv_files:
            # Use the most recent file
            latest_file = sorted(csv_files)[-1]
            file_path = os.path.join(results_dir, latest_file)
            
            try:
                # Load and analyze data
                df = pd.read_csv(file_path)
                
                print(f"Analyzing results from: {latest_file}")
                print(f"  Data points: {len(df)}")
                print(f"  Axial range: {df['z_position_m'].min():.2f} - {df['z_position_m'].max():.2f} m")
                print(f"  Temperature range: {df['temperature_K'].min():.1f} - {df['temperature_K'].max():.1f} K")
                print(f"  Pressure range: {df['pressure_bar'].min():.2f} - {df['pressure_bar'].max():.2f} bar")
                
                # Calculate some key metrics
                temp_rise = df['temperature_K'].max() - df['temperature_K'].min()
                pressure_drop = df['pressure_bar'].min() - df['pressure_bar'].max()
                
                print(f"  Temperature rise: {temp_rise:.1f} K")
                print(f"  Pressure drop: {pressure_drop:.2f} bar")
                
            except Exception as e:
                print(f"Error analyzing results: {e}")
        else:
            print("No CSV result files found. Run a simulation first.")
    else:
        print("Results directory not found. Run a simulation first.")

def example_5_batch_processing():
    """
    Example 5: Batch processing multiple reactants
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Batch Processing")
    print("=" * 60)
    
    try:
        database = load_reactant_database()
        reactants_to_test = ['ethane', 'propane']  # Limit for demo
        
        print("Batch processing example:")
        for reactant in reactants_to_test:
            if reactant in database['reactants']:
                info = database['reactants'][reactant]
                print(f"  - {reactant}: {info['name']} ({info['formula']})")
                print(f"    Description: {info['description']}")
                print(f"    Mechanism: {info['mechanism_file']}")
                print(f"    Target products: {', '.join(info['target_products'][:3])}...")
                print()
        
        print("(Actual batch processing would run simulations for each reactant)")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")

def main():
    """
    Main function to run all examples
    """
    print("Generalized PFR Simulation System - Basic Usage Examples")
    print("=" * 60)
    
    # Run all examples
    example_1_list_reactants()
    example_2_generate_config()
    example_3_run_simulation()
    example_4_analyze_results()
    example_5_batch_processing()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)
    print("For more advanced usage, see:")
    print("- README.md for comprehensive documentation")
    print("- Main_GeneralizedPFR.py for API reference")
    print("- examples/ directory for more examples")

if __name__ == "__main__":
    main()
