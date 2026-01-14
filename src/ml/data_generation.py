#!/usr/bin/env python3
"""
Training Data Generation for ML Surrogate Models
=================================================

This script generates a massive training dataset by running multiple PFR simulations
with varied parameters. The data will be used to train ML models to replace Cantera.

Author: Nikolas Karefyllidis, PhD
ML Surrogate Models Module
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
import time
from pathlib import Path
import warnings
import logging

# IMPORTANT: Import cantera BEFORE adding src to sys.path
# This prevents namespace conflict: without this, Python would find src/cantera/ 
# (our package) instead of the actual cantera library when pfr_simulator.py imports it
import cantera as ct

# Suppress all Cantera/SUNDIALS warnings and messages
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*rank.*')
warnings.filterwarnings('ignore', message='.*CVode.*')
warnings.filterwarnings('ignore', message='.*SUNDIALS.*')
warnings.filterwarnings('ignore', message='.*WARNING.*')
logging.getLogger('cantera').setLevel(logging.CRITICAL)
logging.getLogger('sundials').setLevel(logging.CRITICAL)

# Add project root to path (after cantera is imported)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.cantera.pfr_simulator import (
    load_reactant_database,
    generate_config_for_reactant,
    setup_mechanism,
    setup_initial_conditions,
    setup_heat_flux,
    run_simulation,
    calculate_conversion,
    calculate_product_yields
)

class TrainingDataGenerator:
    """Generate training data by running parameter sweeps."""
    
    def __init__(self, output_dir='data/training', disable_plots=True):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save training data
        disable_plots : bool
            Disable plot generation for faster data collection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.disable_plots = disable_plots
        
        # Create temporary directory for heat flux files
        self.temp_dir = Path('temp')  # Temporary files directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load reactant database
        self.database = load_reactant_database()
        
        # Parameter ranges for data generation
        self.param_ranges = {
            'temperature_K': np.linspace(800, 1200, 10),  # 10 points
            'pressure_bar': np.linspace(1.5, 3.0, 8),    # 8 points
            'length_m': np.linspace(3.0, 7.0, 6),        # 6 points
            'diameter_mm': np.linspace(20.0, 40.0, 5),    # 5 points
            'mass_flow_rate_kgps': np.linspace(0.05, 0.10, 6),  # 6 points
            'heat_flux_Wm2': np.linspace(100000, 200000, 5)  # 5 points
        }
        
        print(f"Training data will be saved to: {self.output_dir}")
        print(f"Total parameter combinations: {self._calculate_total_combinations()}")
    
    def _calculate_total_combinations(self):
        """Calculate total number of parameter combinations."""
        total = 1
        for key, values in self.param_ranges.items():
            total *= len(values)
        return total * len(self.database['reactants'])
    
    def generate_parameter_combinations(self, max_combinations=None, random_sample=True, 
                                         random_sample_bounds=None):
        """
        Generate parameter combinations for training data.
        
        Parameters:
        -----------
        max_combinations : int, optional
            Maximum number of combinations to generate (for random sampling)
        random_sample : bool
            If True, randomly sample combinations; if False, use all combinations
        random_sample_bounds : dict, optional
            Bounds for random sampling. Format: {"param_name": [min, max], ...}
            If provided, random sampling will only generate values within these bounds.
            Example: {"temperature_K": [850, 1200], "pressure_bar": [2.0, 3.0]}
        
        Returns:
        --------
        list
            List of parameter dictionaries
        """
        # Generate all combinations
        keys = list(self.param_ranges.keys())
        values = [self.param_ranges[key] for key in keys]
        
        all_combinations = list(product(*values))
        
        # Convert to dictionaries and apply bounds filtering
        param_dicts = []
        for combo in all_combinations:
            param_dict = dict(zip(keys, combo))
            
            # Apply bounds filtering if provided (check BEFORE unit conversion)
            if random_sample_bounds:
                valid = True
                for param_name, bounds in random_sample_bounds.items():
                    # Check against original parameter names (before conversion)
                    if param_name in param_dict:
                        value = param_dict[param_name]
                        if value < bounds[0] or value > bounds[1]:
                            valid = False
                            break
                if not valid:
                    continue  # Skip this combination
            
            # Convert units (after bounds check)
            if 'diameter_mm' in param_dict:
                param_dict['diameter_m'] = param_dict.pop('diameter_mm') / 1000.0
            if 'pressure_bar' in param_dict:
                param_dict['pressure_Pa'] = param_dict.pop('pressure_bar') * 1e5
            
            param_dicts.append(param_dict)
        
        # Sample if requested
        if random_sample and max_combinations and len(param_dicts) > max_combinations:
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(param_dicts), max_combinations, replace=False)
            param_dicts = [param_dicts[i] for i in indices]
        
        return param_dicts
    
    def create_config_from_params(self, reactant_key, params):
        """
        Create a configuration dictionary from parameters.
        
        Parameters:
        -----------
        reactant_key : str
            Reactant identifier
        params : dict
            Parameter dictionary
        
        Returns:
        --------
        dict
            Configuration dictionary
        """
        # Generate base config
        config = generate_config_for_reactant(reactant_key, self.database)
        
        # Override with provided parameters
        config['initial_conditions']['temperature_K'] = params['temperature_K']
        config['initial_conditions']['pressure_Pa'] = params['pressure_Pa']
        config['reactor_geometry']['length_m'] = params['length_m']
        config['reactor_geometry']['diameter_m'] = params['diameter_m']
        config['operating_conditions']['mass_flow_rate_kgps'] = params['mass_flow_rate_kgps']
        
        # Disable plots and optionally CSV for faster generation
        config['export_controls']['if_plot_out'] = 0
        config['export_controls']['if_csv_out'] = 0  # We'll collect data manually
        
        # Update heat flux profile
        self._update_heat_flux(config, params['heat_flux_Wm2'])
        
        return config
    
    def _update_heat_flux(self, config, heat_flux_value):
        """Update heat flux profile with new value."""
        heat_flux_file = config['mechanism']['heat_flux_file']
        with open(heat_flux_file, 'r') as f:
            heat_flux_data = json.load(f)
        
        # Update all data points with new heat flux
        for point in heat_flux_data['heat_flux_profile']['data_points']:
            point['heat_flux'] = heat_flux_value
        
        # Save temporary heat flux file in dedicated temp directory
        timestamp = int(time.time())
        temp_heat_flux_file = str(self.temp_dir / f'heat_flux_{timestamp}.json')
        with open(temp_heat_flux_file, 'w') as f:
            json.dump(heat_flux_data, f, indent=2)
        
        config['mechanism']['heat_flux_file'] = temp_heat_flux_file
        return temp_heat_flux_file
    
    def run_single_simulation(self, reactant_key, params, sim_id):
        """
        Run a single simulation and collect data.
        
        Parameters:
        -----------
        reactant_key : str
            Reactant identifier
        params : dict
            Parameter dictionary
        sim_id : int
            Simulation ID
        
        Returns:
        --------
        pd.DataFrame or None
            Training data DataFrame, or None if simulation failed
        """
        try:
            print(f"\n[{sim_id}] Running simulation for {reactant_key}...")
            print(f"  T={params['temperature_K']:.1f}K, P={params['pressure_Pa']/1e5:.2f}bar, "
                  f"L={params['length_m']:.1f}m, D={params['diameter_m']*1000:.1f}mm, "
                  f"m={params['mass_flow_rate_kgps']:.4f}kg/s, q={params['heat_flux_Wm2']:.0f}W/m²")
            
            # Validate parameters before running simulation
            if params['pressure_Pa'] <= 0:
                print(f"  [SKIP] Invalid pressure: {params['pressure_Pa']} Pa")
                return None
            if params['temperature_K'] <= 0:
                print(f"  [SKIP] Invalid temperature: {params['temperature_K']} K")
                return None
            if params['mass_flow_rate_kgps'] <= 0:
                print(f"  [SKIP] Invalid mass flow rate: {params['mass_flow_rate_kgps']} kg/s")
                return None
            
            # Create configuration
            config = self.create_config_from_params(reactant_key, params)
            reactant_info = self.database['reactants'][reactant_key]
            
            # Setup mechanism
            gas = setup_mechanism(config)
            
            # Setup initial conditions
            T_0, p_0, composition_0, length, diameter, area, roughness, mass_flow_rate, u_0 = \
                setup_initial_conditions(gas, config)
            
            # Setup heat flux
            hf, z_profile, heatflux_profile = setup_heat_flux(config)
            
            # Run simulation
            states = run_simulation(gas, config, reactant_info, hf, T_0, p_0, 
                                   length, diameter, area, roughness, mass_flow_rate, u_0)
            
            # Collect data
            training_data = self._collect_training_data(
                gas, states, config, reactant_info, params, T_0, p_0, u_0, hf
            )
            
            # Clean up temporary heat flux file
            temp_file = config['mechanism']['heat_flux_file']
            if os.path.exists(temp_file):
                # Check if it's a temp file (in temp/ directory or has temp_ prefix)
                if 'temp' in os.path.dirname(temp_file) or 'heat_flux_' in os.path.basename(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass  # Ignore cleanup errors
            
            print(f"  [OK] Simulation completed: {len(training_data)} data points")
            return training_data
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Handle specific Cantera errors gracefully
            if 'CanteraError' in error_type or 'Cantera' in error_msg:
                if 'density must be positive' in error_msg:
                    print(f"  [SKIP] Physical constraint violation: negative density")
                    print(f"         This usually indicates unrealistic parameter combination")
                    print(f"         (e.g., too high heat flux, pressure drop, or flow rate)")
                elif 'convergence' in error_msg.lower() or 'solver' in error_msg.lower():
                    print(f"  [SKIP] Solver convergence failure")
                else:
                    print(f"  [SKIP] Cantera error: {error_msg[:150]}")
            else:
                # For other errors, show more detail but don't print full traceback
                print(f"  [SKIP] Simulation error ({error_type}): {error_msg[:150]}")
            
            return None
    
    def _collect_training_data(self, gas, states, config, reactant_info, params, T_0, p_0, u_0, hf):
        """
        Collect training data from simulation results.
        
        Returns:
        --------
        pd.DataFrame
            Training data with features and targets
        """
        n_points = len(states.z)
        
        # Initialize data dictionary
        data = {
            # Input features (will be used to predict outputs)
            'reactant_type': [reactant_info['name']] * n_points,
            'initial_temperature_K': [T_0] * n_points,
            'initial_pressure_Pa': [p_0] * n_points,
            'reactor_length_m': [config['reactor_geometry']['length_m']] * n_points,
            'reactor_diameter_m': [config['reactor_geometry']['diameter_m']] * n_points,
            'mass_flow_rate_kgps': [config['operating_conditions']['mass_flow_rate_kgps']] * n_points,
            'heat_flux_Wm2': [params['heat_flux_Wm2']] * n_points,
            'z_position_m': states.z,
            'relative_position': states.z / config['reactor_geometry']['length_m'],
            
            # Target outputs (what we want to predict)
            'temperature_K': states.T,
            'pressure_Pa': states.P,
            'velocity_ms': states.velocity,
            'density_kgm3': states.density,
            'heat_capacity_cp_JkgK': states.cp,
            'heat_capacity_cv_JkgK': states.cv,
            'mean_molecular_weight_kgkmol': states.mean_molecular_weight,
            'enthalpy_Jkg': states.h,
            'entropy_JkgK': states.s,
            'internal_energy_Jkg': states.u,
            'gibbs_free_energy_Jkg': states.g,
            'viscosity_Pas': states.viscosity,
            'thermal_conductivity_WmK': states.thermal_conductivity,
        }
        
        # Add all species mass and mole fractions
        for i, species in enumerate(gas.species_names):
            data[f'Y_{species}'] = states.Y[:, i]
            data[f'X_{species}'] = states.X[:, i]
        
        return pd.DataFrame(data)
    
    def generate_dataset(self, reactants=None, max_combinations_per_reactant=100, 
                        random_sample=True, save_interval=10, random_sample_bounds=None):
        """
        Generate complete training dataset.
        
        Parameters:
        -----------
        reactants : list, optional
            List of reactant keys to use. If None, use all reactants.
        max_combinations_per_reactant : int
            Maximum parameter combinations per reactant
        random_sample : bool
            Randomly sample parameter combinations
        save_interval : int
            Save data every N simulations
        
        Returns:
        --------
        pd.DataFrame
            Complete training dataset
        """
        if reactants is None:
            reactants = list(self.database['reactants'].keys())
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(
            max_combinations=max_combinations_per_reactant,
            random_sample=random_sample,
            random_sample_bounds=random_sample_bounds
        )
        
        print(f"\nGenerating training data for {len(reactants)} reactants")
        print(f"  {len(param_combinations)} parameter combinations per reactant")
        print(f"  Total simulations: {len(reactants) * len(param_combinations)}")
        
        all_data = []
        sim_id = 0
        failed_simulations = 0
        start_time = time.time()
        
        for reactant_key in reactants:
            print(f"\n{'='*60}")
            print(f"Processing reactant: {reactant_key}")
            print(f"{'='*60}")
            
            for params in param_combinations:
                sim_id += 1
                training_data = self.run_single_simulation(reactant_key, params, sim_id)
                
                if training_data is not None:
                    all_data.append(training_data)
                else:
                    failed_simulations += 1
                
                # Save periodically
                if sim_id % save_interval == 0:
                    if all_data:
                        combined_data = pd.concat(all_data, ignore_index=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = self.output_dir / f'training_data_partial_{timestamp}.csv'
                        combined_data.to_csv(filename, index=False)
                        print(f"\n  [SAVED] Partial data saved: {filename} ({len(combined_data)} rows)")
                
                # Progress update
                elapsed = time.time() - start_time
                if sim_id > 0:
                    avg_time = elapsed / sim_id
                    remaining = (len(reactants) * len(param_combinations) - sim_id) * avg_time
                    success_rate = 100 * (sim_id - failed_simulations) / sim_id if sim_id > 0 else 0
                    print(f"  Progress: {sim_id}/{len(reactants) * len(param_combinations)} "
                          f"({100*sim_id/(len(reactants)*len(param_combinations)):.1f}%) | "
                          f"Success: {success_rate:.1f}% | "
                          f"ETA: {remaining/60:.1f} min")
        
        # Combine all data
        if all_data:
            print(f"\n{'='*60}")
            print("Combining all training data...")
            complete_dataset = pd.concat(all_data, ignore_index=True)
            
            # Save final dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.output_dir / f'training_data_complete_{timestamp}.csv'
            complete_dataset.to_csv(filename, index=False)
            
            print(f"[OK] Complete dataset saved: {filename}")
            print(f"  Total rows: {len(complete_dataset):,}")
            print(f"  Total columns: {len(complete_dataset.columns)}")
            print(f"  File size: {os.path.getsize(filename) / 1e6:.2f} MB")
            if failed_simulations > 0:
                print(f"  Successful simulations: {sim_id - failed_simulations}/{sim_id}")
                print(f"  Failed simulations: {failed_simulations} ({100*failed_simulations/sim_id:.1f}%)")
                print(f"  Note: Some parameter combinations may be physically unrealistic")
            
            # Save metadata
            metadata = {
                'generation_date': datetime.now().isoformat(),
                'total_rows': len(complete_dataset),
                'total_columns': len(complete_dataset.columns),
                'reactants': reactants,
                'parameter_ranges': {k: [float(v.min()), float(v.max())] 
                                    for k, v in self.param_ranges.items()},
                'n_combinations_per_reactant': len(param_combinations),
                'total_simulations': sim_id,
                'successful_simulations': sim_id - failed_simulations,
                'failed_simulations': failed_simulations,
                'success_rate': 100 * (sim_id - failed_simulations) / sim_id if sim_id > 0 else 0
            }
            
            metadata_file = self.output_dir / f'metadata_{timestamp}.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[OK] Metadata saved: {metadata_file}")
            
            return complete_dataset
        else:
            print("[ERROR] No data collected!")
            return None


def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_generation.py <config_file.json>")
        print("Example: python data_generation.py configs/ml_data_generation_config.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load configuration from JSON
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    reactants = config.get('reactants', None)
    max_combinations = config.get('max_combinations_per_reactant', 100)
    output_dir = config.get('output_dir', 'data/training')
    save_interval = config.get('save_interval', 10)
    random_sample = config.get('random_sample', True)
    random_sample_bounds = config.get('random_sample_bounds', None)
    
    # Create generator
    generator = TrainingDataGenerator(output_dir=output_dir)
    
    # Update parameter ranges if provided in config
    if 'parameter_ranges' in config:
        param_ranges = config['parameter_ranges']
        for key, value in param_ranges.items():
            if isinstance(value, list) and len(value) == 3:
                # Format: [min, max, n_points] - convert to numpy array
                if key in generator.param_ranges:
                    generator.param_ranges[key] = np.linspace(value[0], value[1], value[2])
    
    # Generate dataset
    dataset = generator.generate_dataset(
        reactants=reactants,
        max_combinations_per_reactant=max_combinations,
        random_sample=random_sample,
        save_interval=save_interval,
        random_sample_bounds=random_sample_bounds
    )
    
    if dataset is not None:
        print(f"\n{'='*60}")
        print("TRAINING DATA GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Features: {len([c for c in dataset.columns if c not in ['temperature_K', 'pressure_Pa', 'velocity_ms', 'density_kgm3']])}")
        print(f"Targets: Multiple (temperature, pressure, species, etc.)")


if __name__ == "__main__":
    main()
