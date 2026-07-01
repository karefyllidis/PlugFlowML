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
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
from scipy.stats import qmc

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

from src.ml.dataframe_pickle import save_dataframe_pickle, load_dataframe_pickle
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


def _params_to_conditions_log_row(params):
    """Map generator param dict to SLURM conditions-log column names (pressure_bar, diameter_mm)."""
    row = {
        "temperature_K": params.get("temperature_K"),
        "pressure_bar": params.get("pressure_bar"),
        "length_m": params.get("length_m"),
        "diameter_mm": params.get("diameter_mm"),
        "mass_flow_rate_kgps": params.get("mass_flow_rate_kgps"),
        "heat_flux_Wm2": params.get("heat_flux_Wm2"),
    }
    if row["pressure_bar"] is None and params.get("pressure_Pa") is not None:
        row["pressure_bar"] = float(params["pressure_Pa"]) / 1e5
    if row["diameter_mm"] is None and params.get("diameter_m") is not None:
        row["diameter_mm"] = float(params["diameter_m"]) * 1000.0
    return row


def _write_generation_progress(
    path,
    task_id,
    ntasks,
    completed,
    total,
    successful,
    failed,
    elapsed_s,
):
    """Atomic JSON status for monitoring (e.g. tail/monitor on HPC)."""
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pct = (100.0 * completed / total) if total else 0.0
        data = {
            "task_id": task_id,
            "ntasks": ntasks,
            "completed": completed,
            "total_this_task": total,
            "successful": successful,
            "failed": failed,
            "percent_this_task": round(pct, 2),
            "elapsed_s": round(elapsed_s, 1),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(p)
    except Exception:
        pass


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
    
    def generate_parameter_combinations_lhs(self, n_samples, random_sample_bounds=None, seed=42):
        """
        Generate parameter combinations using Latin Hypercube Sampling (LHS).
        LHS gives better coverage of the parameter space than random sampling.
        
        Parameters:
        -----------
        n_samples : int
            Number of parameter combinations to generate.
        random_sample_bounds : dict, optional
            Override bounds per parameter. Format: {"param_name": [min, max], ...}
            If not provided, uses min/max from self.param_ranges.
        seed : int
            Random seed for reproducibility.
        
        Returns:
        --------
        list
            List of parameter dictionaries (same format as generate_parameter_combinations).
        """
        keys = list(self.param_ranges.keys())
        dim = len(keys)
        
        # Bounds for each parameter [min, max]
        bounds = np.zeros((dim, 2))
        for i, key in enumerate(keys):
            if random_sample_bounds and key in random_sample_bounds:
                bounds[i, 0] = random_sample_bounds[key][0]
                bounds[i, 1] = random_sample_bounds[key][1]
            else:
                arr = self.param_ranges[key]
                bounds[i, 0] = float(np.min(arr))
                bounds[i, 1] = float(np.max(arr))
        
        # Latin Hypercube sample in [0, 1]^d
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        u = sampler.random(n=n_samples)
        # Scale to parameter bounds
        samples = qmc.scale(u, bounds[:, 0], bounds[:, 1])
        
        param_dicts = []
        for row in samples:
            param_dict = dict(zip(keys, row))
            # Unit conversions (same as generate_parameter_combinations)
            if 'diameter_mm' in param_dict:
                param_dict['diameter_m'] = param_dict.pop('diameter_mm') / 1000.0
            if 'pressure_bar' in param_dict:
                param_dict['pressure_Pa'] = param_dict.pop('pressure_bar') * 1e5
            param_dicts.append(param_dict)
        
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
            
            print(f"  [OK] Simulation {sim_id} completed: {len(training_data)} data points")
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
                        random_sample=True, save_interval=10, random_sample_bounds=None,
                        n_jobs=1, sampling_method='random', lhs_seed=42, save_metadata=True, save_training_data=True,
                        save_complete_csv=False, task_id=None, ntasks=None,
                        conditions_log_path=None, append_condition_log=None,
                        record_completed_run=None, n_runs_this_task=None,
                        progress_status_path=None, **kwargs):
        """
        Generate complete training dataset.
        
        Parameters:
        -----------
        reactants : list, optional
            List of reactant keys to use. If None, use all reactants.
        max_combinations_per_reactant : int
            Maximum parameter combinations per reactant
        random_sample : bool
            Randomly sample parameter combinations (ignored if sampling_method='latin_hypercube')
        save_interval : int
            Save data every N simulations
        random_sample_bounds : dict, optional
            Bounds for random or LHS sampling
        n_jobs : int
            Number of parallel jobs. Use -1 for all available CPUs, 1 for sequential
        sampling_method : str
            'random' = random sample; 'full_grid' / 'structured_grid' / 'grid' = all combinations from parameter_ranges;
            'latin' or 'latin_hypercube' = Latin Hypercube Sampling (better space coverage)
        lhs_seed : int
            Random seed for Latin Hypercube Sampling (reproducibility)
        save_metadata : bool
            If True, save metadata JSON file; if False, skip (e.g. for notebook flags)
        save_training_data : bool
            If True, save partial and final training data (pickle); if False, keep only in memory and return
        save_complete_csv : bool, optional
            If True, also write ``training_data_complete_*.csv`` beside the pickle (large / slow). Default False.
        task_id : int, optional
            For SLURM/multi-process: this task's index (0 to ntasks-1). If set with ntasks, only runs
            simulations where global_index % ntasks == task_id for balanced parallelism.
        ntasks : int, optional
            Total number of parallel tasks. Use with task_id.
        conditions_log_path : str, optional
            Path to append per-run conditions (used with ``append_condition_log``).
        append_condition_log : callable, optional
            ``f(csv_path, conditions_dict)`` invoked before each simulation.
        record_completed_run : callable, optional
            ``f(completed, total_n)`` invoked after each simulation (SLURM chunk logging).
        n_runs_this_task : int, optional
            Total runs for this process; passed to ``record_completed_run``.
        progress_status_path : str, optional
            JSON file rewritten after each simulation with completion counts (for monitoring).
        
        Returns:
        --------
        pd.DataFrame
            Complete training dataset
        """
        if reactants is None:
            reactants = list(self.database['reactants'].keys())
        
        # Normalize sampling_method: accept aliases
        _method = (sampling_method or 'random').strip().lower()
        if _method == 'latin':
            _method = 'latin_hypercube'
        if _method in ('structured_grid', 'grid'):
            _method = 'full_grid'
        
        # Generate parameter combinations
        if _method == 'latin_hypercube':
            param_combinations = self.generate_parameter_combinations_lhs(
                n_samples=max_combinations_per_reactant,
                random_sample_bounds=random_sample_bounds,
                seed=lhs_seed
            )
        else:
            # random: random sample; full_grid / structured_grid: all combinations from parameter_ranges
            use_full_grid = (_method == 'full_grid')
            param_combinations = self.generate_parameter_combinations(
                max_combinations=max_combinations_per_reactant,
                random_sample=not use_full_grid,
                random_sample_bounds=random_sample_bounds
            )
        
        # Determine number of workers
        # On SLURM, respect the allocated CPU count (SLURM_CPUS_PER_TASK) rather than
        # the total node CPUs returned by cpu_count(), which would over-subscribe the node.
        import os as _os
        if n_jobs == -1:
            slurm_cpus = _os.environ.get("SLURM_CPUS_PER_TASK")
            n_jobs = int(slurm_cpus) if slurm_cpus else cpu_count()
        elif n_jobs < 1:
            n_jobs = 1
        
        total_simulations = len(reactants) * len(param_combinations)
        print(f"\nGenerating training data for {len(reactants)} reactants")
        print(f"  Sampling method: {_method}")
        print(f"  {len(param_combinations)} parameter combinations per reactant")
        print(f"  Total simulations: {total_simulations}")
        print(f"  Parallel jobs: {n_jobs} {'(sequential)' if n_jobs == 1 else f'(using {n_jobs} CPUs)'}")
        
        all_data = []
        saved_files = []  # Track all saved partial files for final combination
        successful_simulations = 0  # Track successful simulations across all saves
        start_time = time.time()
        
        # Prepare all simulation tasks
        all_tasks = []
        sim_id = 0
        for reactant_key in reactants:
            for params in param_combinations:
                sim_id += 1
                all_tasks.append((reactant_key, params, sim_id))
        
        # Chunk for SLURM/multi-process: only run this task's share
        if task_id is not None and ntasks is not None and ntasks > 1:
            all_tasks = [t for i, t in enumerate(all_tasks) if i % ntasks == task_id]
            total_simulations = len(all_tasks)
            print(f"  Task chunk: {task_id}/{ntasks} -> {total_simulations} simulations for this process")

        _write_generation_progress(
            progress_status_path, task_id, ntasks, 0, total_simulations, 0, 0, 0.0
        )

        if total_simulations == 0:
            print(
                "  [SKIP] No simulations assigned to this task chunk "
                "(normal when total runs < number of SLURM ranks)."
            )
            return None

        # Run simulations (parallel or sequential)
        if n_jobs > 1:
            # Parallel execution
            print(f"\n{'='*60}")
            print(f"Running {total_simulations} simulations in parallel ({n_jobs} workers)...")
            print(f"{'='*60}")
            
            # Create a wrapper function that can be pickled
            # We need to pass instance data as arguments since methods can't be pickled directly
            def run_simulation_wrapper(args):
                """Wrapper for parallel execution."""
                reactant_key, params, sim_id, output_dir, temp_dir = args
                try:
                    return _run_single_simulation_parallel_standalone(
                        reactant_key, params, sim_id, output_dir, temp_dir
                    )
                except Exception as e:
                    return None  # Fail silently in parallel mode
            
            # Prepare tasks with necessary data for parallel execution
            parallel_tasks = [
                (reactant_key, params, sim_id, str(self.output_dir), str(self.temp_dir))
                for reactant_key, params, sim_id in all_tasks
            ]
            
            # Use multiprocessing Pool
            with Pool(processes=n_jobs) as pool:
                completed = 0
                failed_simulations = 0
                
                # Process results as they complete
                for result in pool.imap(run_simulation_wrapper, parallel_tasks):
                    completed += 1
                    if result is not None:
                        all_data.append(result)
                        successful_simulations += 1
                    else:
                        failed_simulations += 1
                    
                    # Progress update - print after every simulation for early visibility
                    elapsed = time.time() - start_time
                    if completed > 0:
                        avg_time = elapsed / completed
                        remaining = (total_simulations - completed) * avg_time
                        success_rate = 100 * successful_simulations / completed if completed > 0 else 0
                        current_rows = sum(len(df) for df in all_data) if all_data else 0
                        
                        # Print progress with current analysis (on new line so it's always visible)
                        print(f"[Progress] {completed}/{total_simulations} "
                              f"({100*completed/total_simulations:.1f}%) | "
                              f"✓ Success: {successful_simulations} ({success_rate:.1f}%) | "
                              f"✗ Failed: {failed_simulations} | "
                              f"Data points: {current_rows:,} | "
                              f"ETA: {remaining/60:.1f} min")
                        _write_generation_progress(
                            progress_status_path, task_id, ntasks, completed, total_simulations,
                            successful_simulations, failed_simulations, elapsed,
                        )
                    
                    # Save periodically (only if saving training data to disk)
                    if save_training_data and completed % save_interval == 0:
                        if all_data:
                            combined_data = pd.concat(all_data, ignore_index=True)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = self.output_dir / f'training_data_partial_{timestamp}.pkl'
                            save_dataframe_pickle(combined_data, filename)
                            saved_files.append(filename)
                            print(f"\n  [SAVED] Partial data saved: {filename} ({len(combined_data):,} rows)")
                            # Clear all_data to free memory (data is now saved)
                            all_data = []
        else:
            # Sequential execution: iterate over all_tasks (so chunking by task_id/ntasks is respected)
            failed_simulations = 0
            completed = 0
            for reactant_key, params, sim_id in all_tasks:
                completed += 1
                if append_condition_log is not None and conditions_log_path:
                    append_condition_log(conditions_log_path, _params_to_conditions_log_row(params))
                if completed == 1 or (completed - 1) % 50 == 0:
                    print(f"\n{'='*60}")
                    print(f"Processing reactant: {reactant_key} (sim {completed}/{total_simulations})")
                    print(f"{'='*60}")
                training_data = self.run_single_simulation(reactant_key, params, sim_id)
                if training_data is not None:
                    all_data.append(training_data)
                    successful_simulations += 1
                else:
                    failed_simulations += 1
                # Progress update
                elapsed = time.time() - start_time
                if completed > 0:
                    avg_time = elapsed / completed
                    remaining = (total_simulations - completed) * avg_time
                    success_rate = 100 * successful_simulations / completed if completed > 0 else 0
                    current_rows = sum(len(df) for df in all_data) if all_data else 0
                    print(f"[Progress] {completed}/{total_simulations} "
                          f"({100*completed/total_simulations:.1f}%) | "
                          f"✓ Success: {successful_simulations} ({success_rate:.1f}%) | "
                          f"✗ Failed: {failed_simulations} | "
                          f"Data points: {current_rows:,} | "
                          f"ETA: {remaining/60:.1f} min")
                _cb_total = n_runs_this_task if n_runs_this_task is not None else total_simulations
                if record_completed_run is not None and _cb_total:
                    record_completed_run(completed, _cb_total)
                _write_generation_progress(
                    progress_status_path, task_id, ntasks, completed, total_simulations,
                    successful_simulations, failed_simulations, elapsed,
                )
                # Save periodically
                if save_training_data and completed % save_interval == 0:
                    if all_data:
                        combined_data = pd.concat(all_data, ignore_index=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = self.output_dir / f'training_data_partial_{timestamp}.pkl'
                        save_dataframe_pickle(combined_data, filename)
                        saved_files.append(filename)
                        print(f"\n  [SAVED] Partial data saved: {filename} ({len(combined_data):,} rows)")
                        all_data = []
        
        # Combine all data from saved files and any remaining in-memory data
        print(f"\n{'='*60}")
        print("Combining all training data..." + (" from partial files" if saved_files else " (in-memory only)"))
        
        # Load all partial files
        all_datasets = []
        if saved_files:
            print(f"  Loading {len(saved_files)} partial files...")
            for filepath in saved_files:
                df = load_dataframe_pickle(filepath)
                all_datasets.append(df)
                print(f"    Loaded {filepath.name}: {len(df):,} rows")
        
        # Add any remaining in-memory data
        if all_data:
            remaining_data = pd.concat(all_data, ignore_index=True)
            all_datasets.append(remaining_data)
            print(f"    Added remaining in-memory data: {len(remaining_data):,} rows")
        
        if all_datasets:
            complete_dataset = pd.concat(all_datasets, ignore_index=True)
            
            # Save final dataset to disk only if requested
            if save_training_data:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename_pkl = self.output_dir / f'training_data_complete_{timestamp}.pkl'
                save_dataframe_pickle(complete_dataset, filename_pkl)
                print(f"[OK] Complete dataset saved: {filename_pkl}")
                print(f"  Total rows: {len(complete_dataset):,}")
                print(f"  Total columns: {len(complete_dataset.columns)}")
                print(f"  File size: {os.path.getsize(filename_pkl) / 1e6:.2f} MB")
                if save_complete_csv:
                    filename_csv = self.output_dir / f'training_data_complete_{timestamp}.csv'
                    complete_dataset.to_csv(filename_csv, index=False)
                    print(f"[OK] CSV also saved: {filename_csv}")
                    print(f"  CSV file size: {os.path.getsize(filename_csv) / 1e6:.2f} MB")
            else:
                print(f"[OK] Complete dataset in memory only (not saved to disk)")
                print(f"  Total rows: {len(complete_dataset):,}")
                print(f"  Total columns: {len(complete_dataset.columns)}")
            
            # Calculate failed simulations (successful_simulations was tracked during execution)
            failed_simulations = total_simulations - successful_simulations
            
            if failed_simulations > 0:
                print(f"  Estimated successful simulations: ~{successful_simulations}/{total_simulations}")
                print(f"  Estimated failed simulations: ~{failed_simulations} ({100*failed_simulations/total_simulations:.1f}%)")
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
                'total_simulations': total_simulations,
                'successful_simulations': successful_simulations,
                'failed_simulations': failed_simulations,
                'success_rate': 100 * successful_simulations / total_simulations if total_simulations > 0 else 0,
                'n_jobs': n_jobs,
                'sampling_method': _method,
                'lhs_seed': lhs_seed if _method == 'latin_hypercube' else None,
                'partial_files': [str(f) for f in saved_files],
                'save_interval': save_interval
            }
            
            if save_metadata:
                metadata_file = self.output_dir / f'metadata_{timestamp}.json'
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"[OK] Metadata saved: {metadata_file}")
            
            # Clean up partial files after successful completion (only if we wrote them)
            if save_training_data and saved_files:
                print(f"\n{'='*60}")
                print("Cleaning up partial files...")
                deleted_count = 0
                deleted_size = 0
                for filepath in saved_files:
                    try:
                        if filepath.exists():
                            file_size = filepath.stat().st_size
                            filepath.unlink()
                            deleted_count += 1
                            deleted_size += file_size
                            print(f"  Deleted: {filepath.name} ({file_size / 1e6:.2f} MB)")
                    except Exception as e:
                        print(f"  Warning: Could not delete {filepath.name}: {e}")
                
                if deleted_count > 0:
                    print(f"\n[OK] Cleanup complete:")
                    print(f"  Deleted {deleted_count} partial file(s)")
                    print(f"  Freed {deleted_size / 1e6:.2f} MB of disk space")
                else:
                    print(f"  No partial files to clean up")
            
            return complete_dataset
        else:
            print("[ERROR] No data collected!")
            return None


def _run_single_simulation_parallel_standalone(reactant_key, params, sim_id, output_dir, temp_dir):
    """
    Standalone function for parallel execution (not a method, so it can be pickled).
    This function is called by each worker process.
    """
    try:
        # Load database in each process (needed for multiprocessing)
        database = load_reactant_database()
        
        # Create a temporary generator instance for this process
        temp_gen = TrainingDataGenerator(output_dir=output_dir)
        temp_gen.temp_dir = Path(temp_dir)
        
        # Create configuration
        config = temp_gen.create_config_from_params(reactant_key, params)
        reactant_info = database['reactants'][reactant_key]
        
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
        training_data = temp_gen._collect_training_data(
            gas, states, config, reactant_info, params, T_0, p_0, u_0, hf
        )
        
        # Clean up temporary heat flux file
        temp_file = config['mechanism']['heat_flux_file']
        if os.path.exists(temp_file):
            if 'temp' in os.path.dirname(temp_file) or 'heat_flux_' in os.path.basename(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass  # Ignore cleanup errors
        
        return training_data
            
    except Exception as e:
        # Fail silently in parallel mode to avoid cluttering output
        return None


def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_generation.py <config_file.json>")
        print("Example: python data_generation.py configs/ml/main2_data_generation_config.json")
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
    random_sample_bounds = config.get('random_sample_bounds', None)
    n_jobs = config.get('n_jobs', 1)  # Default to sequential
    sampling_method = config.get('sampling_method', 'random')  # 'random' | 'latin' | 'full_grid' | 'structured_grid' | 'grid'
    lhs_seed = config.get('lhs_seed', 42)
    
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
        save_interval=save_interval,
        random_sample_bounds=random_sample_bounds,
        n_jobs=n_jobs,
        sampling_method=sampling_method,
        lhs_seed=lhs_seed,
        save_complete_csv=config.get("save_complete_csv", False),
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
