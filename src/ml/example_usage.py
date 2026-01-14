#!/usr/bin/env python3
"""
ML Surrogate Models: Example Usage Script
=========================================

Example demonstrating how to use ML surrogate models for PFR predictions.

Author: Nikolas Karefyllidis, PhD
ML Surrogate Models Module
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.inference import MLPFRPredictor
import pandas as pd
import matplotlib.pyplot as plt


def example_single_prediction():
    """Example: Predict reactor state at a single point."""
    print("\n" + "="*60)
    print("Example 1: Single Point Prediction")
    print("="*60)
    
    # Load predictor
    predictor = MLPFRPredictor(
        model_dir='models',
        model_type='neural_network',  # or 'random_forest', 'xgboost', etc.
        target_type='primary'
    )
    
    # Predict at a specific position
    result = predictor.predict_single_point(
        initial_temperature_K=925.0,
        initial_pressure_Pa=200000.0,  # 2.0 bar
        reactor_length_m=5.0,
        reactor_diameter_m=0.03,  # 30 mm
        mass_flow_rate_kgps=0.07,
        heat_flux_Wm2=150000.0,
        z_position_m=2.5  # Middle of reactor
    )
    
    print(f"\nPredicted values at z = 2.5 m:")
    print(f"  Temperature: {result['temperature_K']:.1f} K")
    print(f"  Pressure: {result['pressure_Pa']/1e5:.2f} bar")
    print(f"  Velocity: {result['velocity_ms']:.2f} m/s")
    print(f"  Density: {result['density_kgm3']:.2f} kg/m³")


def example_profile_prediction():
    """Example: Predict complete reactor profile."""
    print("\n" + "="*60)
    print("Example 2: Complete Reactor Profile")
    print("="*60)
    
    # Load predictor
    predictor = MLPFRPredictor(
        model_dir='models',
        model_type='neural_network',
        target_type='primary'
    )
    
    # Predict complete profile
    profile = predictor.predict_profile(
        initial_temperature_K=925.0,
        initial_pressure_Pa=200000.0,
        reactor_length_m=5.0,
        reactor_diameter_m=0.03,
        mass_flow_rate_kgps=0.07,
        heat_flux_Wm2=150000.0,
        n_points=200
    )
    
    print(f"\nGenerated profile with {len(profile)} points")
    print(f"Columns: {list(profile.columns)}")
    
    # Save to CSV
    output_file = 'outputs/example_profile.csv'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_file, index=False)
    print(f"\n[OK] Profile saved to: {output_file}")
    
    # Display summary
    print("\nProfile Summary:")
    print(f"  Initial T: {profile['temperature_K'].iloc[0]:.1f} K")
    print(f"  Final T: {profile['temperature_K'].iloc[-1]:.1f} K")
    print(f"  Temperature rise: {profile['temperature_K'].iloc[-1] - profile['temperature_K'].iloc[0]:.1f} K")
    print(f"  Initial P: {profile['pressure_Pa'].iloc[0]/1e5:.2f} bar")
    print(f"  Final P: {profile['pressure_Pa'].iloc[-1]/1e5:.2f} bar")
    print(f"  Pressure drop: {(profile['pressure_Pa'].iloc[0] - profile['pressure_Pa'].iloc[-1])/1e5:.2f} bar")


def example_visualization():
    """Example: Visualize predictions."""
    print("\n" + "="*60)
    print("Example 3: Visualization")
    print("="*60)
    
    # Load predictor
    predictor = MLPFRPredictor(
        model_dir='models',
        model_type='neural_network',
        target_type='primary'
    )
    
    # Predict profile
    profile = predictor.predict_profile(
        initial_temperature_K=925.0,
        initial_pressure_Pa=200000.0,
        reactor_length_m=5.0,
        reactor_diameter_m=0.03,
        mass_flow_rate_kgps=0.07,
        heat_flux_Wm2=150000.0,
        n_points=200
    )
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Temperature profile
    axes[0, 0].plot(profile['z_position_m'], profile['temperature_K'], 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Position [m]')
    axes[0, 0].set_ylabel('Temperature [K]')
    axes[0, 0].set_title('Temperature Profile (ML Prediction)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pressure profile
    axes[0, 1].plot(profile['z_position_m'], profile['pressure_Pa']/1e5, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Position [m]')
    axes[0, 1].set_ylabel('Pressure [bar]')
    axes[0, 1].set_title('Pressure Profile (ML Prediction)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Velocity profile
    axes[1, 0].plot(profile['z_position_m'], profile['velocity_ms'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Position [m]')
    axes[1, 0].set_ylabel('Velocity [m/s]')
    axes[1, 0].set_title('Velocity Profile (ML Prediction)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Density profile
    axes[1, 1].plot(profile['z_position_m'], profile['density_kgm3'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Position [m]')
    axes[1, 1].set_ylabel('Density [kg/m³]')
    axes[1, 1].set_title('Density Profile (ML Prediction)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'outputs/example_ml_predictions.png'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {output_file}")
    plt.close()


def example_parameter_study():
    """Example: Parameter study using ML models."""
    print("\n" + "="*60)
    print("Example 4: Parameter Study")
    print("="*60)
    
    # Load predictor
    predictor = MLPFRPredictor(
        model_dir='models',
        model_type='neural_network',
        target_type='primary'
    )
    
    # Vary temperature
    temperatures = [850, 900, 950, 1000, 1050]
    final_temperatures = []
    
    for T in temperatures:
        result = predictor.predict_single_point(
            initial_temperature_K=T,
            initial_pressure_Pa=200000.0,
            reactor_length_m=5.0,
            reactor_diameter_m=0.03,
            mass_flow_rate_kgps=0.07,
            heat_flux_Wm2=150000.0,
            z_position_m=5.0  # Outlet
        )
        final_temperatures.append(result['temperature_K'])
    
    # Display results
    print("\nParameter Study: Effect of Initial Temperature")
    print("Initial T [K] | Final T [K] | Temperature Rise [K]")
    print("-" * 50)
    for T_init, T_final in zip(temperatures, final_temperatures):
        print(f"  {T_init:8.0f}   | {T_final:8.1f}   | {T_final - T_init:15.1f}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ML Surrogate Models - Example Usage")
    print("="*60)
    print("\nNote: Make sure you have trained models in models/")
    print("      Run model_training.py first if needed.\n")
    
    try:
        # Check if models exist
        model_dir = Path('models')
        if not model_dir.exists() or not list(model_dir.glob('*.pkl')) and not list(model_dir.glob('*.h5')):
            print("[WARNING] No trained models found!")
            print("  Please run: python src/ml/model_training.py --data data/training/*.csv")
            return
        
        # Run examples
        example_single_prediction()
        example_profile_prediction()
        example_visualization()
        example_parameter_study()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("  Make sure models are trained first!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
