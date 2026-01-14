#!/usr/bin/env python3
"""
ML Inference Script
===================

Use trained ML models to predict PFR behavior instead of using Cantera.
This replaces the Cantera simulation with fast ML predictions.

Author: Nikolas Karefyllidis, PhD
ML Surrogate Models Module
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib
from typing import Dict, List, Optional, Tuple

# ML Libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, LabelEncoder


class MLPFRPredictor:
    """ML-based PFR predictor to replace Cantera."""
    
    def __init__(self, model_dir='models', model_type='neural_network', 
                 target_type='primary'):
        """
        Initialize ML predictor.
        
        Parameters:
        -----------
        model_dir : str
            Directory containing trained models
        model_type : str
            Type of model to use ('neural_network', 'random_forest', 'xgboost', 'gradient_boosting')
        target_type : str
            Type of targets ('primary', 'secondary', 'species')
        """
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.target_type = target_type
        
        # Load model and scalers
        self._load_model()
        
    def _load_model(self):
        """Load trained model and scalers."""
        model_name = f'{self.model_type}_{self.target_type}'
        model_path = self.model_dir / f'{model_name}.pkl'
        
        if self.model_type == 'neural_network':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not available for neural network inference")
            model_path = self.model_dir / f'{model_name}.h5'
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model = keras.models.load_model(model_path)
        else:
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model = joblib.load(model_path)
        
        # Load scalers
        scaler_path = self.model_dir / f'{model_name}_scalers.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scalers not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            self.feature_scaler = scaler_data['feature_scaler']
            self.target_scaler = scaler_data['target_scaler']
            self.label_encoder = scaler_data['label_encoder']
            self.target_names = scaler_data['target_names']
        
        print(f"[OK] Loaded {self.model_type} model for {self.target_type} targets")
        print(f"  Model: {model_path}")
        print(f"  Targets: {len(self.target_names)}")
    
    def predict_single_point(self, initial_temperature_K: float,
                            initial_pressure_Pa: float,
                            reactor_length_m: float,
                            reactor_diameter_m: float,
                            mass_flow_rate_kgps: float,
                            heat_flux_Wm2: float,
                            z_position_m: float,
                            reactant_type: Optional[str] = None) -> Dict[str, float]:
        """
        Predict reactor state at a single position.
        
        Parameters:
        -----------
        initial_temperature_K : float
            Initial temperature in Kelvin
        initial_pressure_Pa : float
            Initial pressure in Pascal
        reactor_length_m : float
            Reactor length in meters
        reactor_diameter_m : float
            Reactor diameter in meters
        mass_flow_rate_kgps : float
            Mass flow rate in kg/s
        heat_flux_Wm2 : float
            Heat flux in W/m²
        z_position_m : float
            Axial position in meters
        reactant_type : str, optional
            Reactant type (if model was trained with it)
        
        Returns:
        --------
        dict
            Dictionary of predicted values
        """
        # Prepare features
        features = np.array([[
            initial_temperature_K,
            initial_pressure_Pa,
            reactor_length_m,
            reactor_diameter_m,
            mass_flow_rate_kgps,
            heat_flux_Wm2,
            z_position_m,
            z_position_m / reactor_length_m  # relative_position
        ]])
        
        # Add reactant type if available
        if reactant_type is not None and hasattr(self.label_encoder, 'classes_'):
            try:
                reactant_encoded = self.label_encoder.transform([reactant_type])[0]
                features = np.hstack([features, [[reactant_encoded]]])
            except:
                pass  # Reactant type not in training data
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Predict
        predictions_scaled = self.model.predict(features_scaled, verbose=0)
        
        # Inverse transform
        predictions = self.target_scaler.inverse_transform(predictions_scaled)[0]
        
        # Create result dictionary
        results = dict(zip(self.target_names, predictions))
        
        return results
    
    def predict_profile(self, initial_temperature_K: float,
                       initial_pressure_Pa: float,
                       reactor_length_m: float,
                       reactor_diameter_m: float,
                       mass_flow_rate_kgps: float,
                       heat_flux_Wm2: float,
                       n_points: int = 200,
                       reactant_type: Optional[str] = None) -> pd.DataFrame:
        """
        Predict complete reactor profile.
        
        Parameters:
        -----------
        initial_temperature_K : float
            Initial temperature in Kelvin
        initial_pressure_Pa : float
            Initial pressure in Pascal
        reactor_length_m : float
            Reactor length in meters
        reactor_diameter_m : float
            Reactor diameter in meters
        mass_flow_rate_kgps : float
            Mass flow rate in kg/s
        heat_flux_Wm2 : float
            Heat flux in W/m²
        n_points : int
            Number of points along reactor
        reactant_type : str, optional
            Reactant type
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions at each position
        """
        # Generate positions
        z_positions = np.linspace(0, reactor_length_m, n_points)
        
        # Predict at each position
        predictions = []
        for z in z_positions:
            pred = self.predict_single_point(
                initial_temperature_K, initial_pressure_Pa,
                reactor_length_m, reactor_diameter_m,
                mass_flow_rate_kgps, heat_flux_Wm2, z, reactant_type
            )
            pred['z_position_m'] = z
            pred['relative_position'] = z / reactor_length_m
            predictions.append(pred)
        
        return pd.DataFrame(predictions)
    
    def predict_with_adaptive_step(self, initial_temperature_K: float,
                                  initial_pressure_Pa: float,
                                  reactor_length_m: float,
                                  reactor_diameter_m: float,
                                  mass_flow_rate_kgps: float,
                                  heat_flux_Wm2: float,
                                  max_step_size: float = 0.1,
                                  reactant_type: Optional[str] = None) -> pd.DataFrame:
        """
        Predict reactor profile with adaptive step size.
        
        Uses smaller steps where gradients are large (more resolution where needed).
        
        Parameters:
        -----------
        initial_temperature_K : float
            Initial temperature in Kelvin
        initial_pressure_Pa : float
            Initial pressure in Pascal
        reactor_length_m : float
            Reactor length in meters
        reactor_diameter_m : float
            Reactor diameter in meters
        mass_flow_rate_kgps : float
            Mass flow rate in kg/s
        heat_flux_Wm2 : float
            Heat flux in W/m²
        max_step_size : float
            Maximum step size in meters
        reactant_type : str, optional
            Reactant type
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions
        """
        z_positions = [0.0]
        predictions = []
        
        # Initial prediction
        pred = self.predict_single_point(
            initial_temperature_K, initial_pressure_Pa,
            reactor_length_m, reactor_diameter_m,
            mass_flow_rate_kgps, heat_flux_Wm2, 0.0, reactant_type
        )
        pred['z_position_m'] = 0.0
        predictions.append(pred)
        
        z = 0.0
        while z < reactor_length_m:
            # Calculate step size based on gradient
            if len(predictions) > 1:
                # Use temperature gradient to determine step size
                temp_grad = abs(predictions[-1]['temperature_K'] - predictions[-2]['temperature_K'])
                step_size = min(max_step_size, max(0.01, max_step_size / (1 + temp_grad / 10)))
            else:
                step_size = max_step_size
            
            z = min(z + step_size, reactor_length_m)
            z_positions.append(z)
            
            pred = self.predict_single_point(
                initial_temperature_K, initial_pressure_Pa,
                reactor_length_m, reactor_diameter_m,
                mass_flow_rate_kgps, heat_flux_Wm2, z, reactant_type
            )
            pred['z_position_m'] = z
            pred['relative_position'] = z / reactor_length_m
            predictions.append(pred)
        
        return pd.DataFrame(predictions)


def main():
    """Main execution function for ML inference."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <config_file.json>")
        print("Example: python inference.py configs/ml_inference_config.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load configuration from JSON
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    model_dir = config.get('model_dir', 'models')
    model_type = config.get('model_type', 'neural_network')
    target_type = config.get('target_type', 'primary')
    
    sim_params = config.get('simulation_parameters', {})
    pred_settings = config.get('prediction_settings', {})
    output_config = config.get('output', {})
    
    # Create predictor
    predictor = MLPFRPredictor(
        model_dir=model_dir,
        model_type=model_type,
        target_type=target_type
    )
    
    # Extract simulation parameters
    temp = sim_params.get('initial_temperature_K', 925.0)
    pressure_bar = sim_params.get('initial_pressure_bar', 2.0)
    length = sim_params.get('reactor_length_m', 5.0)
    diameter_mm = sim_params.get('reactor_diameter_mm', 30.0)
    mass_flow = sim_params.get('mass_flow_rate_kgps', 0.07)
    heat_flux = sim_params.get('heat_flux_Wm2', 150000.0)
    reactant_type = sim_params.get('reactant_type', None)
    
    # Predict profile
    print(f"\nPredicting reactor profile...")
    print(f"  T={temp}K, P={pressure_bar}bar, L={length}m")
    
    # Use adaptive step if requested
    if pred_settings.get('adaptive_step', False):
        profile = predictor.predict_with_adaptive_step(
            initial_temperature_K=temp,
            initial_pressure_Pa=pressure_bar * 1e5,
            reactor_length_m=length,
            reactor_diameter_m=diameter_mm / 1000.0,
            mass_flow_rate_kgps=mass_flow,
            heat_flux_Wm2=heat_flux,
            max_step_size=pred_settings.get('max_step_size', 0.1),
            reactant_type=reactant_type
        )
    else:
        n_points = pred_settings.get('n_points', 200)
        profile = predictor.predict_profile(
            initial_temperature_K=temp,
            initial_pressure_Pa=pressure_bar * 1e5,
            reactor_length_m=length,
            reactor_diameter_m=diameter_mm / 1000.0,
            mass_flow_rate_kgps=mass_flow,
            heat_flux_Wm2=heat_flux,
            n_points=n_points,
            reactant_type=reactant_type
        )
    
    # Save results
    output_file = output_config.get('file', 'outputs/predictions.csv')
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_config.get('format', 'csv') == 'csv':
        profile.to_csv(output_path, index=False)
    else:
        profile.to_json(output_path, orient='records', indent=2)
    
    print(f"\n[OK] Predictions saved to: {output_path}")
    print(f"  Rows: {len(profile)}")
    print(f"  Columns: {len(profile.columns)}")
    print(f"\nSample predictions:")
    print(profile[['z_position_m', 'temperature_K', 'pressure_Pa', 'velocity_ms', 'density_kgm3']].head())


if __name__ == "__main__":
    main()
