#!/usr/bin/env python3
"""
ML Model Training Script
========================

Train multiple ML models to replace Cantera simulations.
Supports: Random Forest, XGBoost, Gradient Boosting, and (planned) PyTorch neural networks.

Author: Nikolas Karefyllidis, PhD
ML Surrogate Models Module
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Deep neural nets: planned stack is PyTorch (see train_neural_network); not required for tree models.

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. XGBoost training will be skipped.")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
class MLModelTrainer:
    """Train ML models to replace Cantera."""
    
    def __init__(self, data_file, output_dir='models'):
        """
        Initialize the ML trainer.
        
        Parameters:
        -----------
        data_file : str
            Path to training data CSV file
        output_dir : str
            Directory to save trained models
        """
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"Loading training data from: {data_file}")
        self.data = pd.read_csv(data_file)
        print(f"Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
        
        # Prepare features and targets
        self._prepare_features_and_targets()
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scalers = {}
        self.label_encoder = LabelEncoder()
        
    def _prepare_features_and_targets(self):
        """Prepare feature and target columns."""
        # Features: inputs to the model
        feature_cols = [
            'initial_temperature_K',
            'initial_pressure_Pa',
            'reactor_length_m',
            'reactor_diameter_m',
            'mass_flow_rate_kgps',
            'heat_flux_Wm2',
            'z_position_m',
            'relative_position'
        ]
        
        # Add reactant type if available
        if 'reactant_type' in self.data.columns:
            feature_cols.append('reactant_type')
        
        # Targets: outputs to predict
        # Primary targets (always predict these)
        primary_targets = [
            'temperature_K',
            'pressure_Pa',
            'velocity_ms',
            'density_kgm3'
        ]
        
        # Secondary targets (thermodynamic properties)
        secondary_targets = [
            'heat_capacity_cp_JkgK',
            'heat_capacity_cv_JkgK',
            'mean_molecular_weight_kgkmol',
            'enthalpy_Jkg',
            'entropy_JkgK',
            'viscosity_Pas',
            'thermal_conductivity_WmK'
        ]
        
        # Species targets: mass fractions only (Y_* and Y_lump_* both use Y_ prefix); mole X_* excluded
        species_targets = [col for col in self.data.columns if col.startswith('Y_')]
        
        self.feature_cols = [col for col in feature_cols if col in self.data.columns]
        self.primary_targets = [col for col in primary_targets if col in self.data.columns]
        self.secondary_targets = [col for col in secondary_targets if col in self.data.columns]
        self.species_targets = species_targets
        
        print(f"\nFeatures ({len(self.feature_cols)}): {self.feature_cols}")
        print(f"Primary targets ({len(self.primary_targets)}): {self.primary_targets}")
        print(f"Secondary targets ({len(self.secondary_targets)}): {self.secondary_targets}")
        print(f"Species targets ({len(self.species_targets)}): {len(self.species_targets)} species")
    
    def prepare_data(self, target_type='primary', test_size=0.2, random_state=42):
        """
        Prepare training and testing data.
        
        Parameters:
        -----------
        target_type : str
            'primary', 'secondary', 'species', or 'all'
        test_size : float
            Fraction of data for testing
        random_state : int
            Random seed
        
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, target_names)
        """
        # Select features
        X = self.data[self.feature_cols].copy()
        
        # Handle categorical features
        if 'reactant_type' in X.columns:
            X['reactant_type'] = self.label_encoder.fit_transform(X['reactant_type'])
        
        # Select targets
        if target_type == 'primary':
            target_cols = self.primary_targets
        elif target_type == 'secondary':
            target_cols = self.secondary_targets
        elif target_type == 'species':
            target_cols = self.species_targets
        elif target_type == 'all':
            target_cols = self.primary_targets + self.secondary_targets + self.species_targets
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        y = self.data[target_cols].copy()
        
        # Remove rows with NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"\nPreparing data for {target_type} targets...")
        print(f"  Valid samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Targets: {y.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Scale targets
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)
        self.target_scalers[target_type] = target_scaler
        
        return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
                target_cols, y_train, y_test)
    
    def train_neural_network(self, X_train, X_test, y_train, y_test, target_names,
                             target_type='primary', epochs=50, batch_size=256):
        """
        Train a neural network surrogate.

        The previous TensorFlow/Keras path was removed. HydrAI will standardize on PyTorch
        for deep models; wire torch.nn + multi-output regression here when ready.
        """
        _ = (X_train, X_test, y_train, y_test, target_names, target_type, epochs, batch_size)
        if not getattr(self, "_pytorch_nn_notice_shown", False):
            print(
                "Neural network: PyTorch trainer not implemented yet — skipping. "
                "Use tree models (RF / XGBoost / GB) or Main_4_train_tree_models.ipynb."
            )
            self._pytorch_nn_notice_shown = True
        return None

    def train_random_forest(self, X_train, X_test, y_train, y_test, target_names, 
                           target_type='primary', n_estimators=100, max_depth=20):
        """Train a Random Forest model."""
        print(f"\nTraining Random Forest for {target_type} targets...")
        
        # Random Forest doesn't need scaled data, but we'll use it anyway
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_original = self.target_scalers[target_type].inverse_transform(y_pred)
        y_test_original = self.target_scalers[target_type].inverse_transform(y_test)
        
        metrics = self._calculate_metrics(y_test_original, y_pred_original, target_names)
        
        # Save model
        model_name = f'random_forest_{target_type}'
        model_path = self.output_dir / f'{model_name}.pkl'
        joblib.dump(model, model_path)
        
        # Save scalers
        scaler_path = self.output_dir / f'{model_name}_scalers.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scalers[target_type],
                'label_encoder': self.label_encoder,
                'target_names': target_names
            }, f)
        
        print(f"[OK] Model saved: {model_path}")
        self._print_metrics(metrics, model_name)
        
        return {
            'model': model,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metrics': metrics,
            'type': 'random_forest'
        }
    
    def train_xgboost(self, X_train, X_test, y_train, y_test, target_names, 
                     target_type='primary', n_estimators=100, max_depth=6):
        """Train an XGBoost model."""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available. Skipping XGBoost training.")
            return None
        
        print(f"\nTraining XGBoost for {target_type} targets...")
        
        # XGBoost can handle multiple outputs via MultiOutputRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_original = self.target_scalers[target_type].inverse_transform(y_pred)
        y_test_original = self.target_scalers[target_type].inverse_transform(y_test)
        
        metrics = self._calculate_metrics(y_test_original, y_pred_original, target_names)
        
        # Save model
        model_name = f'xgboost_{target_type}'
        model_path = self.output_dir / f'{model_name}.pkl'
        joblib.dump(model, model_path)
        
        # Save scalers
        scaler_path = self.output_dir / f'{model_name}_scalers.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scalers[target_type],
                'label_encoder': self.label_encoder,
                'target_names': target_names
            }, f)
        
        print(f"[OK] Model saved: {model_path}")
        self._print_metrics(metrics, model_name)
        
        return {
            'model': model,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metrics': metrics,
            'type': 'xgboost'
        }
    
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test, target_names, 
                               target_type='primary', n_estimators=100, max_depth=5):
        """Train a Gradient Boosting model."""
        print(f"\nTraining Gradient Boosting for {target_type} targets...")
        
        from sklearn.multioutput import MultiOutputRegressor
        
        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                random_state=42
            )
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_original = self.target_scalers[target_type].inverse_transform(y_pred)
        y_test_original = self.target_scalers[target_type].inverse_transform(y_test)
        
        metrics = self._calculate_metrics(y_test_original, y_pred_original, target_names)
        
        # Save model
        model_name = f'gradient_boosting_{target_type}'
        model_path = self.output_dir / f'{model_name}.pkl'
        joblib.dump(model, model_path)
        
        # Save scalers
        scaler_path = self.output_dir / f'{model_name}_scalers.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scalers[target_type],
                'label_encoder': self.label_encoder,
                'target_names': target_names
            }, f)
        
        print(f"[OK] Model saved: {model_path}")
        self._print_metrics(metrics, model_name)
        
        return {
            'model': model,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metrics': metrics,
            'type': 'gradient_boosting'
        }
    
    def _calculate_metrics(self, y_true, y_pred, target_names):
        """Calculate evaluation metrics."""
        metrics = {}
        for i, target_name in enumerate(target_names):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            metrics[target_name] = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            }
        
        # Overall metrics
        overall_mse = mean_squared_error(y_true, y_pred)
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(y_true, y_pred)
        
        metrics['overall'] = {
            'mse': float(overall_mse),
            'mae': float(overall_mae),
            'rmse': float(overall_rmse),
            'r2': float(overall_r2)
        }
        
        return metrics
    
    def _print_metrics(self, metrics, model_name):
        """Print evaluation metrics."""
        print(f"\n{model_name} - Evaluation Metrics:")
        print(f"  Overall R²: {metrics['overall']['r2']:.4f}")
        print(f"  Overall RMSE: {metrics['overall']['rmse']:.4f}")
        print(f"  Overall MAE: {metrics['overall']['mae']:.4f}")
    
    def train_all_models(self, target_types=['primary'], models=['all']):
        """
        Train all specified models.
        
        Parameters:
        -----------
        target_types : list
            List of target types to train ('primary', 'secondary', 'species')
        models : list
            List of models to train ('neural_network' placeholder until PyTorch; 'random_forest', 'xgboost',
            'gradient_boosting', or 'all'). ``all`` expands to the three tree/boosting models only.
        """
        if 'all' in models:
            # Exclude neural_network until PyTorch trainer is implemented (see train_neural_network).
            models = ['random_forest', 'xgboost', 'gradient_boosting']
        
        all_results = {}
        
        for target_type in target_types:
            print(f"\n{'='*60}")
            print(f"Training models for {target_type} targets")
            print(f"{'='*60}")
            
            # Prepare data
            X_train, X_test, y_train, y_test, target_names, y_train_orig, y_test_orig = \
                self.prepare_data(target_type=target_type)
            
            results = {}
            
            # Train each model
            if 'neural_network' in models:
                results['neural_network'] = self.train_neural_network(
                    X_train, X_test, y_train, y_test, target_names, target_type
                )
            
            if 'random_forest' in models:
                results['random_forest'] = self.train_random_forest(
                    X_train, X_test, y_train, y_test, target_names, target_type
                )
            
            if 'xgboost' in models and XGBOOST_AVAILABLE:
                results['xgboost'] = self.train_xgboost(
                    X_train, X_test, y_train, y_test, target_names, target_type
                )
            
            if 'gradient_boosting' in models:
                results['gradient_boosting'] = self.train_gradient_boosting(
                    X_train, X_test, y_train, y_test, target_names, target_type
                )
            
            all_results[target_type] = results
        
        # Save training summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'data_file': str(self.data_file),
            'results': {}
        }
        
        for target_type, results in all_results.items():
            summary['results'][target_type] = {}
            for model_name, result in results.items():
                if result:
                    summary['results'][target_type][model_name] = {
                        'model_path': str(result['model_path']),
                        'scaler_path': str(result['scaler_path']),
                        'metrics': result['metrics']
                    }
        
        summary_file = self.output_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[OK] Training summary saved: {summary_file}")
        
        return all_results


def main():
    """Main execution function."""
    import sys
    import glob
    
    if len(sys.argv) < 2:
        print("Usage: python model_training.py <config_file.json>")
        print("Example: python model_training.py configs/ml/model_training_script_config.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load configuration from JSON
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    data_file_pattern = config.get('data_file', 'data/training/training_data_complete_*.csv')
    output_dir = config.get('output_dir', 'models')
    target_types = config.get('target_types', ['primary'])
    models = config.get('models', ['all'])
    
    # Expand glob pattern for data file
    data_files = glob.glob(data_file_pattern)
    if not data_files:
        print(f"Error: No data files found matching pattern: {data_file_pattern}")
        sys.exit(1)
    
    if len(data_files) > 1:
        print(f"Warning: Multiple data files found. Using first: {data_files[0]}")
    
    data_file = data_files[0]
    
    # Create trainer
    trainer = MLModelTrainer(data_file, output_dir)
    
    # Update model-specific parameters if provided
    if 'neural_network' in config and hasattr(trainer, 'train_neural_network'):
        # Store config for later use
        trainer.nn_config = config['neural_network']
    
    # Train models
    results = trainer.train_all_models(
        target_types=target_types,
        models=models
    )
    
    print(f"\n{'='*60}")
    print("ML MODEL TRAINING COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
