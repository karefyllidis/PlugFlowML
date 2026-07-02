#!/usr/bin/env python3
"""
ML Inference — Fast surrogate predictions using trained tree models.

Loads the .joblib artifacts exported by Main_4 / Main_5 and provides
single-point and full-profile prediction.  100-1000x faster than Cantera.

Author: Nikolas Karefyllidis, PhD
"""

import glob
import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union


class MLPFRPredictor:
    """
    Fast surrogate predictor for steam-cracking PFR simulations.

    Wraps the .joblib artifact produced by Main_4 / Main_5 (for example
    ``tree_models_exit.joblib`` or legacy ``tree_models_exit_<timestamp>.joblib``).
    Each artifact contains:
        'models'      : dict {model_key: fitted MultiOutputRegressor}
        'scaler_X'    : fitted StandardScaler for input features
        'feature_cols': list of feature column names (len=8)
        'target_cols' : list of target column names (len≈320)
        'X_train/test', 'y_train/test', 'X_train_s/test_s': train/test data

    Parameters
    ----------
    artifact_path : str or Path
        Path to a .joblib artifact, OR a directory in which the preferred
        ``tree_models_<mode>.joblib`` (or legacy ``tree_models_<mode>_<timestamp>.joblib``)
        is auto-discovered (newest matching file wins).
    model_key : str, optional
        Which model to use ('random_forest', 'gradient_boosting', 'xgboost',
        'adaboost').  Defaults to the first model in the artifact.
    mode : str, optional
        When ``artifact_path`` is a directory, filter artifacts by mode name
        (e.g. 'exit', 'full_profile').

    Examples
    --------
    >>> predictor = MLPFRPredictor('models/tree_baseline/', model_key='xgboost', mode='exit')
    >>> result = predictor.predict_exit(
    ...     initial_temperature_K=1050,
    ...     initial_pressure_Pa=2e5,
    ...     reactor_length_m=12.0,
    ...     reactor_diameter_m=0.03,
    ...     mass_flow_rate_kgps=0.07,
    ...     heat_flux_Wm2=150_000,
    ... )
    >>> profile = predictor.predict_profile(
    ...     initial_temperature_K=1050,
    ...     initial_pressure_Pa=2e5,
    ...     reactor_length_m=12.0,
    ...     reactor_diameter_m=0.03,
    ...     mass_flow_rate_kgps=0.07,
    ...     heat_flux_Wm2=150_000,
    ...     n_points=200,
    ... )
    """

    def __init__(
        self,
        artifact_path: Union[str, Path],
        model_key: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        artifact_path = Path(artifact_path)

        if artifact_path.is_dir():
            pattern = str(artifact_path / 'tree_models_*.joblib')
            candidates = sorted(glob.glob(pattern), reverse=True)
            if not candidates:
                raise FileNotFoundError(
                    f"No tree_models_*.joblib artifacts found in {artifact_path}. "
                    "Run Main_4 with IF_MODEL_EXPORT=True first."
                )
            if mode is not None:
                esc = re.escape(mode)
                pat = re.compile(
                    rf"^tree_models_{esc}(?:_\d{{8}}(?:_\d{{6}})?)?\.joblib$",
                    re.IGNORECASE,
                )
                candidates = [c for c in candidates if pat.match(Path(c).name)]
                if not candidates:
                    raise FileNotFoundError(
                        f"No artifact found for mode='{mode}' in {artifact_path}."
                    )
            artifact_path = Path(candidates[0])

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        self._artifact = joblib.load(artifact_path)
        self._artifact_path = artifact_path

        available = list(self._artifact['models'].keys())
        if model_key is None:
            model_key = available[0]
        if model_key not in available:
            raise ValueError(
                f"model_key '{model_key}' not in artifact. Available: {available}"
            )

        self.model_key    = model_key
        self.model        = self._artifact['models'][model_key]
        self.scaler_X     = self._artifact['scaler_X']
        self.feature_cols = self._artifact['feature_cols']
        self.target_cols  = self._artifact['target_cols']

        print(f"[MLPFRPredictor] {artifact_path.name}")
        print(f"  model   : {model_key}")
        print(f"  features: {len(self.feature_cols)}")
        print(f"  targets : {len(self.target_cols)}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_feature_row(
        self,
        initial_temperature_K: float,
        initial_pressure_Pa: float,
        reactor_length_m: float,
        reactor_diameter_m: float,
        mass_flow_rate_kgps: float,
        heat_flux_Wm2: float,
        z_position_m: float,
    ) -> np.ndarray:
        """Return a (1, n_features) array in the same column order as training."""
        relative_position = z_position_m / reactor_length_m if reactor_length_m > 0 else 0.0
        values = {
            'initial_temperature_K': initial_temperature_K,
            'initial_pressure_Pa':   initial_pressure_Pa,
            'reactor_length_m':      reactor_length_m,
            'reactor_diameter_m':    reactor_diameter_m,
            'mass_flow_rate_kgps':   mass_flow_rate_kgps,
            'heat_flux_Wm2':         heat_flux_Wm2,
            'z_position_m':          z_position_m,
            'relative_position':     relative_position,
        }
        row = np.array([[values.get(c, 0.0) for c in self.feature_cols]], dtype=float)
        return row

    def _predict_rows(self, X_raw: np.ndarray) -> pd.DataFrame:
        """Scale features, run model, return DataFrame with target column names."""
        X_scaled = self.scaler_X.transform(X_raw)
        y_pred   = self.model.predict(X_scaled)
        return pd.DataFrame(y_pred, columns=self.target_cols)

    # ── public API ────────────────────────────────────────────────────────────

    def predict_exit(
        self,
        initial_temperature_K: float,
        initial_pressure_Pa: float,
        reactor_length_m: float,
        reactor_diameter_m: float,
        mass_flow_rate_kgps: float,
        heat_flux_Wm2: float,
    ) -> Dict[str, float]:
        """
        Predict reactor exit conditions (z = reactor_length_m).

        Returns
        -------
        dict
            {target_name: predicted_value} for all ~320 output targets.
        """
        X = self._build_feature_row(
            initial_temperature_K, initial_pressure_Pa,
            reactor_length_m, reactor_diameter_m,
            mass_flow_rate_kgps, heat_flux_Wm2,
            z_position_m=reactor_length_m,
        )
        df = self._predict_rows(X)
        return df.iloc[0].to_dict()

    def predict_profile(
        self,
        initial_temperature_K: float,
        initial_pressure_Pa: float,
        reactor_length_m: float,
        reactor_diameter_m: float,
        mass_flow_rate_kgps: float,
        heat_flux_Wm2: float,
        n_points: int = 200,
    ) -> pd.DataFrame:
        """
        Predict the complete axial reactor profile at ``n_points`` positions.

        Returns
        -------
        pd.DataFrame
            Rows = axial positions; columns = all output targets plus
            ``z_position_m`` and ``relative_position``.
        """
        z_positions = np.linspace(0.0, reactor_length_m, n_points)
        X_rows = np.vstack([
            self._build_feature_row(
                initial_temperature_K, initial_pressure_Pa,
                reactor_length_m, reactor_diameter_m,
                mass_flow_rate_kgps, heat_flux_Wm2, z,
            )
            for z in z_positions
        ])
        df = self._predict_rows(X_rows)
        df.insert(0, 'z_position_m',     z_positions)
        df.insert(1, 'relative_position', z_positions / reactor_length_m)
        return df

    # ── convenience ───────────────────────────────────────────────────────────

    def available_models(self) -> List[str]:
        """Return all model keys stored in the loaded artifact."""
        return list(self._artifact['models'].keys())

    def switch_model(self, model_key: str) -> None:
        """Hot-swap to a different model stored in the same artifact."""
        available = self.available_models()
        if model_key not in available:
            raise ValueError(f"'{model_key}' not in artifact. Available: {available}")
        self.model_key = model_key
        self.model     = self._artifact['models'][model_key]
        print(f"[MLPFRPredictor] switched to: {model_key}")
