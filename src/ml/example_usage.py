#!/usr/bin/env python3
"""
ML Surrogate Models — Example Usage
====================================

Minimal examples demonstrating the MLPFRPredictor API for fast PFR predictions
using trained tree-based surrogate models.

Prerequisites
-------------
Run notebooks/Main_4_train_tree_models.ipynb first with IF_TREE_MODEL_EXPORT=True
so that ``models/tree_models_<mode>_<timestamp>.joblib`` exists.

Usage
-----
    python src/ml/example_usage.py

Author: Nikolas Karefyllidis, PhD
"""

import sys
from pathlib import Path

# Project root on sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.ml.inference import MLPFRPredictor


# ── Common operating conditions ──────────────────────────────────────────────
CASE = dict(
    initial_temperature_K = 925.0,
    initial_pressure_Pa   = 200_000.0,   # 2.0 bar
    reactor_length_m      = 5.0,
    reactor_diameter_m    = 0.03,        # 30 mm
    mass_flow_rate_kgps   = 0.07,
    heat_flux_Wm2         = 150_000.0,
)


def example_exit_prediction():
    """Predict reactor exit conditions with a trained surrogate model."""
    print("\n" + "=" * 60)
    print("Example 1: Reactor exit prediction (single point)")
    print("=" * 60)

    predictor = MLPFRPredictor(
        artifact_path = project_root / "models",
        model_key     = "xgboost",   # or 'random_forest', 'gradient_boosting', 'adaboost'
        mode          = "exit",
    )

    result = predictor.predict_exit(**CASE)

    print(f"\n[exit conditions]")
    print(f"  Temperature : {result.get('temperature_K', float('nan')):.1f} K")
    print(f"  Pressure    : {result.get('pressure_Pa', float('nan')) / 1e5:.2f} bar")
    print(f"  Velocity    : {result.get('velocity_ms', float('nan')):.2f} m/s")
    print(f"  Density     : {result.get('density_kgm3', float('nan')):.3f} kg/m³")


def example_full_profile():
    """Predict the complete axial profile and save to CSV."""
    print("\n" + "=" * 60)
    print("Example 2: Full axial profile (200 points)")
    print("=" * 60)

    predictor = MLPFRPredictor(
        artifact_path = project_root / "models",
        model_key     = "xgboost",
        mode          = "exit",
    )

    profile = predictor.predict_profile(**CASE, n_points=200)

    print(f"\n  Profile shape: {profile.shape}")
    print(f"  T inlet  → T outlet : "
          f"{profile['temperature_K'].iloc[0]:.1f} K → "
          f"{profile['temperature_K'].iloc[-1]:.1f} K")
    print(f"  P inlet  → P outlet : "
          f"{profile['pressure_Pa'].iloc[0] / 1e5:.2f} bar → "
          f"{profile['pressure_Pa'].iloc[-1] / 1e5:.2f} bar")

    out_csv = project_root / "outputs" / "example_profile.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")


def example_compare_models():
    """Compare exit predictions from each trained model in the artifact."""
    print("\n" + "=" * 60)
    print("Example 3: Compare all models in the artifact")
    print("=" * 60)

    predictor = MLPFRPredictor(artifact_path=project_root / "models", mode="exit")

    print(f"\n  Available models: {predictor.available_models()}\n")
    print(f"  {'Model':<22} {'T_out [K]':>10} {'P_out [bar]':>14}")
    print("  " + "-" * 48)
    for key in predictor.available_models():
        predictor.switch_model(key)
        out = predictor.predict_exit(**CASE)
        print(f"  {key:<22} {out['temperature_K']:>10.1f} "
              f"{out['pressure_Pa'] / 1e5:>14.3f}")


def main():
    """Run all examples; abort gracefully if no artifact is available."""
    try:
        example_exit_prediction()
        example_full_profile()
        example_compare_models()
        print("\n" + "=" * 60)
        print("All examples completed.")
        print("=" * 60)
    except FileNotFoundError as err:
        print(f"\n[ERROR] {err}")
        print("  Run notebooks/Main_4_train_tree_models.ipynb first to export the model artifact.")
        sys.exit(1)


if __name__ == "__main__":
    main()
