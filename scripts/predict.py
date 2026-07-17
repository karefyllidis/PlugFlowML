#!/usr/bin/env python3
"""PlugFlowML inference script — run surrogate predictions from the command line.

Supports tree-ensemble models (Main_4 / Main_5) and the PyTorch SimpleNN
(Main_6 full-profile). Pass reactor inlet conditions as CLI arguments;
results are printed to stdout and optionally saved to CSV.

Usage examples
--------------
Exit-plane prediction (tree model):
    python scripts/predict.py \\
        --T 850 --P 2.5 --L 12 --D 0.032 --mdot 0.07 --q 180000

Full axial profile (NN model, 200 points):
    python scripts/predict.py \\
        --model nn --mode full_profile \\
        --T 850 --P 2.5 --L 12 --D 0.032 --mdot 0.07 --q 180000 \\
        --n-points 200 --output profile.csv

MC-Dropout uncertainty (NN, full profile only):
    python scripts/predict.py \\
        --model nn --mode full_profile --mc-samples 50 \\
        --T 850 --P 2.5 --L 12 --D 0.032 --mdot 0.07 --q 180000

From JSON (batch of conditions):
    python scripts/predict.py --json conditions.json --output results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_tree_predictor(models_dir: Path, mode: str, model_key: str):
    from src.ml.inference import MLPFRPredictor
    return MLPFRPredictor(models_dir, model_key=model_key or None, mode=mode)


def _load_nn_predictor(models_dir: Path, mode: str):
    """Load the full-profile SimpleNN (Main_6) from exported state_dict + scalers."""
    import joblib
    import torch
    from src.models import SimpleNN

    if mode != "full_profile":
        raise ValueError("--model nn only supports --mode full_profile (Main_6 SimpleNN).")

    stem = "simple_nn_full_profile"
    stem_dir = models_dir / stem
    manifest_path = stem_dir / f"{stem}_manifest.json"
    scalers_path  = stem_dir / f"{stem}_scalers.joblib"
    state_path    = stem_dir / f"{stem}_state_dict.pt"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run Main_6 with IF_MODEL_EXPORT=True."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    arch    = manifest["architecture"]
    scalers = joblib.load(scalers_path)
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers.get("scaler_y")

    device = "cpu"
    model = SimpleNN(
        arch["in_features"], arch["h1"], arch["h2"], arch["h3"],
        arch["out_features"], dropout=arch["dropout"],
    )
    model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    model.eval()

    return model, scaler_X, scaler_y, manifest


def _load_pinn_predictor(models_dir: Path):
    """Load a PINNPFR (Main_7) from exported state_dict + scalers."""
    import joblib
    import torch
    from src.models import PINNPFR

    stem = "pinn_pfr"
    stem_dir = models_dir / stem
    manifest_path = stem_dir / f"{stem}_manifest.json"
    scalers_path  = stem_dir / f"{stem}_scalers.joblib"
    state_path    = stem_dir / f"{stem}_state_dict.pt"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run Main_7 with IF_MODEL_EXPORT=True."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    arch    = manifest["architecture"]
    scalers = joblib.load(scalers_path)
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers.get("scaler_y")

    device = "cpu"
    model = PINNPFR(
        arch["in_features"], arch["h1"], arch["h2"], arch["h3"],
        arch["out_features"], dropout=arch["dropout"],
    )
    model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    model.eval()

    return model, scaler_X, scaler_y, manifest


def _build_inlet_row(args, feature_cols: list[str]) -> np.ndarray:
    """Build a single-row feature array from CLI args in the correct column order."""
    values = {
        "initial_temperature_K": args.T,
        "initial_pressure_Pa":   args.P * 1e5,   # bar → Pa
        "reactor_length_m":      args.L,
        "reactor_diameter_m":    args.D,
        "mass_flow_rate_kgps":   args.mdot,
        "heat_flux_Wm2":         args.q,
    }
    return np.array([[values.get(c, 0.0) for c in feature_cols]], dtype=float)


# ── Prediction routines ───────────────────────────────────────────────────────

def predict_tree(args) -> pd.DataFrame:
    models_dir = PROJECT_ROOT / "models" / "tree_baseline"
    pred = _load_tree_predictor(models_dir, mode=args.mode, model_key=args.model_key)

    if args.mode == "full_profile":
        return pred.predict_profile(
            initial_temperature_K=args.T,
            initial_pressure_Pa=args.P * 1e5,
            reactor_length_m=args.L,
            reactor_diameter_m=args.D,
            mass_flow_rate_kgps=args.mdot,
            heat_flux_Wm2=args.q,
            n_points=args.n_points,
        )
    else:
        result = pred.predict_exit(
            initial_temperature_K=args.T,
            initial_pressure_Pa=args.P * 1e5,
            reactor_length_m=args.L,
            reactor_diameter_m=args.D,
            mass_flow_rate_kgps=args.mdot,
            heat_flux_Wm2=args.q,
        )
        return pd.DataFrame([result])


def predict_nn(args) -> pd.DataFrame:
    import torch

    models_dir = PROJECT_ROOT / "models"  # _load_nn_predictor resolves the stem subfolder
    model, scaler_X, scaler_y, manifest = _load_nn_predictor(models_dir, mode=args.mode)

    feature_cols = manifest.get("inlet_cols") or manifest.get("feature_cols", [])
    target_cols  = manifest.get("target_cols", [])

    z_vals = np.linspace(0.0, args.L, args.n_points)
    rows = []
    for z in z_vals:
        row = {
            "initial_temperature_K": args.T,
            "initial_pressure_Pa":   args.P * 1e5,
            "reactor_length_m":      args.L,
            "reactor_diameter_m":    args.D,
            "mass_flow_rate_kgps":   args.mdot,
            "heat_flux_Wm2":         args.q,
            "z_position_m":          z,
            "relative_position":     z / args.L if args.L > 0 else 0.0,
        }
        rows.append([row.get(c, 0.0) for c in feature_cols])
    X_raw = np.array(rows, dtype=float)

    X_scaled = scaler_X.transform(X_raw)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    if args.mc_samples > 0:
        mean, std = model.predict_with_uncertainty(X_t, n_samples=args.mc_samples, scaler_y=scaler_y)
        df_mean = pd.DataFrame(mean, columns=target_cols)
        df_std  = pd.DataFrame(std,  columns=[f"{c}_std" for c in target_cols])
        df = pd.concat([df_mean, df_std], axis=1)
    else:
        model.eval()
        import torch
        with torch.no_grad():
            y_s = model(X_t).numpy()
        y = scaler_y.inverse_transform(y_s) if scaler_y is not None else y_s
        df = pd.DataFrame(y, columns=target_cols)

    df.insert(0, "z_position_m", z_vals)
    df.insert(1, "relative_position", df["z_position_m"] / args.L)

    return df


def predict_pinn(args) -> pd.DataFrame:
    """PINNPFR full axial profile prediction (Main_7 artefacts)."""
    import torch

    models_dir = PROJECT_ROOT / "models"  # _load_pinn_predictor resolves the stem subfolder
    model, scaler_X, scaler_y, manifest = _load_pinn_predictor(models_dir)

    feature_cols = manifest.get("feature_cols") or manifest.get("inlet_cols", [])
    target_cols  = manifest.get("target_cols", [])

    z_vals = np.linspace(0.0, args.L, args.n_points)
    rows = []
    for z in z_vals:
        row = {
            "initial_temperature_K": args.T,
            "initial_pressure_Pa":   args.P * 1e5,
            "reactor_length_m":      args.L,
            "reactor_diameter_m":    args.D,
            "mass_flow_rate_kgps":   args.mdot,
            "heat_flux_Wm2":         args.q,
            "z_position_m":          z,
            "relative_position":     z / args.L if args.L > 0 else 0.0,
        }
        rows.append([row.get(c, 0.0) for c in feature_cols])
    X_raw = np.array(rows, dtype=float)

    X_scaled = scaler_X.transform(X_raw)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_s = model(X_t).numpy()
    y = scaler_y.inverse_transform(y_s) if scaler_y is not None else y_s
    df = pd.DataFrame(y, columns=target_cols)
    df.insert(0, "z_position_m", z_vals)
    df.insert(1, "relative_position", z_vals / args.L)
    return df


def predict_from_json(args) -> pd.DataFrame:
    """Run exit-plane predictions for a batch of conditions from a JSON file."""
    with open(args.json) as f:
        conditions = json.load(f)

    if isinstance(conditions, dict):
        conditions = [conditions]

    results = []
    for cond in conditions:
        args.T    = float(cond.get("temperature_K",       cond.get("T", 850)))
        args.P    = float(cond.get("pressure_bar",        cond.get("P", 2.5)))
        args.L    = float(cond.get("reactor_length_m",    cond.get("L", 12.0)))
        args.D    = float(cond.get("reactor_diameter_m",  cond.get("D", 0.032)))
        args.mdot = float(cond.get("mass_flow_rate_kgps", cond.get("mdot", 0.07)))
        args.q    = float(cond.get("heat_flux_Wm2",       cond.get("q", 150_000)))

        if args.model == "nn":
            fn = predict_nn
        elif args.model == "pinn":
            fn = predict_pinn
        else:
            fn = predict_tree
        df = fn(args)
        for k, v in cond.items():
            df[f"input_{k}"] = v
        results.append(df)

    return pd.concat(results, ignore_index=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PlugFlowML surrogate inference. Predict PFR outlet or full axial profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["tree", "nn", "pinn"], default="tree",
                   help="Surrogate type: 'tree' (Main_4/5), 'nn' (Main_6 SimpleNN, full_profile only), 'pinn' (Main_7 PINNPFR)")
    p.add_argument("--mode",  choices=["exit", "full_profile"], default="exit",
                   help="Predict exit-plane only, or full axial profile ('nn' model requires full_profile)")
    p.add_argument("--model-key", default=None,
                   help="Tree model key, e.g. 'xgboost', 'gradient_boosting' (tree only)")

    # Reactor conditions
    p.add_argument("--T",    type=float, default=850.0,    help="Inlet temperature [K]")
    p.add_argument("--P",    type=float, default=2.5,      help="Inlet pressure [bar]")
    p.add_argument("--L",    type=float, default=12.0,     help="Reactor length [m]")
    p.add_argument("--D",    type=float, default=0.032,    help="Reactor diameter [m]")
    p.add_argument("--mdot", type=float, default=0.07,     help="Mass flow rate [kg/s]")
    p.add_argument("--q",    type=float, default=150_000.0, help="Wall heat flux [W/m²]")
    p.add_argument("--n-points", type=int, default=200,    help="Axial points for full_profile mode")

    # Batch / UQ
    p.add_argument("--json",       default=None, help="JSON file with a list of inlet conditions for batch prediction")
    p.add_argument("--mc-samples", type=int, default=0,
                   help="MC-Dropout samples for uncertainty (nn model only; 0 = deterministic)")
    p.add_argument("--output", default=None, help="CSV path to save results (default: print to stdout)")

    return p


def main():
    args = _build_parser().parse_args()

    if args.json:
        df = predict_from_json(args)
    elif args.model == "nn":
        df = predict_nn(args)
    elif args.model == "pinn":
        df = predict_pinn(args)
    else:
        df = predict_tree(args)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"[OK] {len(df)} row(s) saved to {args.output}")
    else:
        pd.set_option("display.max_columns", 10)
        pd.set_option("display.width", 120)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
