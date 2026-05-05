"""
Smoke tests — fast checks that exercise the public surface without GPUs,
heavy datasets or Cantera simulations. Designed to run in seconds on CI.
"""
import importlib
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAS_CANTERA = importlib.util.find_spec("cantera") is not None


# ── 1. Import surface ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CANTERA, reason="cantera not installed")
def test_import_data_generator():
    """The TrainingDataGenerator class must be importable from src.ml."""
    from src.ml.data_generation import TrainingDataGenerator
    assert callable(TrainingDataGenerator)


def test_import_inference():
    """MLPFRPredictor must be importable; missing artifact errors are OK."""
    from src.ml.inference import MLPFRPredictor
    assert callable(MLPFRPredictor)


def test_import_plot_style():
    """The plot-style helper used by every notebook must import cleanly."""
    from src.utils.plot_style import setup_matplotlib
    setup_matplotlib()


# ── 2. Configuration files ────────────────────────────────────────────────────

@pytest.mark.parametrize("config_name", [
    "ml_data_generation_config.json",
    "ml_training_config.json",
    "ml_inference_config.json",
    "reactant_database.json",
    "heat_flux_profile.json",
])
def test_config_files_are_valid_json(config_name):
    """Every shipped JSON config must parse without errors."""
    path = PROJECT_ROOT / "configs" / config_name
    assert path.exists(), f"Missing config file: {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), f"{config_name} must be a JSON object"


def test_reactant_database_mechanisms_exist():
    """Every reactant entry must reference a mechanism YAML file (Cantera-bundled mechs are skipped)."""
    db_path = PROJECT_ROOT / "configs" / "reactant_database.json"
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    reactants = db.get("reactants", {})
    assert reactants, "Reactant database is empty"

    cantera_bundled = {"gri30.yaml", "h2o2.yaml", "air.yaml"}
    for key, info in reactants.items():
        mech = info.get("mechanism_file")
        if not mech or Path(mech).name in cantera_bundled:
            continue
        mech_path = PROJECT_ROOT / mech
        assert mech_path.exists(), (
            f"Mechanism file missing for '{key}': {mech_path}"
        )


# ── 3. Generator instantiation (no simulation) ────────────────────────────────

@pytest.mark.skipif(not HAS_CANTERA, reason="cantera not installed")
def test_training_generator_instantiates(tmp_path):
    """TrainingDataGenerator must instantiate without running any simulation."""
    from src.ml.data_generation import TrainingDataGenerator

    gen = TrainingDataGenerator(output_dir=str(tmp_path), disable_plots=True)
    assert hasattr(gen, "param_ranges")
    assert hasattr(gen, "generate_dataset")
