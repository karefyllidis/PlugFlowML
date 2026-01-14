# HydrAI v3.0.0 - Major Updates

## Overview

Version 3.0.0 represents a major restructuring and enhancement of the HydrAI project, introducing ML Surrogate Models, improved organization, and enhanced usability.

## Major Changes

### 1. Project Restructuring

**New Directory Organization:**
- `src/` - All source code organized into packages
  - `src/cantera/` - Cantera-based simulation code
  - `src/ml/` - ML Surrogate Models
  - `src/utils/` - Utility modules (plot styling, etc.)
- `configs/` - All configuration files centralized
- `mechanisms/` - Chemical kinetic mechanisms (renamed from `mechanism/`)
- `data/` - Training data directory
- `models/` - Trained ML models directory
- `outputs/` - All simulation outputs
  - `outputs/results/` - CSV and summary files
  - `outputs/figures/` - Generated plots
- `styles/` - Figure aesthetics configuration

**Benefits:**
- Clear separation of concerns
- Professional Python package structure
- Easy to navigate and maintain
- Scalable for future additions

### 2. ML Surrogate Models

**Complete ML Framework:**
- Training data generation with parameter sweeps
- Multiple ML algorithms (Neural Networks, Random Forest, XGBoost, Gradient Boosting)
- Fast inference (100-1000x speedup)
- JSON-based configuration for reproducibility

**Key Files:**
- `src/ml/data_generation.py` - Generate training datasets
- `src/ml/model_training.py` - Train ML models
- `src/ml/inference.py` - Fast ML predictions
- `configs/ml_*_config.json` - Configuration files

**Documentation:**
- `docs/ml/README.md` - Comprehensive guide
- `docs/ml/QUICKSTART.md` - Quick start
- `docs/ML_CONFIG_GUIDE.md` - Configuration guide

### 3. JSON Configuration System

**All ML workflows now use JSON configs:**
- No more command-line argument parsing
- Reproducible configurations
- Easy to version control
- Self-documenting

**Config Files:**
- `configs/ml_data_generation_config.json` - Training data generation
- `configs/ml_training_config.json` - Model training
- `configs/ml_inference_config.json` - Inference/prediction

### 4. Centralized Figure Aesthetics

**New Styling System:**
- `styles/figure_aesthetics.json` - Centralized styling configuration
- `src/utils/plot_style.py` - Utility functions for applying aesthetics
- Consistent appearance across all plots
- Easy customization

**Features:**
- Colors, fonts, line styles all configurable
- Individual profile settings
- Professional, publication-ready figures

### 5. Updated Entry Points

**New Main Entry Point:**
- `run_pfr.ipynb` - Main interactive entry point for simulations (Jupyter notebook)
- Replaces direct execution of `Main_GeneralizedPFR.py`

**Updated Scripts:**
- `scripts/run_simulation.sh` - Updated to use new entry point
- All paths updated to new structure

## Migration Guide

### For Existing Users

1. **Update Imports:**
   ```python
   # Old
   from Main_GeneralizedPFR import load_reactant_database
   
   # New
   from src.cantera.pfr_simulator import load_reactant_database
   ```

2. **Update File Paths:**
   - Config files: `configs/`
   - Mechanisms: `mechanisms/`
   - Outputs: `outputs/results/` and `outputs/figures/`

3. **Update Execution:**
   ```bash
   # Old
   python Main_GeneralizedPFR.py --reactant ethane
   
   # New
   jupyter notebook run_pfr.ipynb
   ```

### For ML Users

1. **Create Config Files:**
   - Copy templates from `configs/`
   - Customize for your needs

2. **Update Script Calls:**
   ```bash
   # Old (command-line args)
   python src/ml/data_generation.py configs/ml_data_generation_config.json
   
   # New (JSON config)
   python src/ml/data_generation.py configs/ml_data_generation_config.json
   ```

## New Features

### ML Surrogate Models
- Generate training data from Cantera simulations
- Train multiple ML algorithms
- Fast predictions (100-1000x speedup)
- High accuracy (R² > 0.95)

### Figure Aesthetics
- Centralized styling configuration
- Consistent appearance
- Easy customization

### JSON Configuration
- Reproducible workflows
- Version control friendly
- Self-documenting

## Documentation Updates

All documentation has been updated:
- `README.md` - Updated structure and features
- `CHANGELOG.md` - Version 3.0.0 entry
- `STRUCTURE.md` - New directory organization
- `DIRECTORY_STRUCTURE.md` - Updated paths
- `docs/ml/` - ML Surrogate Models documentation
- `docs/ML_CONFIG_GUIDE.md` - Configuration guide

## Breaking Changes

1. **File Locations:**
   - Main code: `src/cantera/pfr_simulator.py`
   - Configs: `configs/`
   - Mechanisms: `mechanisms/`
   - Outputs: `outputs/`

2. **Import Paths:**
   - All imports updated to new package structure

3. **ML Scripts:**
   - Now require JSON config files instead of command-line arguments

## Backward Compatibility

- Helper functions handle path resolution automatically
- Old relative paths still work via path utilities
- Configuration system maintains compatibility

## Next Steps

1. Review updated documentation
2. Update any custom scripts to use new paths
3. Create JSON configs for ML workflows
4. Explore ML Surrogate Models for fast predictions
5. Customize figure aesthetics in `styles/figure_aesthetics.json`

## Support

For questions or issues:
- Check `README.md` for general usage
- Check `docs/ml/README.md` for ML workflows
- Check `docs/ML_CONFIG_GUIDE.md` for configuration
- Check `STRUCTURE.md` for directory organization
