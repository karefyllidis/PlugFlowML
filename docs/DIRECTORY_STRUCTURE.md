# HydrAI Directory Structure Analysis
## Complete Project Structure and Compatibility Report

**Date:** February 2026  
**Status:** **FULLY COMPATIBLE** with README specifications  
**Version:** 2.3.0 – Scripts grouped under `scripts/{cluster,local,notebook,dev}`; SLURM progress JSON; smoke config for HPC tests  
**Tested:** Structure verified against repository layout (February 2026)

---

## Complete Directory Tree (Core Structure)

```
HydrAI/
├── .gitignore                           # Git ignore file
├── CHANGELOG.md                         # Version history
├── DIRECTORY_STRUCTURE.md               # This analysis document
├── STRUCTURE.md                         # Detailed structure documentation
├── LICENSE                              # MIT License
├── README.md                            # Main documentation
├── requirements.txt                     # Python dependencies
├── notebooks/
│   ├── Main_1_run_pfr.ipynb                       # Step 1: PFR simulations (Jupyter notebook)
│   ├── Main_2_generate_training_data.ipynb       # Step 2: ML training data generation (Jupyter notebook)
│   ├── Main_3_data_exploration_feature_engineering.ipynb  # Step 3: Data exploration and feature engineering
│   ├── Main_4_train_tree_models.ipynb
│   └── Main_4b_tree_models_comparison.ipynb
├── src/                                 # Source code
│   ├── cantera/                         # Cantera simulation
│   │   └── pfr_simulator.py            # Main PFR simulation
│   ├── ml/                              # ML Surrogate Models
│   │   ├── data_generation.py          # Training data generation
│   │   ├── model_training.py           # ML model training
│   │   ├── inference.py                # ML inference
│   │   └── example_usage.py            # ML examples
│   └── utils/                           # Utilities
│       └── plot_style.py               # Figure aesthetics
├── configs/                             # Configuration files
│   ├── simulation/
│   │   ├── config_template.json
│   │   ├── reactant_database.json
│   │   └── heat_flux_profile.json
│   ├── style/
│   │   └── figure_aesthetics.json
│   └── ml/
│       ├── ml_data_generation_config.json
│       ├── ml_data_generation_config.smoke.json
│       ├── ml_training_config.json
│       └── ml_inference_config.json
├── mechanisms/                          # Chemical kinetic mechanisms (YAMLs git-ignored; .gitkeep tracked)
│   └── .gitkeep                         # Add *.yaml locally per README "Required External Files"
├── data/                                # Data directory
│   ├── training/                        # Training data (generated)
│   └── raw/                             # Raw simulation data
├── models/                              # Trained ML models (generated)
├── outputs/                              # Simulation outputs
│   ├── results/                         # CSV and summary files
│   └── figures/                         # Generated plots
├── styles/                               # Figure aesthetics notes (+ optional example script)
│   └── README.md
├── docs/                                # Documentation
│   ├── API_REFERENCE.md                # Detailed API documentation
│   ├── ML_CONFIG_GUIDE.md             # ML configuration guide
│   ├── UPDATES_v3.0.md                # Version 3.0 update notes
│   └── ml/                              # ML Surrogate Models docs
│       ├── README.md
│       ├── QUICKSTART.md
│       └── IMPLEMENTATION_SUMMARY.md
├── examples/                            # Usage examples
│   └── basic_usage.py                  # Basic usage example
└── scripts/                             # cluster / local / notebook / dev
    ├── cluster/
    │   ├── run_main2_slurm_chunk.py
    │   ├── run_training_mul_CPUs.sh
    │   ├── run_training_smoke_gpu_partition.sh
    │   ├── run_trainning_mul_CPUs.sh
    │   ├── run_training_mul_GPUs.sh
    │   └── run_trainning_mul_GPUs.sh
    ├── local/
    │   ├── run_main2_local_parallel.py
    │   └── run_main1_local_simulation.sh
    ├── notebook/
    │   ├── run_simulation.sh
    │   └── run_simulation_ipynb.sh
    └── dev/
        ├── check_complete_runs.py
        └── show_structure.sh
```

### Generated / runtime files (not all tracked in git)
- **`outputs/figures/`**, **`outputs/results/`**, **`data/training/*.pkl`**, **`data/processed/*.pkl`**, **`data/figures/`** (optional EDA exports), **`models/*.joblib`** and other ML binaries (`*.pt`, `*.pth`, …)
- **`logs/`** (including **`logs/data_generation_progress_task_*.json`** — per-task Main_2 progress when using `run_main2_slurm_chunk.py`)
- **`temp/`** — conditions CSV and heat-flux JSON snippets during data generation
- **`.cursor/`**, `.env` — local IDE / environment (ignored)

See **Version control** in `README.md` and root `.gitignore` for the authoritative list.

---

## Compatibility Analysis

### **Perfect Match with README Specifications**

| Component | README Expectation | Current Status | Status |
|-----------|-------------------|----------------|--------|
| **Main Script** | `src/cantera/pfr_simulator.py` | Present | OK |
| **Step 1 Entry** | `notebooks/Main_1_run_pfr.ipynb` | Present | OK |
| **Step 2 Data Gen** | `notebooks/Main_2_generate_training_data.ipynb` | Present | OK |
| **Step 3 Exploration** | `notebooks/Main_3_data_exploration_feature_engineering.ipynb` | Present | OK |
| **Step 4 Tree ML** | `notebooks/Main_4_train_tree_models.ipynb` | Tree models (RF, GB, XGBoost, AdaBoost) | OK |
| **Database** | `configs/simulation/reactant_database.json` | Present | OK |
| **Template** | `configs/simulation/config_template.json` | Present | OK |
| **Dependencies** | `requirements.txt` | Present | OK |
| **License** | `LICENSE` | Present | OK |
| **Changelog** | `CHANGELOG.md` | Present | OK |
| **Documentation** | `README.md` | Present | OK |
| **Examples** | `examples/basic_usage.py` | Present | OK |
| **API Docs** | `docs/API_REFERENCE.md` | Present | OK |
| **Mechanisms** | `mechanisms/*.yaml` (local; not committed by default) | Add per README | OK |
| **Results** | `outputs/results/` directory | Present | OK |
| **Plots** | `outputs/figures/` directory | Present | OK |
| **Heat Flux** | `configs/simulation/heat_flux_profile.json` | Present | OK |
| **ML Models** | `src/ml/` modules | Present | OK |
| **Aesthetics** | `configs/style/figure_aesthetics.json` | Present | OK |

### **All 4 Reactants Supported**
- **Ethane** - 35 species, 135 reactions
- **Propane** - 53 species, 325 reactions  
- **Naphtha** - 1,951 species, 82,557 reactions
- **n-Hexane** - 153 species, 2,146 reactions

---

## Usage Instructions

### **Method 1: Jupyter Notebook (Recommended)**
```bash
# Launch Jupyter notebook
jupyter notebook notebooks/Main_1_run_pfr.ipynb

# Or use JupyterLab
jupyter lab notebooks/Main_1_run_pfr.ipynb
```

The notebook provides an interactive interface where you can:
- Select reactants interactively
- See real-time simulation progress
- View inline visualizations
- Modify parameters easily

### **Method 2: Bash convenience scripts (Unix)**
From the repository root:
```bash
./scripts/notebook/run_simulation.sh
# equivalent:
./scripts/local/run_main1_local_simulation.sh
```
Each script starts Jupyter and opens `notebooks/Main_1_run_pfr.ipynb`.

### **SLURM / cluster (Main_2 sweeps)**

Submit batch scripts from the repo root (see `README.md` § HPC). Override config for smoke tests:
`export HYDRAI_ML_CONFIG=$PWD/configs/ml/ml_data_generation_config.smoke.json`.  
Monitor: `tail -f logs/data_generation_progress_task_0.json` (use the task id of your worker).
Current `scripts/cluster/*.sh` values are tuned for the University of Cambridge **CSD3** environment; adjust account/partition/QoS/modules for other SLURM clusters.

---

## Test Results

Prior validation used the Jupyter workflow and tree-ML pipeline; SLURM filenames vary by site (edit `#SBATCH` headers before submitting).

### **Sample test output (notebook / CLI-style)**
```
Running simulation for: Ethane
Loaded configuration for Ethane Pyrolysis PFR Simulation
Version: 2.0
Gas mechanism contains 35 species and 135 reactions
...
SIMULATION COMPLETED SUCCESSFULLY!
[OK] Ethane conversion: 0.8%
[OK] Temperature rise: 80.7 K
[OK] Pressure drop: 0.24 bar
[OK] Residence time: 0.030 s
```

---

## Key Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `notebooks/Main_1_run_pfr.ipynb` | Step 1: Main entry point (Jupyter notebook) | Working |
| `notebooks/Main_2_generate_training_data.ipynb` | Step 2: ML data generation (Jupyter notebook) | Working |
| `notebooks/Main_3_data_exploration_feature_engineering.ipynb` | Step 3: Data exploration and feature engineering (Jupyter notebook) | Working |
| `notebooks/Main_4_train_tree_models.ipynb` | Step 4: Tree-based ML training (RF, GB, XGBoost) | OK |
| `notebooks/Main_4b_tree_models_comparison.ipynb` | Step 5: Model comparison metrics & plots | OK |
| `src/cantera/pfr_simulator.py` | Main simulation code | Working |
| `configs/simulation/reactant_database.json` | Reactant definitions | Complete |
| `configs/simulation/config_template.json` | Configuration template | Valid |
| `requirements.txt` | Dependencies list | All installed |
| `scripts/cluster/run_main2_slurm_chunk.py` | Main-2 worker (`TASK_ID` / `NTASKS`) | Use on HPC |
| `scripts/cluster/run_training_mul_CPUs.sh` | Example multi-CPU SLURM job | Site-specific `#SBATCH` |
| `scripts/cluster/run_training_smoke_gpu_partition.sh` | Short smoke test (tiny config) | Edit account/partition |
| `scripts/local/run_main2_local_parallel.py` | Multi-process Main_2 on one machine | All OS |
| `scripts/dev/check_complete_runs.py` | Sweep completeness / manifests | Run from repo root |
| `scripts/notebook/run_simulation.sh` | Opens Main_1 in Jupyter | Working |
| `configs/simulation/heat_flux_profile.json` | Heat flux data | Present |
| `configs/style/figure_aesthetics.json` | Figure styling | Present |

---

## Recommendations

1. **Use Jupyter notebooks** (`notebooks/Main_1_run_pfr.ipynb`, `notebooks/Main_2_generate_training_data.ipynb`, etc.) for interactive execution in pipeline order
2. **All dependencies are properly installed** via pip
3. **Directory structure has been restructured** for better organization (see STRUCTURE.md)
4. **System is ready for production use** with all 4 reactants
5. **ML Surrogate Models available** for fast predictions (see docs/ml/)
6. **Centralized figure aesthetics** in `configs/style/figure_aesthetics.json` (see `styles/README.md`)
7. **Interactive workflows** via Jupyter notebooks for better user experience

---

## Notes

- **Dependencies**: Installed via pip (no virtual environment required)
- **Generated files**: Primary outputs live under `outputs/figures/` and `outputs/results/` (legacy `fig/` / `results/` at repo root may appear in older clones)
- **Compatibility**: The directory structure exactly matches the README specifications
- **Functionality**: All core features are working and tested
- **Export Controls**: New v2.1.0 feature allows optional CSV and plot generation via configuration flags

## Version 2.1.0 Features

### Export Controls
- **`if_csv_out`**: Control CSV data export (245+ columns)
- **`if_plot_out`**: Control plot generation (18+ figures)
- **Flexible Workflows**: Support for data-only, plots-only, or simulation-only modes
- **Performance Optimization**: Faster parameter studies without export overhead

---

**Conclusion**: Your HydrAI project directory structure is **perfectly compatible** with the README specifications and fully functional. The system is ready for use with all supported reactants.
