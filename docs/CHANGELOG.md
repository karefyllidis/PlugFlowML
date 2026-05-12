# Changelog

All notable changes to the Generalized PFR Simulation System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Parallel-axes EDA plots** - New `src/utils/plot_parallel.py` (matplotlib-only) with two reusable helpers: `plot_parallel_coordinates` (Inselberg-style polylines for continuous multidimensional data) and `plot_parallel_sets` (Kosara-style categorical ribbons whose width is proportional to joint count and color encodes the mean of an outcome variable). Both are re-exported from `src.utils` and default to the `magma` colormap. Used by **Main_3 Section 2.5** (new) to visualize the 6D inlet design space (T, P, L, D, ṁ, q″) of the training set as one row per PFR run, colored by exit-plane n-hexane conversion; figures saved to `outputs/figures/Main_3_data_exploration_feature_engineering/eda/parallel_coordinates_design_space.png` and `parallel_sets_design_space.png` when `IF_SAVE_EDA_PLOTS=True`. No new dependencies.
- **XGBoost in hyperparameter tuning** - Main_4 Section 7 now includes XGBoost in param grids and RandomizedSearchCV; N_ITER=100, CV=5 for tuning
- **Limited-data recommendations** - Main_4 intro lists tips for few samples (smaller test_size, tuning, more runs); optional runtime hint when sample count &lt; threshold
- **Main_3 export** - Optional export of `df_features` and `df_target` to `data/processed/` as a single pickle (`features_targets_*.pkl`) via **IF_EXPORT_FEATURES_TARGETS** and **EXPORT_DIR**
- **Main_3 NaN handling** - Load step drops rows with NaN and reports how many were dropped
- **Structured grid sampling** - `sampling_method: "full_grid"`, `"structured_grid"`, or `"grid"` uses all combinations from `parameter_ranges` ([min, max, n_points] per parameter); total runs = product of n_points
- **Pipeline-order notebook names** - Notebooks renamed to `Main_1_run_pfr.ipynb`, `Main_2_generate_training_data.ipynb`, `Main_3_data_exploration_feature_engineering.ipynb`, `Main_4_train_tree_models.ipynb` to reflect workflow steps
- **AdaBoost** - Tree-based training notebook (`Main_4_train_tree_models.ipynb`) now includes AdaBoost; config section `adaboost` in `configs/ml/ml_training_config.json`
- **Figure aesthetics path** - `figure_aesthetics.json` lives under **`configs/style/`** (preferred); `load_aesthetics()` falls back to flat `configs/figure_aesthetics.json` or legacy `styles/figure_aesthetics.json`

### Documentation
- **Contributing** - `CONTRIBUTING.md` moved to `.github/CONTRIBUTING.md` (GitHub conventions); links updated in README, CHANGELOG, `.githooks/README.md`.
- **Git / repo hygiene** - Expanded `.gitignore` (`logs/`, `data/figures/`, PyTorch/other ML export globs, `.cursor/`, `.env` / `.env.local`, `.ruff_cache/`); README **Version control** subsection; `.github/CONTRIBUTING.md` (Git and large files); `STRUCTURE.md` §8; `DIRECTORY_STRUCTURE.md` generated-files list and mechanisms note. README **Required External Files** aligned with ignored `mechanisms/*.yaml`.
- Full pass (historical): README, STRUCTURE.md, DIRECTORY_STRUCTURE.md, docs/ml (README, QUICKSTART, IMPLEMENTATION_SUMMARY), ML_CONFIG_GUIDE, API_REFERENCE, UPDATES_v3.0, CHANGELOG, scripts. All references use `Main_N_` notebook names; tree models (RF, GB, XGBoost, AdaBoost) and joblib artifacts documented; training data described as pkl-primary; "Coming Soon" removed for tree training

### Changed
- **No bold text on figures (project-wide)** - All figure elements (titles, axis labels, axis names, tick labels, legend entries, annotations, colorbar labels, suptitles) now render with `fontweight='normal'`. `setup_matplotlib()` in `src/utils/plot_style.py` locks `axes.titleweight`, `axes.labelweight`, and `figure.titleweight` to `'normal'`; `configs/style/figure_aesthetics.json` `font.title_weight` changed from `"bold"` to `"normal"`; `fontweight='bold'` stripped from `src/utils/plot_parallel.py`, `src/cantera/pfr_simulator.py` defaults, and notebooks Main_3, Main_4, Main_5. Rule documented in `.cursor/rules/HYDRAI_PROJECT_CONVENTIONS.md` §5.9 (also in Critical Don'ts). Use a larger `fontsize` if more visual weight is needed.
- **Neural network training script** — Removed TensorFlow/Keras from `model_training.py`. Deep models are planned on **PyTorch** (`train_neural_network` is a no-op with a one-time notice until implemented). No import-time TensorFlow warning.
- **Config folder layout** - JSON configs grouped under `configs/simulation/` (reactant DB, PFR template, heat flux), `configs/ml/` (data generation, training, inference), and `configs/style/` (`figure_aesthetics.json`). `plot_style` and `pfr_simulator` resolve new paths; legacy flat `configs/figure_aesthetics.json` and `configs/heat_flux_profile.json` in templates still work via fallbacks where noted.
- **ML config defaults** - `gradient_boosting` and `xgboost` use `n_estimators: 150` in `configs/ml/ml_training_config.json` when tuning is off
- **Main_3 documentation** - Concise markdown docs and Summary aligned with actual steps (load, drop NaNs, organize, export)
- **ML config** - `parameter_ranges` is documented and used for grid/structured_grid/full_grid only; `random_sample_bounds` for random/LHS. Removed redundant `random_sample` config key (sampling controlled only by `sampling_method`)

## [3.0.3] - 2025-01-17

### Added
- **Latin Hypercube Sampling (LHS)** - Parameter space sampling via `sampling_method: "latin"` or `"latin_hypercube"` for better coverage with fewer runs; config supports `"random"` or `"latin"` with `lhs_seed` for reproducibility
- **Training space visualization** - In `Main_2_generate_training_data.ipynb`: Step 2.1 (sampling preview) and Step 4.1 (from generated data) with 1D marginals and 2D pairwise scatter plots to assess exploration quality
- **Run control flags** in `Main_2_generate_training_data.ipynb`: `IF_SHOW_PLOTS`, `IF_SAVE_PLOTS`, `IF_SAVE_METADATA`, `IF_SAVE_TRAINING_DATA` to control display/saving of plots, metadata JSON, and training data (pkl/csv)
- **Optional saving** - `generate_dataset()` accepts `save_metadata` and `save_training_data`; when False, metadata or training files are not written (dataset still returned in memory)

### Changed
- **ML config** - `random_sample_bounds` applies to both random and Latin Hypercube sampling; config key `sampling_method` chooses `"random"` or `"latin"`
- **Data exploration notebook** - Column categories: inlet conditions, reactor design, operating conditions, spatial coordinates, state variables, thermodynamic properties, species (Y/X); ML-ready feature/target separation

### Documentation
- Updated README, ML_CONFIG_GUIDE, docs/ml (README, QUICKSTART, IMPLEMENTATION_SUMMARY) for LHS, flags, and training space visualization

## [3.0.2] - 2025-01-16

### Added
- **Pickle file format** - Partial saves now use pickle format (`.pkl`) for faster I/O during data generation
- **Automatic cleanup** - Partial files are automatically deleted after successful completion to save disk space
- **Real-time progress tracking** - Enhanced progress display showing current simulation count, success rate, failed simulations, data points collected, and ETA after each simulation
- **Memory efficiency** - Data is cleared from memory after each save to prevent unbounded memory growth
- **Data exploration notebook** - New `Main_3_data_exploration_feature_engineering.ipynb` notebook for exploring generated training data and performing feature engineering

### Changed
- **File format** - Final dataset saved as both pickle (`.pkl`) and CSV (`.csv`) formats for compatibility
- **Progress display** - Progress information now appears after every simulation for early visibility
- **Documentation** - Updated all ML documentation to reflect new file formats, cleanup behavior, and JSON config usage

### Fixed
- **Memory growth** - Fixed issue where partial save files contained all previous data, causing unbounded file size growth
- **Documentation** - Updated usage examples to use JSON config files instead of deprecated command-line arguments

## [3.0.1] - 2025-01-14

### Fixed
- **Import order** - Fixed namespace conflict by importing `cantera` before adding `src` to `sys.path` in Jupyter notebooks
- **Species access** - Fixed `IndexError` when accessing species from `SolutionArray` using correct 2D indexing (`states1.Y[:, species_idx]`)
- **Species name format** - Added handling for species names with `:1` suffix from database vs mechanism files
- **Temporary files organization** - Moved temporary heat flux files from project root to dedicated `temp/` directory

### Added
- **Combined visualization** - New plot showing reactant consumption and product formation together in single graph
- **Enhanced notebook** - Improved inline visualizations with better error handling and fallback mechanisms
- **Temporary directory** - Created `temp/` directory for temporary files generated during data generation (automatically cleaned up)
- **Warning suppression** - Suppressed all `[WARNING]` messages from Cantera's SUNDIALS solver to reduce output noise during data generation
- **Random sampling bounds** - Added `random_sample_bounds` configuration option to constrain random sampling to specific parameter ranges
- **Concise configuration** - Simplified config files by moving detailed documentation to `docs/ML_CONFIG_GUIDE.md`
- **Parallel processing** - Added multiprocessing support for training data generation with `n_jobs` parameter (1=sequential, -1=all CPUs, N=specific number of cores)

## [3.0.0] - 2025-01-14

### Added
- **Project restructuring** - Complete reorganization into professional package structure
  - `src/` directory for all source code (cantera, ml, utils modules)
  - `configs/` directory for all configuration files
  - `data/` directory for training data
  - `models/` directory for trained ML models
  - `outputs/` directory for simulation results and figures
  - `styles/` directory for figure aesthetics configuration
- **ML Surrogate Models** - Complete machine learning framework
  - Training data generation with parameter sweeps
  - Multiple ML algorithms (Neural Networks, Random Forest, XGBoost, Gradient Boosting)
  - Fast inference (100-1000x speedup over Cantera)
  - JSON-based configuration for all ML workflows
- **Centralized figure aesthetics** - `styles/figure_aesthetics.json` for consistent styling
- **JSON configuration system** - All ML scripts use JSON config files instead of command-line arguments
- **Utility modules** - `src/utils/plot_style.py` for figure styling utilities
- **Enhanced documentation** - Updated all docs to reflect new structure

### Changed
- **Directory structure** - Complete reorganization for better scalability
  - `Main_GeneralizedPFR.py` → `src/cantera/pfr_simulator.py`
  - Mechanism files directory: `mechanisms/`
  - `results/` and `fig/` → `outputs/results/` and `outputs/figures/`
  - Config files moved to `configs/`
- **Entry point** - New `Main_1_run_pfr.ipynb` as main interactive entry point (Jupyter notebook)
- **Import paths** - All imports updated to use new package structure
- **Terminology** - Removed "Phase B" references, now "ML Surrogate Models"
- **ML workflows** - Switched from command-line arguments to JSON configuration files
- **File paths** - All paths now relative to project root with helper functions

### Fixed
- **Path resolution** - All file paths now work regardless of execution location
- **Import errors** - Fixed all import paths after restructuring
- **Documentation** - Updated all documentation to reflect new structure

### Technical Details
- **New structure**: Professional Python package organization
- **Backward compatibility**: Old paths still work via helper functions
- **Configuration**: JSON-based configs for reproducibility
- **Styling**: Centralized aesthetics for consistent figures

## [2.1.0] - 2025-09-20

### Added
- **Comprehensive data export** - 245+ columns of simulation data including all thermodynamic and transport properties
- **Enhanced visualization** - 18+ professional plots including viscosity, thermal conductivity, and heat capacity profiles
- **Export controls** - Optional CSV and plot generation control with `if_csv_out` and `if_plot_out` flags
- **Flexible workflows** - Support for data-only, plots-only, or simulation-only modes

### Changed
- **Export compatibility** - Removed properties not available in current Cantera version to ensure stability
- **Documentation updates** - Updated README and API documentation to reflect comprehensive export capabilities
- **Configuration system** - Added export controls section to configuration template
- **Performance optimization** - Export operations now optional for faster parameter studies

### Fixed
- **Export stability** - Fixed issues with unavailable Cantera properties causing simulation crashes
- **Figure generation** - Ensured all 18 plots are generated successfully
- **CSV export** - Guaranteed 245+ columns of data export without errors

## [2.0.1] - 2025-09-20

### Added
- **Relative position heat flux** - Heat flux profiles now use relative positions (0.0-1.0) for universal applicability
- **Step-wise interpolation** - Option for step-wise heat flux interpolation in addition to linear
- **Automatic heat flux plotting** - Heat flux vs. relative position figure generated automatically
- **Complete species data** - Mass and mole fractions for all species in CSV export
- **Transport properties** - Viscosity and thermal conductivity data export and visualization
- **Thermodynamic completeness** - Internal energy, Gibbs free energy, and heat capacity ratio export

### Changed
- **Heat flux profile format** - Updated to use relative positions (0.0-1.0) instead of absolute meters

## [2.0.0] - 2025-09-18

### Added
- **Generalized multi-reactant support** - System now supports ethane, propane, naphtha, and n-hexane
- **Automatic configuration generation** - Dynamic configuration creation for each reactant type
- **Species name standardization** - Intelligent handling of different naming conventions across mechanisms
- **Professional file naming** - Systematic output file naming with reactant identification
- **Comprehensive documentation** - Professional README, API documentation, and code comments
- **Reactant database system** - JSON-based database for easy reactant management
- **Enhanced error handling** - Robust error handling with informative messages
- **Flexible product tracking** - Configurable target products for each reactant
- **Batch processing support** - Easy execution of multiple simulations
- **Automatic geometry calculations** - Reactor volume and surface area calculated from geometry
- **Realistic heat flux profiles** - Heat flux for high-temperature pyrolysis (150,000 W/m²)
- **Enhanced plotting system** - Individual plots for each variable with expanded product species
- **GitHub repository** - Public repository at https://github.com/karefyllidis/HydrAI

### Changed
- **Mechanism file naming** - Standardized to `[Reactant]_Kinetic-Model.yaml` format
- **Configuration system** - Moved from hardcoded to template-based configuration
- **Output structure** - Enhanced CSV and summary file formats
- **Code organization** - Modular design with clear separation of concerns
- **Heat flux profile** - Updated to realistic values (150,000 W/m²) with 6 data points
- **Reactor geometry** - Volume and surface area now calculated automatically per step
- **Pressure specification** - Clarified as absolute pressure (2.0 bar absolute)

### Fixed
- **Species name mismatches** - Resolved issues with different naming conventions
- **Configuration inconsistencies** - Standardized configuration across all reactants
- **File path issues** - Fixed relative path problems in different environments
- **Numerical stability** - Fixed reactor volume/surface area calculation for solver stability
- **Simulation length** - Fixed simulation to stop at configured reactor length
- **Transport properties** - Added handling for mechanisms without transport properties

### Technical Details
- **New files**: `Main_GeneralizedPFR.py`, `run_simulation.py`, `reactant_database.json`, `config_template.json`
- **Updated files**: All mechanism files renamed for consistency
- **Dependencies**: Updated to require Cantera 3.1.0+, Python 3.8+

## [1.2.0] - 2025-09-16

### Added
- **Simplified installation** - Cantera and dependencies installed via pip (no virtual environment required)
- **Enhanced documentation** - Improved project structure and documentation
- **Heat flux profile information** - Added heat flux data to summary files

### Changed
- **Project structure** - Cleaned up and organized project files
- **Documentation** - Removed personal paths and improved clarity

## [1.1.0] - 2025-09-15

### Added
- **Automatic step size calculation** - Step size now calculated from reactor geometry
- **Species count in filenames** - Mechanism files include species count
- **Enhanced project structure** - Improved documentation and organization

### Changed
- **Mechanism file naming** - Standardized naming convention
- **Documentation** - Enhanced project structure documentation

## [1.0.0] - 2025-09-15

### Added
- **Initial PFR simulation system** - Basic plug flow reactor simulation for propane cracking
- **Cantera integration** - Full integration with Cantera chemical kinetics library
- **Basic visualization** - Temperature, pressure, and species profile plots
- **CSV export** - Data export functionality for analysis
- **Heat flux profiles** - External heating simulation support
- **Pressure drop calculations** - Churchill correlation for friction factor

### Technical Details
- **Core files**: `Main_NaphthaPFR.py`, `config.json`
- **Mechanism**: `Propane_KineticModel.yaml`
- **Dependencies**: Cantera 2.6.0+, Python 3.7+

---

## Version History Summary

| Version | Date | Major Features |
|---------|------|----------------|
| 3.0.0 | 2025-01-14 | Project restructuring, ML Surrogate Models, JSON configuration, centralized aesthetics |
| 2.1.0 | 2025-09-20 | Comprehensive data export, enhanced visualization, export controls |
| 2.0.0 | 2025-09-18 | Multi-reactant support, automatic configuration, professional documentation |
| 1.0.0 | 2024-12-01 | Initial PFR simulation system for propane cracking |

## Future Roadmap

### Completed Features (v3.0.0)
- [x] **Machine Learning** - ML-based surrogate models for fast predictions
- [x] **Parameter Studies** - Automated parameter sweep for training data generation
- [x] **JSON Configuration** - Reproducible configuration system
- [x] **Centralized Styling** - Consistent figure aesthetics

### Planned Features (v3.1.0)
- [ ] **GUI Interface** - Graphical user interface for easier operation
- [ ] **Optimization Tools** - Built-in optimization for reactor design
- [ ] **Advanced Visualization** - Interactive plots and 3D visualizations
- [ ] **Export Formats** - Additional export formats (Excel, HDF5, etc.)
- [ ] **Ensemble Models** - Combine multiple ML models for better accuracy

### Planned Features (v3.2.0)
- [ ] **Additional Reactants** - Support for more feedstocks (butane, pentane, etc.)
- [ ] **Reactor Types** - Support for CSTR and other reactor types
- [ ] **Heat Integration** - Advanced heat integration modeling
- [ ] **Economic Analysis** - Cost estimation and economic optimization
- [ ] **Uncertainty Analysis** - Monte Carlo and sensitivity analysis tools

### Long-term Goals
- [ ] **Cloud Integration** - Cloud-based simulation capabilities
- [ ] **Physics-Informed Neural Networks** - ML models with physics constraints
- [ ] **Real-time Monitoring** - Integration with process control systems
- [ ] **Multi-scale Modeling** - Integration with CFD and other simulation tools

---

**Note**: This changelog follows semantic versioning principles. Breaking changes are indicated by major version increments, new features by minor version increments, and bug fixes by patch version increments.
