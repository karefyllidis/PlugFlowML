# HydrAI Project Conventions and Standards

This document captures all agreed-upon conventions, architectural decisions, and standards for the HydrAI machine learning surrogate modeling project. These rules apply throughout the codebase and should be followed for all future development.

## Table of Contents

- [1. General Principles](#1-general-principles)
- [2. Species Data Handling](#2-species-data-handling)
- [3. ML Pipeline Architecture](#3-ml-pipeline-architecture)
- [4. Notebook Organization](#4-notebook-organization)
- [5. Plotting Standards](#5-plotting-standards)
- [6. Unit Conventions](#6-unit-conventions)
- [7. Performance Metrics](#7-performance-metrics)
- [8. Documentation Standards](#8-documentation-standards)
- [9. Code Style](#9-code-style)
- [10. Data File Management](#10-data-file-management)
- [11. Best Practices Summary](#11-best-practices-summary)
- [12. Terminology Reference](#12-terminology-reference)
- [13. HPC and Parallel Computing](#13-hpc-and-parallel-computing)
- [14. Data Compatibility and Version Management](#14-data-compatibility-and-version-management)
- [15. Matplotlib Style Standards](#15-matplotlib-style-standards)
- [16. Feature Scaling](#16-feature-scaling)
- [17. Pipeline Simplification](#17-pipeline-simplification)
- [18. Portfolio and Professional Standards](#18-portfolio-and-professional-standards)

---

## 1. General Principles

### Project Goal
Build fast, accurate ML surrogate models for plug flow reactor (PFR) simulations that predict both exit conditions (inlet-to-outlet) and full axial evolution profiles.

### Key Design Philosophy
- **Dimensionality reduction through chemistry**: Use lumped species groups rather than tracking hundreds of individual species
- **Mass fractions only**: Exclusively use mass fractions (Y_*), not mole fractions (X_*)
- **Separation of concerns**: Distinct notebooks for different tasks (baseline evaluation vs. hyperparameter tuning)
- **Comprehensive documentation**: Every architectural decision must be documented in model cards

---

## 2. Species Data Handling

### 2.1 Species Fractions

**RULE**: Use **mass fractions only** (Y_*) throughout the ML pipeline. Do NOT use mole fractions (X_*).

**Rationale**: Mass fractions provide sufficient information for predictions and reduce feature dimensionality.

**Note**: The PFR simulator (`src/cantera/pfr_simulator.py`) may compute and plot mole fractions for chemistry validation purposes. This is acceptable — the rule applies to the ML pipeline inputs, targets, and outputs exclusively.

### 2.2 Species Categorization

Species are lumped into chemistry-based groups to reduce ML input dimensions:

#### Chemistry-Based Groups (Primary)
- `hydrogen` - H2 (explicitly its own category, separate from diluent)
- `diluent` - N2, Ar, He (NOT including H2)
- `paraffins` - Saturated hydrocarbons (CnH(2n+2))
- `olefins` - Unsaturated hydrocarbons with C=C bonds
- `diolefins` - Hydrocarbons with two C=C bonds
- `aromatics` - Benzene rings and aromatic compounds
- `acetylenes` - Hydrocarbons with C≡C triple bonds
- `oxygenates` - O-containing species (CO, CO2, H2O, etc.)
- `coke_precursors` - Heavy aromatics, PAHs
- `inert` - Non-carbon species (note: do NOT use "carbon_inert" in prose/human communication; however, the data column name `Y_lump_carbon_inert` is a valid pipeline artifact from the carbon-content lumping scheme)
- `other` - Unclassified species

#### Carbon-Content Groups (Secondary)
- Based on number of carbon atoms: C0, C1, C2, C3, C4, C5+

### 2.3 Lumped Species Export

**Configuration**: Use `EXPORT_SPECIES_AS` flag in Main_3

Options:
- `'individual'`: Export all Y_species_name columns
- `'lumped_carbon'`: Export Y_lump_carbon_C0, Y_lump_carbon_C1, etc.
- `'lumped_chemistry'`: Export Y_lump_hydrogen, Y_lump_paraffins, etc.

**Recommendation**: Use `'lumped_chemistry'` for most ML applications to reduce data file size and improve training efficiency.

### 2.4 Species Naming

Species classification logic (`_classify_species_chemistry` function):
- Paraffins: CnH(2n+2) pattern
- Olefins: CnH2n pattern (exclude benzene C6H6)
- Diolefins: CnH(2n-2) pattern (exclude acetylene C2H2)
- Acetylenes: C2H2 or CnH(2n-4) pattern
- Aromatics: Contains benzene ring patterns (c6h6, benzene, toluene, etc.)
- H2: Explicitly classified as 'hydrogen'
- CO, CO2, H2O: Classified as 'oxygenates'

---

## 3. ML Pipeline Architecture

### 3.1 Notebook Roles

| Notebook | Purpose | Key Features |
|----------|---------|-------------|
| `Main_1` | Case generation | Creates parametric sweeps for PFR simulations |
| `Main_2` | Training data generation | Runs Cantera simulations in parallel |
| `Main_3` | EDA & feature engineering | Species lumping, feature selection, data export |
| `Main_4_train_and_evaluate_tree_models_IO` | **Baseline evaluation** | Compares multiple tree models **without hyperparameter tuning**, exit-plane predictions only |
| `Main_5_train_evaluate_tune_tree_model_evolution` | **Tuning & evolution** | Hyperparameter tuning for one model, both exit-plane and full PFR axial evolution |

### 3.2 Main_2 (Training Data Generation) Standards

**Purpose**: Generate high-quality, physically consistent training data from Cantera PFR runs for downstream ML.

**Scope**:
- Run parameter sweeps over feed and operating conditions.
- Support local parallel execution and SLURM/HPC chunked execution.
- Write canonical outputs to pickle files only (`training_data_complete_*.pkl` + metadata JSON).

**Main Rules**:
- Use `Main_2_generate_training_data.ipynb` as the canonical generation workflow.
- Keep raw generated training files immutable once produced; do not rewrite them in alternative formats.
- Prefer chunked parallel generation for large campaigns (`scripts/cluster/` and `scripts/local/` helpers).
- For HPC, use one worker per task (`--cpus-per-task=1`) and scale with `--ntasks`.
- Keep `Main_2` output schema stable because `Main_3` feature engineering depends on it.

**Post-run Required Steps**:
1. Monitor and validate run completion.
2. Consolidate per-task artifacts with `scripts/dev/consolidate_training_data.py`.
3. Continue with `Main_3_data_exploration_feature_engineering.ipynb`.

**Data Contract to Main_3**:
- `Main_3` may read raw species columns from `Main_2`.
- Final ML targets exported by `Main_3` must remain mass-fraction-based (`Y_*`) and optionally lumped (`Y_lump_*`).
- Mole fractions (`X_*`) are not used as ML targets downstream.

### 3.3 Main_4 (Baseline Evaluation) Standards

**Purpose**: Quickly evaluate multiple tree-based models with default hyperparameters to establish a baseline.

**Key Characteristics**:
- NO hyperparameter tuning (no RandomizedSearchCV)
- Inlet-to-outlet (exit-plane) predictions only
- Compares: RandomForest, GradientBoosting, AdaBoost, XGBoost
- Exports best model to `outputs/models/`
- Figure directory: `outputs/figures/Main_4_train_and_evaluate_tree_models_IO/`

**Required Plots**:
1. Model comparison table (R², MAE, RMSE, Median AE, Max Error, training time)
2. Best model information block (data counts, architecture, hyperparameters)
3. Actual vs. predicted scatter plots for key state variables (temperature, pressure, velocity, density)
4. Normalized MAE bar chart for **chemistry groups** (using lumped species)
5. Normalized MAE bar chart for **state/thermo/aero targets** (temperature, pressure, velocity, density, residence time)
6. Speed comparison report (ML inference time vs. Cantera simulation time)

**Figure Naming**: No `exit_` prefix needed (all plots are exit-plane by default)

### 3.4 Main_5 (Tuning & Evolution) Standards

**Purpose**: Hyperparameter tune ONE selected model and train for both exit-plane and full axial evolution predictions.

**Key Characteristics**:
- Hyperparameter tuning via RandomizedSearchCV
- Trains TWO models: one for exit-plane, one for full PFR axial profile
- Full-profile training uses run-level train/test split to avoid data leakage
- Optional subsampling for full-profile (`FULL_PROFILE_MAX_ROWS`)
- Figure directory: `outputs/figures/Main_5_train_evaluate_tune_tree_model_evolution/`

**Configuration Flags**:
- `MODEL_TO_TUNE`: Select model to tune ('RandomForest', 'GradientBoosting', etc.)
- `TRAIN_FULL_PROFILE`: Default True
- `FULL_PROFILE_MAX_ROWS`: Optional row limit for large datasets
- `IF_HYPERPARAM_TUNING`: Must be True
- `TUNING_N_ITER`, `TUNING_CV`, `TUNING_SCORING`

**Required Plots** (Exit-plane):
1. Same as Main_4, but with `exit_` prefix on all figure filenames
   - `exit_actual_vs_predicted_state_scatter.png`
   - `exit_chemistry_group_nmae.png`
   - `exit_state_thermo_target_nmae.png`

**Required Plots** (Full-profile):
2. Full-profile metrics (R², MAE, RMSE by target)
3. Normalized MAE for state/thermo targets across full axial profile

**Exported Artifact**: `.joblib` file containing:
- `exit_model`: Tuned model for exit-plane predictions
- `full_profile`: Dictionary with model, scaler, metrics, etc.

### 3.5 Speed Comparison Reporting

**Configuration Variables**:
- `CANTERA_EXIT_SECONDS_PER_RUN`: Reference time for Cantera exit-plane simulation
- `CANTERA_FULL_PROFILE_SECONDS_PER_RUN`: Reference time for Cantera full-profile simulation

**Report Contents**:
- ML inference time for test set
- Average time per prediction
- Speedup factor vs. Cantera (if reference times provided)
- Include in exported artifact metadata

---

## 4. Notebook Organization

### 4.1 Cell Structure

1. **Title and summary** (markdown cell at top)
2. **Imports cell** (all imports at beginning)
   - Standard library
   - Third-party (numpy, pandas, matplotlib, sklearn, etc.)
   - Local modules (`src.ml.*`)
3. **Configuration flags** (centralized at top)
4. **Main workflow cells**
5. **Export/save results cell** (at end)

### 4.2 Import Standards

**RULE**: All imports must be in the first code cell(s) of the notebook.

**Example Order**:
```python
# Standard library
import re
import time
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Local modules
from src.ml.dataframe_pickle import load_dataframe_pickle
```

Do NOT scatter imports throughout the notebook (e.g., `import re` in middle cells).

### 4.3 Configuration Flags

Place all configuration flags at the top of notebooks after imports:

```python
# ============================================================================
# Configuration
# ============================================================================
IF_HYPERPARAM_TUNING = True
MODEL_TO_TUNE = 'RandomForest'
TRAIN_FULL_PROFILE = True
FULL_PROFILE_MAX_ROWS = 500_000

# Export controls
EXPORT_SPECIES_AS = 'lumped_chemistry'  # 'individual', 'lumped_carbon', 'lumped_chemistry'

# Speed comparison (optional reference times)
CANTERA_EXIT_SECONDS_PER_RUN = None  # Set to measured value if known
CANTERA_FULL_PROFILE_SECONDS_PER_RUN = None
```

---

## 5. Plotting Standards

### 5.1 Scatter Plot Aesthetics

**RULE**: All actual-vs-predicted scatter plots must use consistent aesthetics:

```python
ax.scatter(actual, predicted, alpha=0.25, s=10, edgecolors='none', c='b')
lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
ax.plot(lims, lims, 'r-', lw=2)  # Perfect prediction line
```

Parameters:
- `alpha=0.25`: Semi-transparent to show density
- `s=10`: Small marker size
- `edgecolors='none'`: No marker edges
- `c='b'`: Blue color
- Perfect prediction line: red solid line (`'r-'`), linewidth 2

### 5.2 Figure Export

**RULE**: Always create output directory before saving:

```python
FIG_DIR = Path('outputs/figures/Main_4_train_and_evaluate_tree_models_IO')
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(FIG_DIR / 'actual_vs_predicted_state_scatter.png', dpi=120, bbox_inches='tight')
```

**Naming Conventions**:
- Main_4 figures: No prefix (all are exit-plane)
- Main_5 exit figures: `exit_` prefix (e.g., `exit_chemistry_group_nmae.png`)
- Main_5 full-profile figures: `full_` prefix (e.g., `full_state_evolution.png`)

### 5.3 Chemistry Group Plots

**RULE**: Plot normalized MAE (NOT R²) for chemistry groups

**Y-axis label**: `'Normalized MAE (%)  [MAE / mean(y_true) × 100]'`

**Horizontal reference lines**:
```python
ax.axhline(y=1.0, color='green', linestyle='--', lw=1.5, alpha=0.7, label='1% NMAE (excellent)')
ax.axhline(y=5.0, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='5% NMAE (good)')
ax.axhline(y=10.0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='10% NMAE (acceptable)')
```

### 5.4 State/Thermo/Aero Target Plots

**RULE**: Plot normalized MAE for key physical quantities:
- Temperature (K)
- Pressure (Pa, convert to bar for display)
- Velocity (m/s)
- Density (kg/m³)
- Residence time (s)

**Figure name**: `state_thermo_target_nmae.png` (Main_4) or `exit_state_thermo_target_nmae.png` (Main_5)

### 5.5 Input Parameter Distributions

**RULE**: Plot as probability density histograms

```python
ax.hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')
ax.set_ylabel('Probability density')
ax.set_title('Input parameter probability distributions')
```

Do NOT use frequency counts; use `density=True` for normalized distributions.

### 5.6 Grid Layouts

**RULE**: Use fixed grid layouts for multi-panel figures

**Example** (2×2 grid for key chemistry groups):
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, group in enumerate(['hydrogen', 'paraffins', 'olefins', 'aromatics']):
    ax = axes[i]
    # Plot on ax
```

Do NOT generate hundreds of subplot panels (causes `tight_layout()` errors).

### 5.7 Tick Label Rotation

**RULE**: X-axis tick label rotation must always be `0` (horizontal).

Use:
```python
ax.tick_params(axis='x', rotation=0)
```

Do NOT use angled rotations (e.g., `rotation=35`) in project plots.

### 5.8 Allowed Marker Shapes

**RULE**: Only two marker shapes are permitted across all plots:

| Marker | Code | Typical use |
|--------|------|-------------|
| Circle | `'o'` | Regular data points, scatter clouds |
| Square | `'s'` | Highlighted / special points (e.g. optimal trial, best model) |

```python
# ✅ GOOD
ax.scatter(x, y, marker='o', ...)           # standard scatter
ax.scatter([x_best], [y_best], marker='s', color='#d62728', ...)  # highlight

# ❌ BAD — not permitted
ax.scatter(..., marker='*', ...)
ax.scatter(..., marker='^', ...)
ax.scatter(..., marker='D', ...)
ax.plot(..., marker='x', ...)
```

Do NOT use stars (`*`), triangles (`^`, `v`), diamonds (`D`), crosses (`x`), or any other marker shape.

### 5.9 Text Weight on Figures

**RULE**: Never use **bold** text on any figure element. All text — titles, axis labels, axis names, tick labels, legend entries, annotations, suptitles, colorbar labels — must be rendered with `fontweight='normal'` (the matplotlib default).

**Rationale**:
- Visual consistency across all plots in the project.
- Bold titles look heavy/unprofessional next to thin axis labels and ticks.
- Keeps the focus on the data (lines, markers, colors), not on typographic emphasis.
- Bold is reserved for prose/markdown, where it carries semantic meaning.

**Apply to**:
- `ax.set_title(...)`, `fig.suptitle(...)`, `plt.suptitle(...)`
- `ax.set_xlabel(...)`, `ax.set_ylabel(...)`
- `ax.text(...)`, `ax.annotate(...)`
- Colorbar labels (`cbar.set_label(...)`)
- Legend titles
- Any custom axis-name labels (e.g. column headers in parallel-coordinates / parallel-sets plots)

**Enforcement**:
- `setup_matplotlib()` in `src/utils/plot_style.py` locks `axes.titleweight`, `axes.labelweight`, and `figure.titleweight` to `'normal'`.
- `configs/style/figure_aesthetics.json` `font.title_weight = "normal"`.
- Do not override these in individual plot helpers.

**Bad**:
```python
ax.set_title('Exit conversion', fontweight='bold')                       # forbidden
fig.suptitle('Training space', fontweight='bold', y=1.02)                # forbidden
ax.text(i, 1.045, label, fontweight='bold')                              # forbidden
plt.rcParams['axes.titleweight'] = 'bold'                                # forbidden
```

**Good**:
```python
ax.set_title('Exit conversion')                                          # normal weight (default)
fig.suptitle('Training space', y=1.02)
ax.text(i, 1.045, label)
```

If a title genuinely needs more visual weight, use a larger `fontsize` instead of bold.

---

## 6. Unit Conventions

### 6.1 Pressure Units

**RULE**: All pressure values displayed in plots MUST be in **bar**.

**Internal Storage**: Data may be in Pascal (Pa)

**Display Conversion**: Always convert Pa → bar for plots:

```python
pressure_bar = pressure_Pa / 1e5

# Plot labels
ax.set_xlabel('Pressure (bar)')
ax.set_ylabel('Predicted Pressure (bar)')
ax.set_title('Exit Pressure: Actual vs. Predicted (bar)')
```

**Apply to**:
- Scatter plot axes labels
- Bar chart x-tick labels
- Figure titles
- Axis limits

### 6.2 Other Units

Standard SI units unless otherwise noted:
- Temperature: Kelvin (K)
- Velocity: meters per second (m/s)
- Density: kilograms per cubic meter (kg/m³)
- Residence time: seconds (s)
- Mass fractions: dimensionless (0-1)

---

## 7. Performance Metrics

### 7.1 Primary Metrics

For model comparison tables:

1. **R² (R-squared)**: Coefficient of determination
2. **MAE (Mean Absolute Error)**: Average absolute error
3. **RMSE (Root Mean Squared Error)**: Root mean squared error
4. **Median AE**: Median absolute error (robust to outliers)
5. **Max Error**: Maximum absolute error
6. **Training Time**: Model fitting duration (seconds)

### 7.2 Normalized MAE (NMAE)

**Definition**:
```python
NMAE (%) = (MAE / mean(y_true)) × 100
```

**Use Cases**:
- Chemistry group performance comparison
- State/thermo/aero target comparison
- Full-profile axial evolution diagnostics

**Thresholds** (for bar chart reference lines):
- **< 1%**: Excellent
- **< 5%**: Good
- **< 10%**: Acceptable
- **> 10%**: Needs improvement

### 7.3 Hyperparameter Tuning Scoring

**Default**: `'r2'` (maximize R²)

**Alternatives** (add as comments in notebooks):
```python
# Alternative scoring metrics:
# - 'neg_mean_absolute_error'       (minimize MAE)
# - 'neg_mean_squared_error'        (minimize MSE)
# - 'neg_root_mean_squared_error'   (minimize RMSE)
# - 'neg_median_absolute_error'     (minimize median AE, robust to outliers)
```

---

## 8. Documentation Standards

### 8.1 Documentation Structure

**Location**: All documentation (except README.md) must be in `docs/` folder

**Files**:
- `README.md` - Root project overview (stays in root)
- `docs/ML_CONFIG_GUIDE.md` - Detailed ML configuration reference
- `docs/MODEL_CARD.md` - High-level surrogate model card
- `docs/SPECIES_LUMPING_MODEL_CARD.md` - Species categorization methodology
- `docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md` - Data generation workflow
- `docs/STRUCTURE.md` - Repository structure overview
- `docs/DIRECTORY_STRUCTURE.md` - Detailed file listing
- `docs/CHANGELOG.md` - Version history

### 8.2 Model Cards

**RULE**: Every major architectural decision or methodology must have a model card

**Model Card Template**:
1. **Purpose**: Why this approach?
2. **Methodology**: How does it work?
3. **Configuration**: What flags control behavior?
4. **Outputs**: What does it produce?
5. **Limitations**: What are the constraints?
6. **Downstream Impact**: How do other notebooks use this?

### 8.3 Cross-References

**RULE**: Use relative paths for internal documentation links

**Example**:
```markdown
See [ML Configuration Guide](ML_CONFIG_GUIDE.md) for details.
See [Species Lumping Model Card](SPECIES_LUMPING_MODEL_CARD.md) for methodology.
```

Do NOT use absolute paths or external URLs for internal docs.

### 8.4 README Updates

**RULE**: Update README.md whenever:
- New notebook is added or renamed
- Workflow steps change
- Configuration flags are added
- Model card structure changes

**Key Sections to Maintain**:
- Get Started (ordered list of notebooks)
- ML workflow diagram/description
- Figure export controls
- Model cards section (with links)

---

## 9. Code Style

### 9.1 Comments

**RULE**: Do NOT add obvious, narration-style comments

**Bad Examples** (avoid these):
```python
# Import the module
import numpy as np

# Define the function
def calculate_mae(y_true, y_pred):
    # Return the result
    return mean_absolute_error(y_true, y_pred)

# Increment the counter
counter += 1
```

**Good Examples** (use these):
```python
# Use run-level split to avoid data leakage (rows from same run must stay together)
train_runs, test_runs = train_test_split(run_ids, test_size=0.2, random_state=42)

# Convert Pa to bar for display (internal data is in Pa)
pressure_bar = pressure_Pa / 1e5

# Subsample to manage memory (full dataset may exceed RAM)
if FULL_PROFILE_MAX_ROWS and len(df) > FULL_PROFILE_MAX_ROWS:
    df = df.sample(n=FULL_PROFILE_MAX_ROWS, random_state=42)
```

**Guidelines**:
- Explain WHY, not WHAT
- Document non-obvious intent
- Explain trade-offs and constraints
- Clarify business logic or domain knowledge

### 9.2 Variable Naming

**Conventions**:
- `df_*`: DataFrames (e.g., `df_data`, `df_features`, `df_target`)
- `X_*`, `y_*`: ML features and targets (e.g., `X_train`, `y_test`)
- `*_lump_*`: Lumped/aggregated species (e.g., `Y_lump_hydrogen`)
- `*_normalized`: Normalized/scaled values
- `*_Pa`: Pressure in Pascal
- `*_bar`: Pressure in bar

### 9.3 Pandas Optimization

**RULE**: Subset DataFrame columns BEFORE expensive operations

**Bad** (slow):
```python
for _, g in df_data.groupby(run_cols):
    # Process group...
```

**Good** (fast):
```python
subset_cols = run_cols + ['z_position_m', 'velocity_ms']
for _, g in df_data[subset_cols].groupby(run_cols):
    # Process group...
```

**Rationale**: Reduces memory usage and speeds up groupby operations on wide DataFrames.

---

## 10. Data File Management

### 10.1 Pickle Files

**RULE**: Use `.pkl` files for all training data storage

**Functions**:
- `src.ml.dataframe_pickle.save_dataframe_pickle(df, path)`
- `src.ml.dataframe_pickle.load_dataframe_pickle(path)`

**DO NOT**:
- Use Parquet sidecars (explicitly removed per user request)
- Modify raw data files (they are immutable)
- Create temporary format conversions

### 10.2 File Naming

**Training Data**:
- `training_data_task_{task_id}.pkl` (individual task)
- `training_data_consolidated.pkl` (merged dataset)

**ML Models**:
- `best_tree_model_exit_IO.joblib` (Main_4 output)
- `tuned_{model_name}_exit_and_evolution.joblib` (Main_5 output)

### 10.3 Output Organization

```
outputs/
├── figures/
│   ├── Main_3_data_exploration_feature_engineering/
│   ├── Main_4_train_and_evaluate_tree_models_IO/
│   └── Main_5_train_evaluate_tune_tree_model_evolution/
├── reports/
│   ├── README.md
│   └── Main_N_*.md  (per-notebook summary reports)
└── results/
    └── consolidated_training_data/

models/                                # Root-level directory (git-ignored)
├── best_tree_model_exit_IO.joblib
└── tuned_RandomForest_exit_and_evolution.joblib
```

### 10.4 Gitignore

**Tracked**:
- Notebooks (`.ipynb`)
- Scripts (`.py`, `.sh`)
- Documentation (`.md`)
- Configuration files
- `.gitkeep` placeholders

**Ignored** (in `.gitignore`):
- `.cursor/` directory (including rules)
- `*.pkl`, `*.joblib`, `*.h5` (ML artifacts)
- `outputs/figures/**` (generated plots)
- `data/training/**`, `data/processed/**` (generated data)
- `__pycache__/`, `.ipynb_checkpoints/`

---

## 11. Best Practices Summary

### Critical Do's

✅ Use mass fractions (Y_*) only, never mole fractions (X_*)  
✅ Convert pressure to bar for all plots  
✅ Use normalized MAE for chemistry group and state variable diagnostics  
✅ Centralize all imports at notebook top  
✅ Create output directories before saving figures  
✅ Use consistent scatter plot aesthetics (alpha=0.25, s=10, c='b')  
✅ Document all architectural decisions in model cards  
✅ Update documentation when workflow changes  
✅ Use run-level train/test split for full-profile training  
✅ Include speed comparison reports in both Main_4 and Main_5  

### Critical Don'ts

❌ Don't use mole fractions (X_*)  
❌ Don't use "carbon_inert" terminology (just "inert")  
❌ Don't show pressure in Pa on plots (always convert to bar)  
❌ Don't create hundreds of subplot panels (causes errors)  
❌ Don't add obvious narration comments  
❌ Don't scatter imports throughout notebooks  
❌ Don't use R² for chemistry group diagnostics (use NMAE)  
❌ Don't modify raw data files (they're immutable)  
❌ Don't implement Parquet sidecars (explicitly removed)  
❌ Don't forget to update README when notebooks change  
❌ Don't use marker shapes other than `'o'` (circle) and `'s'` (square) — stars, triangles, diamonds, crosses are all forbidden  
❌ Don't use **bold** text on any figure element (titles, labels, ticks, legends, annotations, colorbars) — `fontweight` must always be `'normal'`. Use a larger `fontsize` if more emphasis is needed.  

---

## 12. Terminology Reference

### Correct Terms

- **Exit-plane predictions**: Inlet-to-outlet predictions (reactor exit conditions)
- **Full-profile predictions**: Full axial/PFR evolution predictions
- **Lumped species**: Aggregated species by chemistry or carbon content
- **Chemistry groups**: hydrogen, paraffins, olefins, aromatics, etc.
- **Normalized MAE (NMAE)**: (MAE / mean(y_true)) × 100
- **State/thermo/aero targets**: Temperature, pressure, velocity, density, residence time
- **Run-level split**: Train/test split that keeps all rows from same run together

### Deprecated Terms (Do NOT Use)

- ~~Mole fractions~~ → Use mass fractions
- ~~"carbon_inert"~~ → Use "inert"
- ~~"Feature importance"~~ → Removed from Main_4
- ~~"Parquet sidecars"~~ → Removed, use .pkl only
- ~~"Exit temperature (Pa)"~~ → Temperature is in K, pressure is in bar

---

## 13. HPC and Parallel Computing

### 13.1 SLURM Scripts

**Target Scale**: Scripts must support 100+ CPU processors for parallel training data generation

**Key Scripts**:
- `scripts/hpc/run_main2_parallel.sh`: Parallelized Main_2 execution
- `scripts/hpc/run_main2_slurm_chunk.py`: Python launcher for SLURM chunks

**Best Practices**:
- Use array jobs for parametric sweeps
- Monitor chunk progress via logs
- Document expected wall-time and resource requirements
- Test on small scale before full production runs

### 13.2 Cross-Platform Compatibility

**RULE**: All scripts must work on **both Windows and Linux**

**Implications**:
- Use `pathlib.Path` for all file paths (not `os.path`)
- Test shell scripts on both PowerShell and Bash
- Use `/` separators in documentation (Path handles conversion)
- Avoid platform-specific commands in Python code

**Example**:
```python
# Good (cross-platform)
from pathlib import Path
data_dir = Path('data/training')
file_path = data_dir / 'training_data.pkl'

# Bad (platform-specific)
import os
data_dir = 'data\\training'  # Windows-only
file_path = os.path.join(data_dir, 'training_data.pkl')  # Less robust
```

---

## 14. Data Compatibility and Version Management

### 14.1 Pickle Compatibility

**Problem**: Pandas pickles are fragile across minor versions (especially StringDtype)

**Solution**: Use `src.ml.dataframe_pickle` module for portable saving/loading

**Implementation**:
```python
# In src/ml/dataframe_pickle.py
def save_dataframe_pickle(df: pd.DataFrame, path: Path | str) -> None:
    # Convert fragile dtypes (StringDtype) to object before saving
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['string']).columns:
        df_copy[col] = df_copy[col].astype('object')
    
    with open(path, 'wb') as f:
        pickle.dump(df_copy, f, protocol=4)

def load_dataframe_pickle(path: Path | str) -> pd.DataFrame:
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except TypeError as e:
        raise RuntimeError(
            f"Could not read {path}: the pickle was likely saved with a different "
            f"pandas version (extension dtypes like StringDtype are fragile across "
            f"minor releases). Fix: (1) `pip install -U 'pandas>=2.2'` or match the "
            f"pandas version used when the file was created; or (2) re-run Main_2 / "
            f"Main_3 to regenerate pickles (new saves use more portable dtypes)."
        ) from e
```

**When Errors Occur**:
1. Update pandas: `pip install -U 'pandas>=2.2'`
2. OR regenerate pickles by re-running Main_2 → Main_3

### 14.2 Requirements Management

**File**: `requirements.txt`

**Update When**:
- New Python packages are used
- Pandas/NumPy/scikit-learn versions change
- Compatibility issues arise

**Key Dependencies**:
```
pandas>=2.2
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
cantera>=3.0
joblib>=1.3
```

---

## 15. Matplotlib Style Standards

### 15.1 Global Style Setup

**RULE**: Use a consistent `setup_matplotlib()` function for all notebooks

**Implementation** (in `src/utils/plot_style.py` or notebook setup cell):

```python
def setup_matplotlib(ax=None):
    """
    Configure matplotlib with HydrAI project aesthetics.
    
    Call at start of notebook to set global style.
    Optionally pass axes to apply per-axis styling.
    """
    # Global style
    plt.rcParams.update({
        # Figure
        "figure.figsize": (24, 6),
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        
        # Fonts
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        
        # Axes
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Colors
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "xtick.minor.pad": 4,
        "ytick.minor.pad": 4,
        
        # Legend
        "legend.frameon": False,
        
        # Fonts export (for publications)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        
        # TeX (disabled for compatibility)
        "text.usetex": False
    })
    
    # Optional per-axis styling
    if ax is not None:
        axes_list = list(ax.flat) if hasattr(ax, "flat") else [ax]
        for a in axes_list:
            a.set_axisbelow(True)
            a.grid(False, linestyle="--", linewidth=0.65, color="gray", alpha=0.75)
            a.tick_params(which="both", direction="in", top=False, bottom=True, 
                         left=True, right=False)
            a.minorticks_on()
```

**Usage in Notebooks**:
```python
# At start of plotting section
setup_matplotlib()

# For specific figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
setup_matplotlib(ax=axes)
```

### 15.2 Style Rationale

**Design Choices**:
- **No top/right spines**: Cleaner, modern look
- **Inward ticks**: Professional scientific style
- **Minor ticks on**: Better scale reading
- **No legend frame**: Reduces visual clutter
- **PDF fonttype 42**: Ensures editable text in PDFs (required for publications)
- **No LaTeX**: Avoids dependencies and cross-platform issues

---

## 16. Feature Scaling

### 16.1 When to Scale

**Question**: Should we scale features with different physical dimensions (pressure, mass fractions, temperature)?

**Answer**: **Yes, always scale features for ML models**

**Rationale**:
- Tree-based models (RF, GBM, XGBoost) are **scale-invariant** but still benefit from scaling for:
  - Better convergence in AdaBoost
  - Consistent feature importance interpretation
  - Future compatibility with other model types (neural networks, linear models)
- Non-tree models **require** scaling

### 16.2 Scaling Implementation

**RULE**: Use `StandardScaler` from scikit-learn

**Pattern** (already implemented in Main_4 and Main_5):
```python
from sklearn.preprocessing import StandardScaler

# Fit scaler on training data only
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Train model on scaled data
model.fit(X_train_scaled, y_train)

# Save both model and scaler for deployment
joblib.dump({'model': model, 'scaler': scaler_X}, 'model_artifact.joblib')
```

**Critical**: Never fit scaler on test data (causes data leakage)

---

## 17. Pipeline Simplification

### 17.1 run_pipeline.py

**RULE**: Main execution script must be **< 10 lines** and extremely simple

**Goal**: Make it trivially easy for users to run the entire pipeline

**Implementation**:
```python
#!/usr/bin/env python
"""Run the complete HydrAI ML pipeline."""

from pathlib import Path
import subprocess
import sys

notebooks = ['Main_1', 'Main_2', 'Main_3', 'Main_4_train_and_evaluate_tree_models_IO']
for nb in notebooks:
    subprocess.run([sys.executable, '-m', 'jupyter', 'execute', f'notebooks/{nb}.ipynb'], check=True)
```

**No loops for advanced features**: Keep it as simple as possible for portfolio/demo purposes

---

## 18. Portfolio and Professional Standards

### 18.1 Project Purpose

**Target Audience**: ML/Data Science recruiters and hiring managers

**Key Messages**:
- Combines domain expertise (hydrocarbon cracking chemistry) with modern ML techniques
- Demonstrates end-to-end pipeline: data generation → EDA → modeling → evaluation
- Shows software engineering best practices (documentation, version control, reproducibility)

### 18.2 Documentation for CV/Portfolio

**README.md Requirements**:
- Clear project overview (what, why, how)
- Quick start instructions (< 5 steps to get running)
- Visual results (include key plots)
- Technical stack (Python, pandas, scikit-learn, Cantera)
- Model performance summary (R², speedup vs. physics simulation)
- Link to detailed documentation in `docs/`

**Professional Tone**:
- Use clear, concise language
- Explain domain concepts for non-expert readers
- Show results and business value (e.g., "1000× speedup vs. Cantera")
- Include limitations and future work sections

### 18.3 Code Quality for Portfolio

**Standards**:
- Clean, readable code (no commented-out blocks)
- Meaningful variable names
- Docstrings for all functions
- Type hints where appropriate
- Consistent formatting (use black or similar)
- No hardcoded paths (use Path and configuration files)

**What to Hide in .gitignore**:
- Temporary/scratch files
- Large data files (provide download instructions instead)
- Environment-specific configs
- `.cursor/` directory (development artifacts)

---

## Version History

- **v1.0** (2026-05-08): Initial rule file creation
  - Captured conventions from agent chat [41b8b498]
  - Includes species lumping, plotting standards, notebook organization
  - Documents Main_4/Main_5 split and NMAE diagnostics

- **v1.1** (2026-05-08): Added conventions from agent chat [6eb23bc7]
  - HPC/SLURM parallel computing standards
  - Cross-platform compatibility (Windows/Linux)
  - Pickle compatibility and pandas version management
  - Matplotlib style standards (setup_matplotlib function)
  - Feature scaling rationale and implementation
  - Pipeline simplification (run_pipeline.py < 10 lines)
  - Portfolio and professional documentation standards

- **v1.2** (2026-05-08): Marker shape constraint from agent chat [7c5baaf3]
  - Section 5.8: only `'o'` and `'s'` markers are allowed in all plots

- **v1.3** (2026-05-12): No-bold-text-on-figures rule from agent chat [Parallel-axes EDA polish](a08436a7-e157-42ac-a2ad-af3ecc2cf96d)
  - Added Section 5.9: bold text is forbidden on every figure element (`fontweight='normal'` everywhere).
  - `setup_matplotlib()` now locks `axes.titleweight`, `axes.labelweight`, `figure.titleweight` to `'normal'`.
  - `configs/style/figure_aesthetics.json`: `font.title_weight` changed from `"bold"` to `"normal"`.
  - Stripped `fontweight='bold'` from `src/utils/plot_parallel.py`, `src/cantera/pfr_simulator.py` defaults, and notebooks Main_3 / Main_4 / Main_5.

---

## Questions or Clarifications?

If any part of these conventions is unclear or seems to conflict with project goals, consult:
1. Model cards in `docs/` folder
2. Agent transcript: [41b8b498]
3. This rule file (most up-to-date source of truth)

When in doubt, follow the patterns established in `Main_4_train_and_evaluate_tree_models_IO.ipynb` and `Main_5_train_evaluate_tune_tree_model_evolution.ipynb` as reference implementations.
