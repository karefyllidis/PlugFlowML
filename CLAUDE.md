# HydrAI — Claude Code Guidelines

## Project overview

ML surrogate models for plug flow reactor (PFR) simulations. Cantera-generated data → tree models → SimpleNN → PINN → symbolic regression → Bayesian optimisation.

## Notebook pipeline

| # | File | Task |
|---|------|------|
| 1 | `Main_1_run_pfr.ipynb` | Single PFR run; Cantera; inline profiles |
| 2 | `Main_2_generate_training_data.ipynb` | Batch sweep; generate training data |
| 3 | `Main_3_data_exploration_feature_engineering.ipynb` | EDA, lumping, export `features_targets_*.pkl` |
| 4 | `Main_4_train_and_evaluate_tree_models_IO.ipynb` | Tree baselines (exit-plane, no tuning) |
| 5 | `Main_5_train_evaluate_tune_tree_model_evolution.ipynb` | Tuned tree + full axial profile |
| 6 | `Main_6_train_evaluate_SimpleNN_IO.ipynb` | PyTorch SimpleNN exit-plane; MC-Dropout UQ |
| 7 | `Main_7_train_evaluate_SimpleNN_full_profile.ipynb` | PyTorch SimpleNN full axial profile |
| 8 | `Main_8_train_evaluate_PINN_full_profile.ipynb` | PINN with PFR ODE residuals (PINNPFR class) |
| 9 | `Main_9_symbolic_regression_SR.ipynb` | PySR distillation of any NN teacher (Main_6/7/8) → symbolic expressions |
| 10 | `Main_10_optimisation_BO_surrogate_vs_cantera.ipynb` | Optuna GP-BO: optimise inlet conditions via MLP + SR surrogates; Cantera validation |

Main_4–Main_10 all consume `data/processed/features_targets_*.pkl` from Main_3.

## Model architecture rules

- **SimpleNN** (`src/models/simple_nn.py`): shared by Main_6 and Main_7. 3 hidden layers + ReLU + Dropout.
- **PINNPFR** (`src/models/pinn.py`): dedicated class for Main_8. Same topology, separate file so PINN changes don't affect data-only models.
- **MC-Dropout UQ** (`predict_with_uncertainty`): only in Main_6 (IO notebook). Do NOT add to Main_7 or Main_8.
- Both exported from `src/models/__init__.py`.

## Data rules

- Use **mass fractions only** (`Y_*`); never mole fractions (`X_*`) in the ML pipeline.
- Species are lumped into chemistry groups: `Y_lump_hydrogen`, `Y_lump_paraffins`, `Y_lump_olefins`, etc.
- Preferred export: `EXPORT_SPECIES_AS = 'lumped_chemistry'` in Main_3.
- Run-level train/test split in Main_5 / Main_7 / Main_8 to prevent profile leakage.
- Do not use `Y_lump_carbon_inert` in prose; the column is a valid pipeline artifact.

## Config structure (`configs/ml/ml_training_config.json`)

- `neural_network.*` — architecture and training (consumed by Main_6, Main_7, Main_8).
- `pinn.loss_weights.*` — λ_data, λ_phys, λ_eos, λ_mass, λ_species_sum, λ_species_nonneg, λ_energy_ode.
- `pinn.training.*` — curriculum_warmup_epochs, n_colloc_per_batch, phys_loss_freq.
- Tree-model blocks (`random_forest`, `gradient_boosting`, etc.) consumed by Main_4 / Main_5.
- `neural_network.*` is ignored by Main_4.

## PINN specifics (Main_8)

- Composite loss: `L = λ_data·MSE + λ_phys·L_physics`
- Curriculum warmup: epochs 0→CURRICULUM_WARMUP are data-loss only; physics added after.
- Physics constraints: EOS (ideal gas), mass conservation `ρuA = ṁ`, species sum = 1, species ≥ 0, energy ODE via `torch.autograd.grad` on `relative_position`.
- Collocation points: inlet conditions from training data + random z/L; no labels required.
- §13 diagnostic: compare trained vs untrained physics residuals along z/L.
- Exports: `models/pinn_pfr_state_dict.pt`, `pinn_pfr_scalers.joblib`, `pinn_pfr_manifest.json`.

## SR specifics (Main_9)

- `TEACHER_STEM` controls which model to distil: `'simple_nn_exit'` (Main_6), `'simple_nn_full_profile'` (Main_7), or `'pinn_pfr'` (Main_8).
- Exit-plane teachers: distillation samples unique inlet conditions from training data.
- Profile teachers (Main_7/8): distillation samples full (run, z) rows — z/L is an SR input.
- Exports land in `models/sr_exit/`, `models/sr_full_profile/`, or `models/sr_pinn/` respectively.
- Each export contains: `*_manifest.json`, `*_equations.py` (callable NumPy functions), `*_metrics.csv`.

## Optimisation specifics (Main_10)

- Optuna `GPSampler` studies run separately for MLP (Main_6) and SR (Main_9) surrogates.
- Search space = 6 inlet conditions bounded to the training domain.
- Objective: maximise `OPT_TARGET` (default `Y_lump_olefins`) at reactor exit.
- Both surrogate optima are validated with a real Cantera PFR; surrogate error is reported.
- Exports figures to `outputs/figures/Main_10_optimisation/`.

## Plotting rules

- `setup_matplotlib()` (`src/utils/plot_style.py`) for global rcParams.
- Preferred line/marker colours: **k** (black), **b** (blue), **r** (red), **m** (magenta), **lime** (guide lines).
- Train curves = **b**, test curves = **r** when both appear on the same axes.
- Bar charts: `facecolor='white'`, `hatch='///'` for second series, visible `edgecolor`.
- Do not use jet/rainbow for scientific quantities. Use `Blues`, `magma`, `coolwarm` for heatmaps.
- No bold text on figures (`axes.titleweight = 'normal'`).

## Notebook section style

Sections use the `# ══ N. TITLE ══` banner pattern. Keep step numbers parallel to Main_4 where sections are analogous (setup → config → features → split → train → eval → export).

## Git / repo

- `nbstripout` is registered as a git filter via `.gitattributes`. Cell outputs are stripped before commit.
- Model artefacts are overwritten each run (stable stems); see `docs/STRUCTURE.md` §10.2.
- Full project conventions: `docs/HYDRAI_PROJECT_CONVENTIONS.md`.
- Architecture detail: `docs/STRUCTURE.md`, `docs/DIRECTORY_STRUCTURE.md`.
