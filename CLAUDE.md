# HydrAI — Claude Code Guidelines

## Project overview

ML surrogate models for plug flow reactor (PFR) simulations. Cantera-generated data → tree models → SimpleNN → PINN → symbolic regression → Bayesian optimisation.

## Notebook pipeline

| # | File | Task |
|---|------|------|
| 1 | `Main_1_run_pfr.ipynb` | Single PFR run; Cantera; inline profiles |
| 2 | `Main_2_generate_training_data.ipynb` | Batch sweep; generate training data |
| 3 | `Main_3_data_exploration_feature_engineering.ipynb` | EDA, lumping, export `features_targets_*.pkl` |
| 4 | `Main_4_train_and_evaluate_tree_models_IO.ipynb` | Tree baselines (exit-plane); optional BayesSearchCV tuning (§7, off by default) |
| 5 | `Main_5_train_evaluate_tune_tree_model_evolution.ipynb` | Tuned tree + full axial profile |
| 6 | `Main_6_train_evaluate_SimpleNN_full_profile.ipynb` | PyTorch SimpleNN full axial profile |
| 7 | `Main_7_train_evaluate_PINN_full_profile.ipynb` | PINN with PFR ODE residuals (PINNPFR class) |
| 8 | `Main_8_symbolic_regression_SR.ipynb` | PySR distillation of any NN teacher (Main_6/7) → symbolic expressions |
| 9 | `Main_9_compare_cantera_pinn_sr.ipynb` | Cantera vs PINN vs SR comparison/validation |
| 10 | `Main_10_optimisation_BO_surrogate_vs_cantera.ipynb` | Optuna GP-BO: optimise inlet conditions via SR surrogate; Cantera validation |

Main_4–Main_10 all consume `data/processed/features_targets_*.pkl` from Main_3.

## Model architecture rules

- **SimpleNN** (`src/models/simple_nn.py`): shared by Main_6. 3 hidden layers + ReLU + Dropout.
- **PINNPFR** (`src/models/pinn.py`): dedicated class for Main_7. Same topology, separate file so PINN changes don't affect data-only models.
- **MC-Dropout UQ** (`predict_with_uncertainty`): not exercised by any notebook; available via `scripts/predict.py --model nn --mode full_profile --mc-samples N` for the Main_6 model.
- Both exported from `src/models/__init__.py`.

## Data rules

- Use **mass fractions only** (`Y_*`); never mole fractions (`X_*`) in the ML pipeline.
- Species are lumped into chemistry groups: `Y_lump_hydrogen`, `Y_lump_paraffins`, `Y_lump_olefins`, etc.
- Preferred export: `EXPORT_SPECIES_AS = 'lumped_chemistry'` in Main_3.
- Run-level train/test split in Main_5 / Main_6 / Main_7 to prevent profile leakage.
- Do not use `Y_lump_carbon_inert` in prose; the column is a valid pipeline artifact.

## Config structure (`configs/ml/`)

Each notebook that reads a JSON config owns a dedicated, descriptively-named file — no config file is shared across two Main notebooks:

| File | Notebook | Contents |
|------|----------|----------|
| `main1_run_pfr_config.json` | Main_1 | `reactant_key` |
| `main2_data_generation_config.json` | Main_2 | reactants, sampling method, parameter ranges |
| `main3_eda_feature_engineering_config.json` | Main_3 | EDA/export flags, species lumping, run-stamp pinning |
| `main4_tree_baseline_config.json` | Main_4 | `test_size`, `random_state`, `models_to_train`, tree-model blocks (`random_forest`, `xgboost`, `gradient_boosting`, `adaboost`), `tuning.*` (BayesSearchCV budget, §7 only) |
| `main5_tree_tuning_config.json` | Main_5 | Same shape as Main_4's file plus `model_to_tune`, `full_profile_max_rows`; independent copy so tuning Main_5 never changes Main_4's baseline |
| `main6_simplenn_config.json` | Main_6 | `test_size`, `random_state`, `runtime.*` (CPU/Optuna-job/subsample-row counts, read early in §2), `neural_network.*` (architecture + training + `tuning.*`) |
| `main7_pinn_config.json` | Main_7 | `test_size`, `random_state`, `full_profile_max_rows`, `neural_network.*` (self-contained, same architecture as Main_6, + `tuning.*` for the optional §6b data-loss proxy search), `pinn.loss_weights.*`, `pinn.training.*` |
| `main8_symbolic_regression_config.json` | Main_8 | PySR budget, `teacher_stem`, distillation sample count |
| `main9_compare_cantera_pinn_sr_config.json` | Main_9 | `sr_teacher_stem`, `n_comparison_runs` |
| `main10_bayesian_optimisation_config.json` | Main_10 | `opt_target`, Optuna `n_trials`, inlet condition `bounds` |
| `model_training_script_config.json` | `src/ml/model_training.py` (standalone legacy CLI, not part of the Main_N pipeline) | `data_file`, `output_dir`, `target_types`, `models` |

- **Convention**: notebook flag cells (`# N. PATHS & FLAGS`) hold only boolean (`IF_*`) literals and `Path` objects. Every number, string, or list a user might tune (model lists, tuning budgets, row caps, CPU/Optuna job counts, teacher/model selectors) is config-driven via `config.get(key, default)`, even where that means an early, narrowly-scoped config read ahead of the notebook's main "Load Config" cell (e.g. Main_6's `runtime.*` block, needed before thread pools are configured). This keeps notebooks safe to re-run without accidentally hand-editing a number in a code cell.
- `pinn.loss_weights.*` — λ_data, λ_phys, λ_eos, λ_mass, λ_species_sum, λ_species_nonneg, λ_energy_ode.
- `pinn.training.*` — curriculum_warmup_epochs, n_colloc_per_batch, phys_loss_freq.
- `configs/simulation/main1_*.json` (PFR run template, reactant database, heat flux profile) are shared by Main_1 and Main_2's underlying `src/cantera/pfr_simulator.py` calls — not split per-notebook since both genuinely depend on the same simulation domain files.
- `configs/style/figure_aesthetics.json` is shared globally across all notebooks (see Plotting rules).

## PINN specifics (Main_7)

- Composite loss: `L = λ_data·MSE + λ_phys·L_physics`
- Curriculum warmup: epochs 0→CURRICULUM_WARMUP are data-loss only; physics added after.
- Physics constraints: EOS (ideal gas), mass conservation `ρuA = ṁ`, species sum = 1, species ≥ 0, energy ODE via `torch.autograd.grad` on `relative_position`.
- Collocation points: inlet conditions from training data + random z/L; no labels required.
- §13 diagnostic: compare trained vs untrained physics residuals along z/L.
- Exports: `models/pinn_pfr_state_dict.pt`, `pinn_pfr_scalers.joblib`, `pinn_pfr_manifest.json`.

## SR specifics (Main_8)

- `TEACHER_STEM` controls which model to distil: `'simple_nn_full_profile'` (Main_6) or `'pinn_pfr'` (Main_7).
- Both teachers are profile models: distillation samples full (run, z) rows — z/L is an SR input.
- Exports land in `models/sr_full_profile/` or `models/sr_pinn/` respectively.
- Each export contains: `*_manifest.json`, `*_equations.py` (callable NumPy functions), `*_metrics.csv`.

## Comparison specifics (Main_9)

- Loads the Main_7 PINN directly (state_dict + scalers) and the Main_8 SR equations distilled from it (`sr_teacher_stem` default `'pinn_pfr'`, i.e. `models/sr_pinn/`).
- Ground truth comes from the processed `features_targets_*.pkl` dataset (Main_3) — no fresh Cantera runs.
- Produces axial-profile overlays (Cantera vs PINN vs SR), parity plots, a per-target R²/NMAE table, and a PINN-vs-SR inference-speed comparison.
- Exports `comparison_metrics.csv` and `comparison_manifest.json` under `outputs/figures/Main_9_compare_cantera_pinn_sr/`.

## Optimisation specifics (Main_10)

- Optuna `GPSampler` study runs on the SR (Main_8) surrogate.
- Search space = 6 inlet conditions bounded to the training domain.
- Objective: maximise `OPT_TARGET` (default `Y_lump_olefins`) at reactor exit.
- The surrogate optimum is validated with a real Cantera PFR; surrogate error is reported.
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
