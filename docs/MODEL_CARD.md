# HydrAI Surrogate Model Card

## Model Details

- **Model family:**
  - Tree ensembles: Random Forest, Gradient Boosting, XGBoost, and AdaBoost (Main_4/Main_5).
  - `SimpleNN` — 3-hidden-layer PyTorch MLP, full axial profile (Main_6).
  - `PINNPFR` — physics-informed variant of the same MLP topology, with PFR ODE/algebraic residuals in the loss (Main_7).
  - Symbolic regression (PySR) — closed-form equations distilled from a trained `SimpleNN` or `PINNPFR` teacher (Main_8).
- **Task:** Learn fast surrogate mappings for steam-cracking plug-flow reactor (PFR) simulations generated with Cantera.
- **Primary workflows:**
  - `notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb`: default-parameter baseline comparison for inlet-to-outlet / exit-plane prediction.
  - `notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb`: one selected tree model for exit-plane and full axial/PFR evolution, with optional hyperparameter tuning.
  - `notebooks/Main_6_train_evaluate_SimpleNN_full_profile.ipynb`: PyTorch MLP over the full axial profile, run-level train/test split, optional Optuna architecture search.
  - `notebooks/Main_7_train_evaluate_PINN_full_profile.ipynb`: `PINNPFR` trained with a composite data + physics loss over the full axial profile.
  - `notebooks/Main_8_symbolic_regression_SR.ipynb`: PySR distillation of the Main_6 or Main_7 teacher into per-target closed-form equations.
- **Framework context:** Cantera-based data generation; scikit-learn/XGBoost for tree ensembles; PyTorch for `SimpleNN`/`PINNPFR`; PySR for symbolic distillation.
- **Intended usage:** Fast screening, design-space exploration, sensitivity studies, and candidate ranking before expensive detailed-chemistry reruns. Symbolic equations (Main_8) are additionally intended for embedding in downstream optimization (Main_10) or external tools where a NumPy/PyTorch runtime is unavailable.

## Inputs and Outputs

- **Inputs:** feed identity, inlet operating conditions, reactor geometry/process variables, and, for full-profile models, normalized axial position (`relative_position`).
- **State/thermo/aero outputs:** temperature, pressure, velocity, density, molecular weight, heat capacities, enthalpy, and thermal conductivity.
- **Species outputs:** mass fractions only (`Y_*`) or lumped mass-fraction targets (`Y_lump_*`) exported from Main_3.
- **Pressure convention:** physics data remain in SI units (`pressure_Pa`), while selected plots may display pressure in bar for readability.

## Training Data

- Generated from Cantera PFR simulations over parameter sweeps.
- Supports multiple feedstocks configured through `configs/simulation/main1_reactant_database.json`.
- Sampling strategies include Latin Hypercube, random sampling, and structured grids.
- Main_3 exports ML-ready `df_features` and `df_target` artifacts under `data/processed/`.
- Species targets are mass-fraction based; mole fractions (`X_*`) are not used as ML targets in the current workflow.
- Main_6 and Main_7 train directly on `features_targets_*.pkl` rows (one row per `(run, z)` pair for full-profile models).
- Main_8's distillation dataset is **not** fresh Cantera data: it is the trained NN teacher evaluated on `N_DISTILL_SAMPLES` inlet points resampled from the same processed dataset. SR equations therefore inherit any teacher error and are one additional approximation step removed from Cantera ground truth (validated directly against Cantera exit-plane rows in Main_8 §8 and against ground truth again in Main_9).

## Training and Evaluation Workflows

### Baseline Exit-Plane Evaluation (`Main_4`)

- Trains default RF, Gradient Boosting, XGBoost, and AdaBoost models.
- Uses one sample per simulation run at the reactor exit.
- Does not perform hyperparameter tuning.
- Reports train/test metrics, actual-vs-predicted scatter plots, species-lump diagnostics, and state/thermo/aero errors.

### Tuned Exit + Full Evolution (`Main_5`)

- Tunes one selected model via `MODEL_TO_TUNE`.
- Uses `BayesSearchCV` when tuning is enabled (`IF_HYPERPARAM_TUNING=True`); otherwise trains with default parameters.
- Full axial/PFR training reuses the exit-plane hyperparameters (no second tuning run).
- Supports exit-plane prediction and full axial/PFR evolution.
- Full-profile mode uses all axial rows and includes `relative_position` as an input.
- Full-profile train/test splitting is done by simulation run to avoid leakage between axial points from the same reactor profile.
- Includes explicit Cantera-vs-ML axial overlays and condition-binned error diagnostics to identify operating regimes where the surrogate performs well or struggles.

### Full-Profile PyTorch MLP (`Main_6`)

- `SimpleNN` (`src/models/simple_nn.py`): 3 hidden layers (`h1`/`h2`/`h3`, default 128/64/32) + ReLU + Dropout (default `p=0.1`), linear output head.
- Inputs: inlet/geometry/process columns (`initial_temperature_K`, `initial_pressure_Pa`, `reactor_length_m`, `reactor_diameter_m`, `mass_flow_rate_kgps`, `heat_flux_Wm2`, `reactant_type` if present) plus `z_position_m` and `relative_position`.
- Outputs: state/thermo/aero targets (temperature, pressure, velocity, density, MW, Cp/Cv, enthalpy, thermal conductivity) plus lumped species (`Y_lump_*`, or raw `Y_*` if lumping wasn't exported).
- Run-level train/test split (`train_test_split` over unique `run_id` groups, not rows) — no axial-profile leakage.
- Row cap for smoke/dev runs (notebook-only `SUBSAMPLE_ROWS` flag gates `runtime.subsample_max_rows`): samples **whole `run_id` groups** up to the row budget, not individual rows, so a capped run is always a complete axial profile rather than a partial one. *(Fixed 2026-07-02 — the prior implementation called `DataFrame.sample()` on individual post-split rows, which scattered `(run, z)` points across many runs and left most kept runs with gaps in their axial profile; see `docs/CHANGELOG.md`.)*
- `StandardScaler` fit on training rows only for both `X` and `y`; test rows are `transform`-only.
- Optional Optuna hyperparameter search (Section 6b, `IF_HYPERPARAM_TUNING=True`):
  - Sampler: TPE with a median pruner; budget from `neural_network.tuning.*` (`n_trials`, `epochs_per_trial`, `validation_fraction`, `timeout_seconds`).
  - Search space: `h1 ∈ [32,256]`, `h2 ∈ [16,128]`, `h3 ∈ [8,64]` (step-quantized ints), `dropout ∈ [0.0,0.3]`, `learning_rate ∈ [1e-4,1e-2]` (log-scale), `batch_size ∈ {64,128,256,512}`.
  - Objective: validation R² (uniform average, physical units) on a fold carved from **train** rows only (`validation_fraction`) — held-out test runs are never touched by tuning.
  - Best trial's hyperparameters overwrite the notebook-level `H1/H2/H3/DROPOUT/LEARNING_RATE/BATCH_SIZE`; the production model is rebuilt and training/evaluation/export proceed unchanged on the tuned net. Recorded in the export manifest's `tuning.*` block (`enabled`, `n_trials_completed`, `best_val_r2`, `best_params`).
- Training uses `ReduceLROnPlateau` keyed on periodic held-out test R², runs the full configured `epochs` (no early stopping), and restores the best test-R² checkpoint before evaluation/export.
- Overfitting controls: dropout, run-level held-out test set, train-only scaler fit, shuffled minibatches. Deliberately not used: weight decay, k-fold CV, data augmentation.
- MC-Dropout uncertainty (`predict_with_uncertainty`) is available on this architecture but not exercised inside the notebook; see `scripts/predict.py --model nn --mode full_profile --mc-samples N`.

### Physics-Informed Neural Network (`Main_7`)

- `PINNPFR` (`src/models/pinn.py`): same MLP topology as `SimpleNN`, kept as a dedicated class so PINN-specific changes never affect the data-only Main_6 model.
- Same run-level split, scaling, and inputs/outputs convention as Main_6 (feature/target column keys match so Main_8/Main_9 can load either teacher through one code path).
- Row cap for smoke/dev runs (top-level `full_profile_max_rows` config key, applied in Section 3 before the run-level split): keeps **whole, randomly-selected `run_id` groups** budgeted by average rows/run. *(Fixed 2026-07-02 — the prior implementation sliced `df.iloc[:N]`, taking rows in raw file order; this truncated whichever run straddled row N mid-profile and produced a non-random, order-biased subset of the design space rather than a representative sample. See `docs/CHANGELOG.md`.)*
- Optional Optuna hyperparameter search (Section 6b, `IF_HYPERPARAM_TUNING=True`, off by default): reuses Main_6's architecture/optimizer search space and TPE + median-pruner setup, but each trial trains on **data loss only** (no physics/collocation terms) for `epochs_per_trial` epochs — a proxy search over architecture/optimizer hyperparameters, not over `pinn.loss_weights.*`. Production training (Section 10) always uses `H1/H2/H3/DROPOUT/LEARNING_RATE` from `main7_pinn_config.json` regardless of this flag; physics loss weights are set directly in the config and are not tuned by Section 6b.
- Composite loss `L = λ_data·MSE + λ_phys·L_physics`; curriculum warmup (`curriculum_warmup_epochs`, default 20) trains on data loss only before physics terms are switched on.
- Physics constraints (default weights in `configs/ml/main7_pinn_config.json → pinn.loss_weights`):
  - Algebraic: ideal-gas EOS, mass conservation (`ρuA = ṁ`), species mass fractions sum to 1, species ≥ 0.
  - ODE (via `torch.autograd.grad` on `relative_position`): energy balance `dT/d(z/L) = LπDq/(ṁ·cp)`.
  - Optional (`use_cantera_residuals=true`, off by default): full `dY_k/dz` and `dT/dz` from Cantera production rates — requires raw, unlumped species (`src/physics/pfr_residuals.py`).
- Collocation: `n_colloc_per_batch` random unlabeled `z/L` points per mini-batch enforce physics at axial positions not present in the training rows.
- §13 diagnostic compares trained-vs-untrained physics residual magnitude along `z/L` — this is the primary signal that the model has actually learned to respect the governing equations, distinct from (and complementary to) test-set R².
- §11b compares test R² directly against the Main_6 data-only model on the same split.

### Symbolic Regression Distillation (`Main_8`)

- Distills a trained `SimpleNN` (`teacher_stem='simple_nn_full_profile'`) or `PINNPFR` (`teacher_stem='pinn_pfr'`) teacher into one closed-form PySR equation per output target.
- Distillation set: teacher evaluated on `n_distill_samples` inlet points (default 5,000) resampled from the processed dataset; for profile teachers this includes `z/L` as an input, so the equation set covers the full axial profile, not just the exit plane.
- PySR search: `niterations`/`populations`/`population_size`/`maxsize` from `configs/ml/main8_symbolic_regression_config.json`; operators restricted to `+ - * /` (binary) and `exp, sqrt, square` (unary) by default.
- Evaluated two ways: parity against the teacher NN on the distillation set (§6/§7), and directly against Cantera exit-plane ground truth (§8) — the latter is the more meaningful accuracy number since it isn't circular through the teacher.
- One PySR fit covers all targets; each output gets an independently-evolved expression, reported with its own complexity and training loss.

## Reported Metrics

- **Global model metrics:** R², MAE, RMSE, MAPE, and train/test R² gap.
- **Species-lump diagnostics:** Normalized MAE (%) by chemistry or carbon-number lump.
- **State/thermo/aero diagnostics:** Normalized MAE (%) by target, including exit temperature, pressure, velocity, density, Cp/Cv, enthalpy, and thermal conductivity.
- **Speed diagnostics:** ML inference latency/throughput is reported in the notebooks. If measured Cantera/PFR runtimes are supplied through `CANTERA_EXIT_SECONDS_PER_RUN` and `CANTERA_FULL_PROFILE_SECONDS_PER_RUN`, the notebooks also report estimated speedup factors.
- **Main_6/Main_7 (NN/PINN) additions:** uniform-average R² reported separately for the full target set, state/thermo columns only, and species/lump columns only (train and test), plus test MAE/RMSE and the train−test R² overfit gap. Main_7 additionally reports test R² against Main_6 on the same split, and the trained-vs-untrained physics residual magnitude along `z/L` (§13) as a physics-consistency diagnostic — not an accuracy metric.
- **Main_8 (SR) additions:** per-target R²/MAE against the teacher NN on the distillation set, per-target R²/MAE against Cantera exit-plane ground truth (when available — the more meaningful of the two), and per-equation complexity + PySR training loss.

## Intended Use

- Rapid surrogate inference inside process-screening and optimization loops.
- Early-stage design-space exploration.
- Comparing operating-condition trends and ranking promising cases.
- Full-profile surrogate studies where axial temperature/species evolution is needed quickly.
- Settings where physical consistency along the axial profile matters more than raw fit quality favor the PINN (Main_7) over the data-only NN (Main_6).
- Settings needing a portable, dependency-free (no PyTorch) predictor — e.g., embedding in Main_10's Optuna surrogate loop or in external/CFD code — favor the SR equations (Main_8).

## Out-of-Scope / Limitations

- Not a substitute for final design validation with high-fidelity Cantera or plant data.
- Accuracy depends on mechanism quality and training-domain coverage.
- Extrapolation outside sampled operating ranges may degrade rapidly.
- Species lumping is heuristic and should be reviewed when changing mechanisms or feedstocks.
- Speedup values should be reported from measured Cantera timings on the same machine/workload rather than assumed.
- PINN physics constraints are soft penalties in the loss, not hard constraints — a low physics-loss weight or short curriculum can leave the trained model still violating conservation laws by a nontrivial margin; always check the §13 residual diagnostic rather than assuming the physics loss guarantees compliance.
- The `use_cantera_residuals` PINN mode requires raw (unlumped) species; it is not compatible with `EXPORT_SPECIES_AS = 'lumped_chemistry'` datasets.
- SR equations are distilled from a teacher NN, not fit directly to Cantera — they add an extra approximation layer and cannot be more accurate than their teacher on the distillation set. Treat the teacher-parity metric as a distillation-fidelity check, and the Cantera-exit-plane metric (Main_8 §8) as the actual accuracy claim.
- SR equation form is limited by the configured operator set (default `+ - * /`, `exp`, `sqrt`, `square`); targets with sharp thresholds, discontinuities, or wide dynamic range (e.g., trace species) may need more iterations, more distillation samples, or additional operators (e.g. `log`) before the fit is trustworthy.

## Responsible Use Notes

- Verify high-impact decisions with full Cantera simulations.
- Preserve configs, mechanism versions, and data-generation metadata alongside trained artifacts.
- Inspect per-target and lumped-species errors, not only global average metrics.
- Treat low-abundance species/lumps carefully because percentage errors can be unstable near zero.
- For the PINN, review the §13 physics-residual diagnostic before trusting a model for physically-sensitive use; a good test-set R² does not by itself confirm conservation-law compliance.
- For SR, re-check equations against Cantera (Main_8 §8, and again in Main_9) whenever the teacher NN is retrained — the equations are not automatically refreshed and can silently go stale relative to a newer teacher.
- If a Main_6/Main_7 artifact was trained with row-capped data (`SUBSAMPLE_ROWS=True` or `full_profile_max_rows` set) **before 2026-07-02**, re-train it: the row cap used per-row sampling (Main_6) or a raw positional slice (Main_7) instead of whole `run_id` profiles, so capped runs could have gaps in or truncated axial profiles rather than the complete pipe profile the model card now assumes.

## Reproducibility Pointers

- Configs: `configs/ml/`, `configs/simulation/`
- Data-generation protocol: `TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md`
- Species lumping methodology: `SPECIES_LUMPING_MODEL_CARD.md`
- Baseline notebook: `notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb`
- Tuning/evolution notebook: `notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb`
- SimpleNN notebook/config: `notebooks/Main_6_train_evaluate_SimpleNN_full_profile.ipynb`, `configs/ml/main6_simplenn_config.json`
- PINN notebook/config: `notebooks/Main_7_train_evaluate_PINN_full_profile.ipynb`, `configs/ml/main7_pinn_config.json`
- SR notebook/config: `notebooks/Main_8_symbolic_regression_SR.ipynb`, `configs/ml/main8_symbolic_regression_config.json`
- Generated model artifacts: `models/` (normally git-ignored; each subfolder below is explicitly un-ignored via `.gitignore` so it stays present, empty, in a fresh clone)
  - Tree baseline (Main_4): `models/tree_baseline/tree_models_exit.joblib`
  - Tree tuned (Main_5): `models/tree_tuned/tree_model_tuned_exit_full.joblib`
  - SimpleNN (Main_6): `models/simple_nn_full_profile/simple_nn_full_profile_{state_dict.pt,scalers.joblib,manifest.json,per_target_metrics.csv,group_metrics.csv}`
  - PINN (Main_7): `models/pinn_pfr/pinn_pfr_{state_dict.pt,scalers.joblib,manifest.json,per_target_metrics.csv}`
  - SR (Main_8): `models/sr_full_profile/` (SimpleNN teacher) or `models/sr_pinn/` (PINN teacher) — each with `*_manifest.json`, `*_equations.py`, `*_metrics.csv`
