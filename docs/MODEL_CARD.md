# HydrAI Surrogate Model Card

## Model Details

- **Model family:** Multi-output tree ensembles: Random Forest, Gradient Boosting, XGBoost, and AdaBoost.
- **Task:** Learn fast surrogate mappings for steam-cracking plug-flow reactor (PFR) simulations generated with Cantera.
- **Primary workflows:**
  - `notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb`: default-parameter baseline comparison for inlet-to-outlet / exit-plane prediction.
  - `notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb`: one selected tree model with hyperparameter tuning for both exit-plane prediction and full axial/PFR evolution.
- **Framework context:** Cantera-based data generation plus scikit-learn/XGBoost surrogate modeling.
- **Intended usage:** Fast screening, design-space exploration, sensitivity studies, and candidate ranking before expensive detailed-chemistry reruns.

## Inputs and Outputs

- **Inputs:** feed identity, inlet operating conditions, reactor geometry/process variables, and, for full-profile models, normalized axial position (`relative_position`).
- **State/thermo/aero outputs:** temperature, pressure, velocity, density, molecular weight, heat capacities, enthalpy, and thermal conductivity.
- **Species outputs:** mass fractions only (`Y_*`) or lumped mass-fraction targets (`Y_lump_*`) exported from Main_3.
- **Pressure convention:** physics data remain in SI units (`pressure_Pa`), while selected plots may display pressure in bar for readability.

## Training Data

- Generated from Cantera PFR simulations over parameter sweeps.
- Supports multiple feedstocks configured through `configs/simulation/reactant_database.json`.
- Sampling strategies include Latin Hypercube, random sampling, and structured grids.
- Main_3 exports ML-ready `df_features` and `df_target` artifacts under `data/processed/`.
- Species targets are mass-fraction based; mole fractions (`X_*`) are not used as ML targets in the current workflow.

## Training and Evaluation Workflows

### Baseline Exit-Plane Evaluation (`Main_4`)

- Trains default RF, Gradient Boosting, XGBoost, and AdaBoost models.
- Uses one sample per simulation run at the reactor exit.
- Does not perform hyperparameter tuning.
- Reports train/test metrics, actual-vs-predicted scatter plots, species-lump diagnostics, and state/thermo/aero errors.

### Tuned Exit + Full Evolution (`Main_5`)

- Tunes one selected model via `MODEL_TO_TUNE`.
- Uses `RandomizedSearchCV` for hyperparameter search.
- Supports exit-plane prediction and full axial/PFR evolution.
- Full-profile mode uses all axial rows and includes `relative_position` as an input.
- Full-profile train/test splitting is done by simulation run to avoid leakage between axial points from the same reactor profile.

## Reported Metrics

- **Global model metrics:** R², MAE, RMSE, MAPE, and train/test R² gap.
- **Species-lump diagnostics:** Normalized MAE (%) by chemistry or carbon-number lump.
- **State/thermo/aero diagnostics:** Normalized MAE (%) by target, including exit temperature, pressure, velocity, density, Cp/Cv, enthalpy, and thermal conductivity.
- **Speed diagnostics:** ML inference latency/throughput is reported in the notebooks. If measured Cantera/PFR runtimes are supplied through `CANTERA_EXIT_SECONDS_PER_RUN` and `CANTERA_FULL_PROFILE_SECONDS_PER_RUN`, the notebooks also report estimated speedup factors.

## Intended Use

- Rapid surrogate inference inside process-screening and optimization loops.
- Early-stage design-space exploration.
- Comparing operating-condition trends and ranking promising cases.
- Full-profile surrogate studies where axial temperature/species evolution is needed quickly.

## Out-of-Scope / Limitations

- Not a substitute for final design validation with high-fidelity Cantera or plant data.
- Accuracy depends on mechanism quality and training-domain coverage.
- Extrapolation outside sampled operating ranges may degrade rapidly.
- Species lumping is heuristic and should be reviewed when changing mechanisms or feedstocks.
- Speedup values should be reported from measured Cantera timings on the same machine/workload rather than assumed.

## Responsible Use Notes

- Verify high-impact decisions with full Cantera simulations.
- Preserve configs, mechanism versions, and data-generation metadata alongside trained artifacts.
- Inspect per-target and lumped-species errors, not only global average metrics.
- Treat low-abundance species/lumps carefully because percentage errors can be unstable near zero.

## Reproducibility Pointers

- Configs: `configs/ml/`, `configs/simulation/`
- Data-generation protocol: `TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md`
- Species lumping methodology: `SPECIES_LUMPING_MODEL_CARD.md`
- Baseline notebook: `notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb`
- Tuning/evolution notebook: `notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb`
- Generated model artifacts: `models/` (normally git-ignored)
