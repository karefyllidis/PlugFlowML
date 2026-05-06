# HydrAI Surrogate Model Card

## Model Details

- **Model family:** Multi-output tree ensembles (Random Forest, Gradient Boosting, XGBoost, AdaBoost)
- **Task:** Predict axial thermo-chemical reactor states for steam-cracking plug-flow reactors
- **Framework context:** Physics-based Cantera data generation + ML surrogates
- **Intended usage:** Fast screening and design-space exploration

## Inputs and Outputs

- **Typical inputs:** Feed identity, inlet operating conditions, geometry/process parameters, and normalized axial position
- **Typical outputs:** Temperature, pressure, thermo properties, and species mass fractions (`Y_*`)

## Training Data

- Generated from high-fidelity Cantera PFR simulations over broad parameter sweeps
- Supports multiple feedstocks (ethane, propane, n-hexane, naphtha)
- Sampling strategies include Latin Hypercube and structured grid

## Reported Performance

- Representative n-hexane runs in this project report mean test **R² ~ 0.97–0.99** across key state and composition targets.

## Intended Use

- Rapid surrogate inference inside optimisation loops
- Early-stage process design studies
- Scenario ranking before expensive full-chemistry reruns

## Out-of-Scope / Limitations

- Not a replacement for first-principles validation in final design decisions
- Performance depends on coverage of the generated training domain
- Extrapolation outside trained operating ranges may degrade rapidly

## Responsible Use Notes

- Always verify high-impact recommendations with full Cantera simulations
- Track data generation settings and mechanism versions for reproducibility

## Reproducibility Pointers

- Configs: `configs/ml/`, `configs/simulation/`
- Pipeline: `notebooks/Main_1*` to `notebooks/Main_4*`, plus `run_pipeline.py`
- Artifacts: `models/` (generated; typically git-ignored)
