# HydrAI

Machine-learning surrogate models for plug-flow reactor (PFR) steam-cracking simulation.
Cantera-generated training data flows through tree ensembles, PyTorch MLPs, a
physics-informed neural network, symbolic regression, and Bayesian optimisation —
all in a reproducible ten-notebook pipeline.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Cantera](https://img.shields.io/badge/Cantera-3.2%2B-green)](https://cantera.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Nikolas Karefyllidis, PhD** · [GitHub](https://github.com/karefyllidis) · [LinkedIn](https://www.linkedin.com/in/karefyllidis/) · [Google Scholar](https://scholar.google.co.uk/citations?user=kLGU85cAAAAJ&hl=en) · [Oxford thesis](https://ora.ox.ac.uk/objects/uuid:2479abe8-fefb-4574-b573-a309c278a614)

---

## Motivation

High-fidelity PFR simulation with detailed chemistry (150+ species, 2000+ reactions) takes
minutes per run. HydrAI replaces the Cantera solver with surrogates that predict the full
axial state profile — temperature, pressure, velocity, and 9 lumped species yields — in
milliseconds, enabling real-time design and optimisation workflows.

---

## Pipeline

| Step | Notebook | Purpose |
|:---:|---|---|
| 1 | `Main_1_run_pfr` | Single Cantera PFR run; inline axial profiles |
| 2 | `Main_2_generate_training_data` | Latin-hypercube sweep; batch PFR generation |
| 3 | `Main_3_data_exploration_feature_engineering` | Velocity QC · run ID · residence time · EDA · species lumping · `df_rates` |
| 4 | `Main_4_train_and_evaluate_tree_models_IO` | Tree baselines — RF, GB, XGB, AdaBoost (exit-plane) |
| 5 | `Main_5_train_evaluate_tune_tree_model_evolution` | Tuned tree + full axial profile reconstruction |
| 6 | `Main_6_train_evaluate_SimpleNN_IO` | PyTorch `SimpleNN` exit-plane; Optuna; MC-Dropout UQ |
| 7 | `Main_7_train_evaluate_SimpleNN_full_profile` | PyTorch `SimpleNN` full axial profile; run-level split |
| 8 | `Main_8_train_evaluate_PINN_full_profile` | `PINNPFR` — PFR ODE residuals; curriculum warmup; collocation |
| 9 | `Main_9_symbolic_regression_SR` | PySR distillation of any NN teacher → closed-form equations |
| 10 | `Main_10_optimisation_BO_surrogate_vs_cantera` | Optuna GP-BO: optimise inlet conditions via MLP + SR; Cantera validation |

Main_4 – Main_10 read from `data/processed/features_targets_*.pkl` produced by Main_3.

---

## Quick Start

```bash
git clone https://github.com/karefyllidis/HydrAI.git
cd HydrAI
pip install -r requirements.txt
```

1. Install [Cantera](https://cantera.org) into the same environment.
2. Place mechanism YAML files in `mechanisms/` (see `configs/simulation/reactant_database.json`).
3. Run notebooks in order: `Main_1` → `Main_2` → `Main_3` → … → `Main_10`.
4. All hyperparameters live in `configs/ml/ml_training_config.json`.

**CLI inference** (requires Main_6 / Main_7 / Main_8 exports):

```bash
# SimpleNN exit-plane prediction
python scripts/predict.py --model nn \
    --T 850 --P 2.5 --L 12 --D 0.032 --mdot 0.07 --q 180000

# PINNPFR full axial profile → CSV
python scripts/predict.py --model pinn --n-points 200 --output profile.csv

# Tree ensemble batch prediction from JSON
python scripts/predict.py --model tree --json conditions.json --output results.csv
```

---

## Repository Layout

```
HydrAI/
├── notebooks/          Main_1 – Main_10
├── src/
│   ├── cantera/        PFR simulator (Cantera wrapper)
│   ├── ml/             data generation, tree training, inference
│   ├── models/         SimpleNN, PINNPFR
│   ├── physics/        PFR ODE residuals (PINN physics loss)
│   └── utils/          plot_style, training_progress_log
├── configs/
│   ├── ml/             ml_training_config.json, ml_data_generation_config.json
│   ├── simulation/     reactant_database.json, heat_flux_profile.json
│   └── style/          plot aesthetics
├── scripts/            cluster/, local/, monitor/, dev/
├── data/               training/, processed/, logs/        (git-ignored)
├── models/             trained artefacts + SR equations    (git-ignored)
├── mechanisms/         YAML kinetic files                  (git-ignored)
└── docs/               guides, config reference, conventions
```

---

## Data Card

| Field | Value |
|---|---|
| Feedstock | n-hexane (primary); ethane, propane, naphtha supported |
| Mechanism | 153 species, 2 146 reactions |
| Simulation | Cantera 3.2 PFR, wall heat flux |
| Inlet T / P | 800 – 900 K / 1.5 – 3.5 bar |
| Reactor L / D | 10 – 15 m / 25 – 40 mm |
| Mass flow / Heat flux | 0.05 – 0.10 kg/s / 100 – 250 kW/m² |
| Sampling | Latin Hypercube (6 parameters) |
| Axial resolution | 200 steps per run |
| Targets | 9 state/thermo variables + 9 lumped species mass fractions |
| Train / test split | 80 / 20 run-level (no axial leakage) |
| Velocity QC | Runs with u ≤ 0 or u above 99.5th-percentile removed (Main_3 §2.1b) |

---

## Model Architecture

### SimpleNN (Main_6 / Main_7)

Three-hidden-layer MLP with ReLU activations and Dropout (`src/models/simple_nn.py`).
Main_6 uses the exit-plane variant (6 inlet inputs → 18 outputs) with Optuna
hyperparameter tuning and MC-Dropout uncertainty quantification. Main_7 extends to the
full axial profile by adding z/L as an input.

### PINNPFR (Main_8)

Dedicated physics-informed class (`src/models/pinn.py`), same MLP topology as SimpleNN
but decoupled so PINN-specific changes never affect data-only surrogates.

Composite loss:

```
L = λ_data · MSE(ŷ, y)  +  λ_phys · L_physics
```

Physics constraints enforced: EOS (ideal gas), mass conservation (ρuA = ṁ), species
sum = 1, species ≥ 0, energy ODE via `torch.autograd.grad` on `relative_position`.
Curriculum warmup trains on data loss only for the first `CURRICULUM_WARMUP_EPOCHS`,
then switches on physics. Unlabelled collocation points sample random z/L per
mini-batch. All λ weights in `configs/ml/ml_training_config.json → pinn.loss_weights`.

---

## Symbolic Regression (Main_9)

Main_9 distils any trained NN teacher into closed-form NumPy-callable expressions via
[PySR](https://github.com/MilesCranmer/PySR). Set `TEACHER_STEM` to choose the source:

| `TEACHER_STEM` | Source | Inputs |
|---|---|---|
| `simple_nn_exit` | Main_6 | 6 inlet conditions |
| `simple_nn_full_profile` | Main_7 | 6 inlet conditions + z/L |
| `pinn_pfr` | Main_8 | 6 inlet conditions + z/L |

Equations export to `models/sr_<teacher>/sr_<teacher>_equations.py` — inference requires
no PyTorch runtime. Operators: `+, -, *, /, exp, sqrt, square`.

---

## Bayesian Optimisation (Main_10)

Main_10 uses Optuna `GPSampler` to find inlet conditions that maximise a target yield
(default: `Y_lump_olefins`) within the training domain. Two independent studies run
in sequence — one with the MLP (Main_6) as the objective, one with the SR expressions
(Main_9). Both optima are then validated by a real Cantera PFR simulation, and the
surrogate prediction error is reported alongside a parameter-space comparison plot.

---

## Roadmap

- [x] Cantera PFR simulation (multi-feed, detailed chemistry)
- [x] Latin-hypercube sampling; cluster-aware parallel data generation
- [x] Velocity QC, run ID, residence time, reaction rate proxies (`df_rates`)
- [x] Tree surrogates with baseline comparison and hyperparameter tuning
- [x] Species lumping (chemistry-role and carbon-number groupings)
- [x] `SimpleNN` exit-plane — Optuna, early stopping, MC-Dropout UQ (Main_6)
- [x] `SimpleNN` full axial profile — run-level split, Optuna (Main_7)
- [x] `PINNPFR` — composite loss, curriculum warmup, autograd physics (Main_8)
- [x] Symbolic regression distillation via PySR; multi-teacher support (Main_9)
- [x] Bayesian optimisation: Optuna GP-BO with Cantera validation (Main_10)
- [ ] Profile RNN / LSTM surrogate
- [ ] Bayesian optimisation with safety constraints (SEBO)

---

## Further Reading

- [docs/STRUCTURE.md](docs/STRUCTURE.md) — full repository layout
- [docs/ML_CONFIG_GUIDE.md](docs/ML_CONFIG_GUIDE.md) — config key reference
- [docs/HYDRAI_PROJECT_CONVENTIONS.md](docs/HYDRAI_PROJECT_CONVENTIONS.md) — coding and plotting standards
- [CLAUDE.md](CLAUDE.md) — Claude Code project guidelines

---

[MIT](LICENSE) © Nikolas Karefyllidis
