# HydrAI

ML surrogate modeling for chemical reactor simulation. A ten-stage pipeline trains and validates tree ensembles, deep neural networks, a physics-informed neural network (PINN), and symbolic regression against Cantera-generated plug-flow-reactor data, then runs Bayesian optimisation over the resulting surrogate.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Cantera](https://img.shields.io/badge/Cantera-3.2.0%2B-green)](https://cantera.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.4%2B-lightblue?logo=optuna&logoColor=white)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Nikolas Karefyllidis, PhD** — [Website](https://karefyllidis.github.io/) · [GitHub](https://github.com/karefyllidis) · [LinkedIn](https://www.linkedin.com/in/karefyllidis/) · [Google Scholar](https://scholar.google.co.uk/citations?user=kLGU85cAAAAJ&hl=en)

---

## Motivation

Steam-cracking simulation requires stiff ODE integration over detailed reaction mechanisms (10²–10³ species), making broad design-space exploration computationally prohibitive. HydrAI replaces that inner loop with a millisecond-scale ML surrogate — trained and validated against Cantera — enabling real-time inference and gradient-free optimisation without sacrificing physical fidelity.

---

## Pipeline

| Notebook | Purpose |
|---|---|
| `Main_1` | Single PFR run — verify Cantera setup and visualise axial profiles |
| `Main_2` | Parallel training-data generation (LHS + grid sweep, SLURM-ready) |
| `Main_3` | EDA, species lumping, feature engineering, pickle export |
| `Main_4` | Baseline tree surrogates (RF, GBR, XGBoost, AdaBoost) |
| `Main_5` | Hyperparameter tuning (BayesSearchCV) + full axial-profile tree model |
| `Main_6` | Full axial-profile `SimpleNN` with run-level train/test split |
| `Main_7` | Physics-informed neural network (PINN) with PFR residual loss |
| `Main_8` | Symbolic regression (PySR) distillation of any NN teacher → closed-form equations |
| `Main_9` | Cantera vs PINN vs SR comparison/validation |
| `Main_10` | Bayesian optimisation (Optuna GP sampler) via SR surrogate; Cantera validation |

---

## Architecture

### SimpleNN (`src/models/simple_nn.py`)

Three hidden layers, ReLU + dropout, trained with AdamW and `ReduceLROnPlateau`. Optional Optuna TPE search over a held-out validation fold, with best-checkpoint restore on test R² plateau.

### PINNPFR (`src/models/pinn.py`)

Same topology as `SimpleNN`, plus a physics-residual loss derived from the PFR governing equations (`src/physics/pfr_residuals.py`):

```
L = λ_data · MSE(ŷ, y)  +  λ_phys · L_physics
```

Enforced constraints: ideal-gas EOS, mass conservation (ρuA = ṁ), species sum = 1, species ≥ 0, and the energy ODE via autograd on `relative_position`. Curriculum warmup trains on data loss only before the physics term is switched on.

### Symbolic Regression (`Main_8`)

PySR distillation from either NN teacher (`TEACHER_STEM = simple_nn_full_profile` or `pinn_pfr`); the notebook auto-selects model class, sampling strategy, and export directory. Output: closed-form Python equations with no PyTorch dependency, consumed by Main_9 and Main_10.

### Comparison (`Main_9`)

Validates the PINN and its SR distillation against Cantera ground truth on shared axial profiles: overlays, parity plots, a per-target R²/NMAE table, and a PINN-vs-SR inference-speed comparison.

### Bayesian Optimisation (`Main_10`)

Optuna `GPSampler` maximises olefin yield over six inlet degrees of freedom using the SR surrogate, then validates the optimum against a Cantera PFR run and reports surrogate prediction error.

---

## Quick Start

```bash
git clone https://github.com/karefyllidis/open_HydrAI.git
cd open_HydrAI
pip install -r requirements.txt
```

1. Install **Cantera** ([cantera.org](https://cantera.org)).
2. Place mechanism YAML files in `mechanisms/` (referenced by `configs/simulation/main1_reactant_database.json`).
3. Run notebooks in order: `Main_1` → `Main_2` → `Main_3` → … → `Main_10`.

For cluster data generation:

```bash
bash scripts/dev/sbatch_safe.sh scripts/cluster/run_training_mul_GPUs.sh
bash scripts/monitor/monitor_cluster_jobs.sh
python scripts/dev/check_complete_runs.py
python scripts/dev/consolidate_training_data.py
```

Live training monitor (while Main_6 runs):

```bash
python scripts/monitor/monitor_nn_training_progress.py
```

CLI inference (requires trained model exports from Main_6 / Main_7):

```bash
# SimpleNN full-profile prediction
python scripts/predict.py --model nn --mode full_profile \
    --T 850 --P 2.5 --L 12 --D 0.032 --mdot 0.07 --q 180000

# PINNPFR full axial profile → CSV
python scripts/predict.py --model pinn --n-points 200 --output profile.csv

# Tree ensemble batch prediction from JSON
python scripts/predict.py --model tree --json conditions.json --output results.csv
```

---

## Repository Layout

```
open_HydrAI/
├── notebooks/          # Main_1 → Main_10
├── src/
│   ├── cantera/        # PFR simulation wrapper
│   ├── ml/             # data generation, training, inference
│   ├── models/         # SimpleNN, PINNPFR
│   ├── physics/        # PFR residual loss terms
│   └── utils/          # plot style, run logging, parallel coordinates
├── configs/            # simulation/, ml/, style/
├── scripts/            # cluster/, local/, monitor/, dev/
├── data/               # training/, processed/, logs/  (generated; git-ignored)
├── models/             # trained artifacts              (generated; git-ignored)
├── mechanisms/         # kinetic YAML files             (local; git-ignored)
└── docs/               # guides, API reference, structure trees
```

Full tree with file-level descriptions: [docs/STRUCTURE.md](docs/STRUCTURE.md).

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

## Roadmap

- [x] Multi-feed PFR simulation with detailed chemistry (Cantera)
- [x] LHS / grid sampling, SLURM-aware parallel data generation
- [x] Species lumping by carbon number and chemistry role
- [x] Baseline tree surrogates with per-target metrics
- [x] Hyperparameter tuning (BayesSearchCV on trees, Optuna TPE on NNs)
- [x] Full axial-profile PyTorch MLP with run-level split (`Main_6`)
- [x] Physics-informed neural network with PFR residual loss (`Main_7`)
- [x] Symbolic regression distillation from any NN teacher (`Main_8`)
- [x] Cantera vs PINN vs SR comparison/validation (`Main_9`)
- [x] Gaussian-process Bayesian optimisation with Cantera validation (`Main_10`)
- [ ] Profile RNN / LSTM surrogate
- [ ] Bayesian optimisation with safety constraints (SEBO)

---

## References

- Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- Training data protocol: [docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md](docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md)
- Species lumping methodology: [docs/SPECIES_LUMPING_MODEL_CARD.md](docs/SPECIES_LUMPING_MODEL_CARD.md)
- ML configuration keys: [docs/ML_CONFIG_GUIDE.md](docs/ML_CONFIG_GUIDE.md)
- CSD3 HPC setup: [docs/HPC_GUIDE.md](docs/HPC_GUIDE.md)
- Project conventions: [docs/HYDRAI_PROJECT_CONVENTIONS.md](docs/HYDRAI_PROJECT_CONVENTIONS.md)

Parts of the scientific foundation originate from prior doctoral work: [University of Oxford thesis](https://ora.ox.ac.uk/objects/uuid:2479abe8-fefb-4574-b573-a309c278a614).

---

MIT © Nikolas Karefyllidis
