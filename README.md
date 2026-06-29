# HydrAI

Machine-learning surrogates for steam-cracking plug-flow reactors. Cantera-based PFR simulations provide the reference dataset; a ten-notebook pipeline trains and evaluates tree ensembles, deep neural networks, a physics-informed network, and symbolic regression models — then uses Gaussian-process Bayesian optimisation to find optimal inlet conditions and validates the result against full Cantera simulations.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Cantera](https://img.shields.io/badge/Cantera-3.2.0%2B-green)](https://cantera.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Nikolas Karefyllidis, PhD** — [GitHub](https://github.com/karefyllidis) · [LinkedIn](https://www.linkedin.com/in/karefyllidis/) · [Google Scholar](https://scholar.google.co.uk/citations?user=kLGU85cAAAAJ&hl=en)

---

## Motivation

High-fidelity steam-cracking simulation requires stiff ODE integration coupled to detailed reaction mechanisms (10²–10³ species), making broad operating-space exploration prohibitively slow. HydrAI replaces the inner loop with millisecond-scale surrogate inference while keeping a fully validated Cantera reference for dataset generation and result verification.

---

## Pipeline

| Notebook | Purpose |
|---|---|
| `Main_1` | Single PFR run — verify Cantera setup and visualise axial profiles |
| `Main_2` | Parallel training-data generation (LHS + grid sweep, SLURM-ready) |
| `Main_3` | EDA, species lumping, feature engineering, pickle export |
| `Main_4` | Baseline tree surrogates (RF, GBR, XGBoost, AdaBoost) |
| `Main_5` | Hyperparameter tuning (BayesSearchCV) + full axial-profile tree model |
| `Main_6` | Exit-plane PyTorch MLP (`SimpleNN`) with Optuna TPE tuning |
| `Main_7` | Full axial-profile `SimpleNN` with run-level train/test split |
| `Main_8` | Physics-informed neural network (PINN) with PFR residual loss |
| `Main_9` | Symbolic regression (PySR) distillation of any NN teacher → closed-form equations |
| `Main_10` | Bayesian optimisation (Optuna GP sampler) via MLP and SR surrogates; Cantera validation |

---

## Architecture

### SimpleNN (`src/models/simple_nn.py`)

Three hidden layers with ReLU activations and dropout, trained with AdamW and `ReduceLROnPlateau`. Optional Optuna TPE hyperparameter search on a held-out validation fold. Best-checkpoint restore on test R² plateau.

### PINNPFR (`src/models/pinn.py`)

Same topology as `SimpleNN` with an added physics-residual loss derived from the PFR governing equations (`src/physics/pfr_residuals.py`):

```
L = λ_data · MSE(ŷ, y)  +  λ_phys · L_physics
```

Constraints enforced: ideal-gas EOS, mass conservation (ρuA = ṁ), species sum = 1, species ≥ 0, energy ODE via autograd on `relative_position`. Curriculum warmup trains on data loss only for the first `CURRICULUM_WARMUP_EPOCHS` epochs before switching on the physics term.

### Symbolic Regression (`Main_9`)

PySR distillation from any NN teacher. Set `TEACHER_STEM` to `simple_nn_exit`, `simple_nn_full_profile`, or `pinn_pfr`; the notebook auto-selects model class, sampling strategy, and export directory. Output: human-readable Python equations importable by Main_10 with no PyTorch dependency.

### Bayesian Optimisation (`Main_10`)

Optuna `GPSampler` maximises olefin yield over six inlet degrees of freedom. Runs two independent studies (MLP surrogate and SR equations), then validates both optima with Cantera PFR simulations and reports surrogate prediction error alongside a parameter-space comparison plot.

---

## Quick Start

```bash
git clone https://github.com/karefyllidis/open_HydrAI.git
cd open_HydrAI
pip install -r requirements.txt
```

1. Install **Cantera** ([cantera.org](https://cantera.org)).
2. Place mechanism YAML files in `mechanisms/` (referenced by `configs/simulation/reactant_database.json`).
3. Run notebooks in order: `Main_1` → `Main_2` → `Main_3` → … → `Main_10`.

For cluster data generation:

```bash
bash scripts/dev/sbatch_safe.sh scripts/cluster/run_training_mul_GPUs.sh
bash scripts/monitor/monitor_cluster_jobs.sh
python scripts/dev/check_complete_runs.py
python scripts/dev/consolidate_training_data.py
```

Live training monitor (while Main_6 or Main_7 runs):

```bash
python scripts/monitor/monitor_nn_training_progress.py
```

CLI inference (requires trained model exports from Main_6 / Main_7 / Main_8):

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
- [x] Exit-plane PyTorch MLP (`Main_6`)
- [x] Full axial-profile PyTorch MLP with run-level split (`Main_7`)
- [x] Physics-informed neural network with PFR residual loss (`Main_8`)
- [x] Symbolic regression distillation from any NN teacher (`Main_9`)
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
