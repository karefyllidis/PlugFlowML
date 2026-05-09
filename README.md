# HydrAI

Steam cracking simulations are accurate but slow: one detailed chemistry solve can take seconds to minutes, which makes rapid reactor screening difficult. HydrAI addresses this with a physics-grounded machine-learning surrogate trained on high-fidelity plug-flow reactor data. The result is millisecond-scale inference for design iteration while keeping chemistry-aware behavior anchored to simulation data.

**Measured on held-out test data from a large n-hexane dataset, mean test R² is ~0.97-0.99 across thermodynamic state variables and species concentrations.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Cantera](https://img.shields.io/badge/Cantera-3.2.0%2B-green)](https://cantera.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Nikolas Karefyllidis, PhD** — [GitHub](https://github.com/karefyllidis) · [LinkedIn](https://www.linkedin.com/in/karefyllidis/) · [Google Scholar](https://scholar.google.co.uk/citations?user=kLGU85cAAAAJ&hl=en)

Parts of the scientific foundation for this project originate from prior doctoral work: [University of Oxford thesis repository entry](https://ora.ox.ac.uk/objects/uuid:2479abe8-fefb-4574-b573-a309c278a614).

---

## Problem

High-fidelity steam-cracking simulation is computationally expensive, so using it directly for broad operating-space exploration is slow. This creates a bottleneck for iterative design, model comparison, and what-if studies across multiple feedstocks. HydrAI targets this gap by preserving a simulation-grade baseline while enabling fast approximations.

<details>
<summary>Background: why detailed simulation is expensive</summary>

Accurate predictions require stiff ODE integration coupled to detailed reaction mechanisms that can include 10^2-10^3 species. HydrAI uses Cantera-based PFR simulation as the reference baseline and trains surrogates against that data.

</details>

## Solution

HydrAI provides a reproducible workflow that pairs detailed simulation with machine-learning surrogates:

1. **Simulate** — Cantera PFR sweeps over broad operating conditions for multiple feedstocks (ethane, propane, n-hexane, naphtha).
2. **Train surrogates** — multi-output tree ensembles (RF, Gradient Boosting, XGBoost, AdaBoost), with optional `RandomizedSearchCV`.
3. **Evaluate** — held-out metrics (R², RMSE, MAPE, MBE) per model and per output target in dedicated notebooks.

## Results

- Inference latency shifts from **seconds-minutes** for full chemistry solves to **milliseconds** for surrogate predictions.
- Representative held-out accuracy on a large n-hexane dataset is mean test **R² ~ 0.97-0.99** across thermodynamic state variables and species concentrations.
- One surrogate can be trained across chemically distinct feedstocks rather than only interpolating within a single feed.

![Representative axial evolution](assets/axial_evolution.png)

*Typical axial evolution along normalized reactor length (z/L): temperature rise, pressure drop, and reactant depletion — the full-resolution targets HydrAI learns to predict.*

### Why It Matters

HydrAI delivers millisecond inference instead of second-to-minute chemistry solves, which substantially reduces iteration time during screening and early design studies. That speedup enables larger parameter sweeps and faster model-selection cycles without discarding a high-fidelity reference workflow. Because the surrogates are trained on detailed PFR simulations, the approach remains grounded in physically informed data. The same framework supports both local experimentation and cluster-scale data generation.

## Impact

- **Faster engineering loops:** screen operating windows in milliseconds instead of waiting for full chemistry solves.
- **Broader search coverage:** evaluate more feedstock-condition combinations within the same compute budget.
- **Better HPC utilization:** reserve expensive detailed simulations for dataset generation and final validation, then use surrogates for rapid iteration.
- **Reproducible decision support:** move from simulation to model evaluation with a documented notebook and config-driven workflow.

## How to Run

```bash
git clone https://github.com/karefyllidis/HydrAI.git
cd HydrAI
pip install -r requirements.txt
```

1. Install **Cantera** for your interpreter ([guide](https://cantera.org)).
2. Place mechanism YAML files in `mechanisms/` (configured in `configs/simulation/reactant_database.json`).
3. Run notebooks in order:
   - `Main_1` (single PFR run)
   - `Main_2` (training data generation)
   - `Main_3` (EDA + feature engineering)
   - `Main_4_train_and_evaluate_tree_models_IO` (baseline inlet-to-outlet evaluation)
   - `Main_5_train_evaluate_tune_tree_model_evolution` (one-model tuning + full PFR evolution)
4. Local parallel sweep: `python scripts/local/run_main2_local_parallel.py --ntasks 4`

For cluster execution, use `scripts/cluster/` and follow the post-run monitor/verify/consolidate workflow:
- Monitor: `bash scripts/dev/monitor_run.sh`
- Verify: `python scripts/dev/check_complete_runs.py`
- Consolidate: `python scripts/dev/consolidate_training_data.py` (`--no-cleanup` to keep task files, `--dry-run` to preview only)
- Continue with `notebooks/Main_3_data_exploration_feature_engineering.ipynb`

Cluster submission tip (CRLF-safe): `bash scripts/dev/sbatch_safe.sh scripts/cluster/run_training_mul_GPUs.sh`

Advanced references:
- CSD3-specific SLURM setup: [docs/HPC_GUIDE.md](docs/HPC_GUIDE.md)
- Figure export flags and species-lumping controls: [docs/SPECIES_LUMPING_MODEL_CARD.md](docs/SPECIES_LUMPING_MODEL_CARD.md)
- Full ML config keys: [docs/ML_CONFIG_GUIDE.md](docs/ML_CONFIG_GUIDE.md)
- Repository layout details: [docs/STRUCTURE.md](docs/STRUCTURE.md)

## Repository Structure

    HydrAI/
    ├── notebooks/            # Main_1 → Main_5  ·  PFR → sweep → EDA → baseline eval → tuned evolution
    ├── src/                  # cantera/, ml/, utils/
    ├── configs/              # simulation/, ml/, style/
    ├── scripts/              # cluster/, local/, dev/
    ├── data/                 # training/, processed/ (generated; git-ignored)
    ├── models/               # trained artifacts (generated; git-ignored)
    ├── mechanisms/           # local YAML kinetic files (git-ignored)
    └── docs/                 # guides, API reference, structure trees

## Roadmap

- [x] Multi-feed PFR with detailed chemistry
- [x] LHS / grid sampling, SLURM-aware parallel data generation
- [x] Multi-output tree surrogates, baseline comparison, and one-model hyperparameter tuning notebook
- [x] Species lumping for dimensionality reduction (by carbon number & chemistry role); optional lumped export for ML
- [x] Full axial-profile tuning workflow — `Main_5_train_evaluate_tune_tree_model_evolution.ipynb` tunes one selected tree model on full PFR evolution with `relative_position` as an input
- [ ] PyTorch / physics-informed neural surrogate
- [ ] Bayesian / gradient-free design optimisation loop

---

## Contributing · License

Contributions are welcome via pull requests. [MIT](LICENSE) © Nikolas Karefyllidis

## Model cards

- Repository model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- Hugging Face-ready template: [docs/HF_MODEL_CARD_TEMPLATE.md](docs/HF_MODEL_CARD_TEMPLATE.md)
- Training data generation protocol: [docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md](docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md)
- **Species lumping methodology** (grouping rules, sum-of-`Y_*` aggregation, export modes): [docs/SPECIES_LUMPING_MODEL_CARD.md](docs/SPECIES_LUMPING_MODEL_CARD.md)
