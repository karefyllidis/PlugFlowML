# HydrAI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Cantera](https://img.shields.io/badge/Cantera-3.2.0%2B-green)](https://cantera.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **HydrAI = Hydrocarbon + AI** — physics-grounded steam-cracking simulation with machine-learning surrogates for fast reactor screening and design.

**Nikolas Karefyllidis, PhD** — [GitHub](https://github.com/karefyllidis) · [LinkedIn](https://www.linkedin.com/in/karefyllidis/) · [Google Scholar](https://scholar.google.co.uk/citations?user=kLGU85cAAAAJ&hl=en)

Parts of the scientific foundation for this project originate from prior doctoral work: [University of Oxford thesis repository entry](https://ora.ox.ac.uk/objects/uuid:2479abe8-fefb-4574-b573-a309c278a614).

---

## Why it matters

Steam cracking is central to olefin production and one of the most energy-intensive unit operations in the chemical industry. Accurate predictions demand **stiff ODE integration coupled to detailed kinetics** — mechanisms of 10²–10³ species — fidelity that is essential for R&D but far too slow for iterative design.

HydrAI closes that gap with a **reproducible, full-stack workflow**:

1. **Simulate** — Cantera PFR solver sweeping a broad operating space across multiple feedstocks (ethane, propane, n-hexane, naphtha).
2. **Surrogateify** — multi-output tree ensembles trained on that data; inference is **milliseconds vs. seconds–minutes** for a full chemistry solve.
3. **Evaluate** — rigorous held-out metrics (R², RMSE, MAPE, MBE) per model and per output target in a dedicated comparison notebook.

Representative accuracy on a large n-hexane dataset: mean test **R² ~ 0.97–0.99** across all thermodynamic state variables and species concentrations.

---

## What you get

| | |
|--|--|
| **High-fidelity baseline** | PFR with configurable wall heat flux, Churchill pressure drop, and multi-feed YAML kinetics (35–1951 species). |
| **Scalable dataset generation** | Latin Hypercube or structured grid sweeps over 6 parameters; parallel on a workstation or SLURM-chunked on HPC. |
| **Multi-reactant generalisation** | One surrogate trained across chemically distinct feedstocks — not just interpolation within a single feed. |
| **Production ML pipeline** | RF, Gradient Boosting, XGBoost, AdaBoost; optional `BayesSearchCV` tuning in Main_5; `MLPFRPredictor` for sub-ms batch inference. |
| **Clean, extensible architecture** | JSON configs per concern (simulation / ML / style); Jupyter notebooks as the end-to-end interface; importable `src/` library. |

---

## Axial profiles

![Representative axial evolution](assets/axial_evolution.png)

*Typical axial evolution along normalized reactor length (z/L): temperature rise, pressure drop, and reactant depletion — the full-resolution targets HydrAI learns to predict.*

---

## Repository structure

    HydrAI/
    ├── notebooks/            # Main_1 → Main_5  ·  PFR → sweep → EDA → baseline eval → tuned evolution
    ├── src/                  # cantera/, ml/, utils/
    ├── configs/              # simulation/, ml/, style/
    ├── scripts/              # cluster/, local/, dev/
    ├── data/                 # training/, processed/ (generated; git-ignored)
    ├── models/               # trained artifacts (generated; git-ignored)
    ├── mechanisms/           # local YAML kinetic files (git-ignored)
    └── docs/                 # guides, API reference, structure trees

---

## Get started

```bash
git clone https://github.com/karefyllidis/HydrAI.git
cd HydrAI
pip install -r requirements.txt
```

1. Install **Cantera** for your interpreter ([guide](https://cantera.org/stable/install/windows.html)).
2. Place mechanism **YAML** files in `mechanisms/` — paths declared in `configs/simulation/reactant_database.json`.
3. Run notebooks in order under `notebooks/`:
   - `Main_1` → single PFR run
   - `Main_2` → training data generation
   - `Main_3` → EDA and feature engineering
   - `Main_4_train_and_evaluate_tree_models_IO` → baseline tree evaluation (inlet→outlet, no tuning)
  - `Main_5_train_evaluate_tune_tree_model_evolution` → one-model exit + full-PFR workflow (with or without tuning)
4. Parallel sweeps on one machine: `python scripts/local/run_main2_local_parallel.py --ntasks 4`.

→ Full config keys: [docs/ML_CONFIG_GUIDE.md](docs/ML_CONFIG_GUIDE.md) · Detailed layout: [docs/STRUCTURE.md](docs/STRUCTURE.md)

### HPC note

The scripts under `scripts/cluster/` are currently tuned for the **University of Cambridge CSD3** SLURM environment (accounts, partitions, and module names). On CSD3 **ampere**, GPU jobs cap CPUs per GPU (e.g. 32 CPUs per 1 GPU); parallel `srun` workers use **`--ntasks=N --cpus-per-task=1`**, not one task with `N` CPUs, or `srun` will fail with “More processors requested than permitted.” GPU **smoke** jobs use **`--time=00:10:00`** and **`--qos=INTR`** (interactive-style short runs). For multi-hour production sweeps, use a non-interactive QoS and longer `--time` in `run_training_mul_CPUs.sh` or a custom `#SBATCH` header. For other clusters, edit `#SBATCH` and `module load` before submission.

**Post-run workflow:**
1. Monitor: `bash scripts/dev/monitor_run.sh`
2. Verify: `python scripts/dev/check_complete_runs.py`
3. Consolidate: `python scripts/dev/consolidate_training_data.py`
   - Default behavior: merges task outputs, writes **`training_data_complete_*.pkl`**, and cleans old `data/training/task_*` artifacts.
   - Keep task files: `python scripts/dev/consolidate_training_data.py --no-cleanup`
   - Preview only: `python scripts/dev/consolidate_training_data.py --dry-run`
4. Continue to `notebooks/Main_3_data_exploration_feature_engineering.ipynb`

**Cluster submission tip (avoids CRLF `sbatch` failures):**
- Use: `bash scripts/dev/sbatch_safe.sh scripts/cluster/run_training_mul_GPUs.sh`
- The wrapper auto-converts DOS line endings to Unix LF before `sbatch`.

**Figure export controls (notebooks):**
- `Main_1_run_pfr.ipynb`: `IF_SAVE_PLOTS=True` saves quick figures to `outputs/figures/Main_1_run_pfr/`.
- `Main_3_data_exploration_feature_engineering.ipynb`: `IF_SAVE_EDA_PLOTS=True` saves EDA figures to `outputs/figures/Main_3_data_exploration_feature_engineering/eda/`.
  - Includes exit-plane distributions and species-lumping visual aids (bar charts by carbon number and by chemistry role).
  - **ML targets use mass fractions only** (`Y_*`). Mole fractions (`X_*`) may still appear in raw training pickles from Main_2 but are **not** written into `df_target` or used as surrogate outputs.
  - Lumping flags (organize species for EDA and optional export):
    - `IF_SEPARATE_SPECIES_BY_CARBON` — buckets such as `C1`, `C2`, … + `inert`
    - `IF_CATEGORIZE_BY_CHEMISTRY` — process roles: olefins, aromatics, paraffins, coke precursors, radicals, feedstock, **hydrogen** (H₂), diluent, other
  - **`EXPORT_SPECIES_AS`** (`individual` \| `lumped_chemistry` \| `lumped_carbon`): when lumped, individual `Y_*` columns in the exported pickle are replaced by summed **`Y_lump_*`** columns (smaller `data/processed/features_targets_*.pkl`; Main_4 trains on those directly). Requires the matching lumping flag above.
  - Full taxonomy and aggregation rules: [docs/SPECIES_LUMPING_MODEL_CARD.md](docs/SPECIES_LUMPING_MODEL_CARD.md)

---

## Roadmap

- [x] Multi-feed PFR with detailed chemistry
- [x] LHS / grid sampling, SLURM-aware parallel data generation
- [x] Multi-output tree surrogates, baseline comparison, and one-model hyperparameter tuning notebook
- [x] Species lumping for dimensionality reduction (by carbon number & chemistry role); optional lumped export for ML
- [x] Full axial-profile workflow — `Main_5_train_evaluate_tune_tree_model_evolution.ipynb` trains one selected tree model on full PFR evolution with `relative_position` as an input (optional hyperparameter tuning)
- [ ] PyTorch / physics-informed neural surrogate
- [ ] Bayesian / gradient-free design optimisation loop

---

## Contributing · License

[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) · [MIT](LICENSE) © Nikolas Karefyllidis

## Model cards

- Repository model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- Hugging Face-ready template: [docs/HF_MODEL_CARD_TEMPLATE.md](docs/HF_MODEL_CARD_TEMPLATE.md)
- Training data generation protocol: [docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md](docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md)
- Species lumping methodology (grouping rules, sum-of-`Y_*` aggregation, export modes): [docs/SPECIES_LUMPING_MODEL_CARD.md](docs/SPECIES_LUMPING_MODEL_CARD.md)
