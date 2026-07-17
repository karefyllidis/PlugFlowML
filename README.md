# PlugFlowML — ML surrogates for physical simulation, taught on a real reactor

**A free, 10-lesson course for graduate students in any simulation-heavy field.** You'll take an expensive physics simulator and learn, hands-on, how to replace it with machine-learning surrogates: tree ensembles → neural networks → physics-informed neural networks → symbolic-regression equations → Bayesian optimisation over the result. The worked example is a chemical plug-flow reactor, but every method is taught as a transferable tool — if your field runs stiff ODEs, CFD, climate kernels, or battery models, this pipeline maps onto it.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karefyllidis/PlugFlowML/blob/main/notebooks/Main_3_data_exploration_feature_engineering.ipynb)
[![Cantera](https://img.shields.io/badge/Cantera-3.2.0%2B-green)](https://cantera.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.4%2B-lightblue?logo=optuna&logoColor=white)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Start here

**No install needed.** Every hands-on lesson runs on free Google Colab: open it, run the bootstrap cell (it fetches the code and a 150-run sample dataset), and you're working within a minute.

👉 **[Open Lesson 3 in Colab](https://colab.research.google.com/github/karefyllidis/PlugFlowML/blob/main/notebooks/Main_3_data_exploration_feature_engineering.ipynb)** — the first hands-on lesson.

Or run locally:

```bash
git clone https://github.com/karefyllidis/PlugFlowML.git
cd PlugFlowML
pip install -r requirements.txt
jupyter lab notebooks/
```

(The sample dataset downloads automatically via each lesson's bootstrap cell; local users get the same files.)

## The course

| Lesson | Notebook | You learn | Mode |
|---|---|---|---|
| 1 | [Main_1](notebooks/Main_1_run_pfr.ipynb) | Simulation as a data source — why stiff ODE ground truth is expensive | **hands-on** |
| 2 | [Main_2](notebooks/Main_2_generate_training_data.ipynb) | Design of experiments: Latin-Hypercube sampling, parallel batch generation | **hands-on** |
| 3 | [Main_3](notebooks/Main_3_data_exploration_feature_engineering.ipynb) | EDA on simulation data; domain-driven dimensionality reduction (species lumping) | **hands-on** |
| 4 | [Main_4](notebooks/Main_4_train_and_evaluate_tree_models_IO.ipynb) | Tree-ensemble baselines and honest per-target metrics | **hands-on** |
| 5 | [Main_5](notebooks/Main_5_train_evaluate_tune_tree_model_evolution.ipynb) | Bayesian hyperparameter tuning; predicting full spatial profiles | **hands-on** |
| 6 | [Main_6](notebooks/Main_6_train_evaluate_SimpleNN_full_profile.ipynb) | Neural surrogates done honestly: leakage-safe splits, physics-aware evaluation | **hands-on** |
| 7 | [Main_7](notebooks/Main_7_train_evaluate_PINN_full_profile.ipynb) | Physics-informed neural networks: residual losses, collocation, curricula | **hands-on** |
| 8 | [Main_8](notebooks/Main_8_symbolic_regression_SR.ipynb) | Distilling networks into closed-form equations (PySR) | **hands-on** |
| 9 | [Main_9](notebooks/Main_9_compare_cantera_pinn_sr.ipynb) | Validating surrogates against ground truth; error accumulation through a model chain | **hands-on** |
| 10 | [Main_10](notebooks/Main_10_optimisation_BO_surrogate_vs_cantera.ipynb) | Bayesian optimisation over a surrogate — and why you validate the optimum | **hands-on** |

Each lesson opens with learning objectives, teaches concepts in short boxes at the point of use, and ends sections with exercises (solutions included, collapsed). Default configs are sized for free Colab — minutes per lesson, not hours; every budget is a config knob you can turn up (`configs/ml/`, research-scale values documented inline).

Every lesson is fully hands-on — Lessons 1–2 run a real Cantera simulation live, using an openly-licensed n-hexane pyrolysis mechanism ([mechanisms/naptha_example.yaml](mechanisms/naptha_example.yaml)) built for this course; everything from Lesson 3 onward works off the sample dataset instead.

## Who this is for

Graduate students (or anyone) comfortable with Python and basic ML vocabulary. **No chemical-engineering background is assumed** — the reactor physics you need is explained in one or two sentences where it matters. What you should bring: numpy/pandas basics, and curiosity about how ML earns its place next to a trusted simulator.

## The example system

Steam cracking of n-hexane in a plug-flow reactor, simulated with [Cantera](https://cantera.org). Each simulation integrates detailed chemistry along the reactor axis and is expensive enough (10²–10³ species mechanisms) that exploring a design space directly is prohibitive — exactly the situation where a millisecond-scale surrogate pays for itself.

### Sample dataset (what the lessons use)

| Field | Value |
|---|---|
| Runs | 150 (Latin Hypercube over 6 inlet parameters, seed 7) |
| Feedstock | n-hexane |
| Mechanism | 153 species, 2 146 reactions (Cantera 3.2, wall heat flux PFR) |
| Inlet T / P | 800–900 K / 1.5–3.5 bar |
| Reactor L / D | 10–15 m / 25–40 mm |
| Mass flow / heat flux | 0.05–0.10 kg/s / 100–250 kW/m² |
| Axial resolution | 200 steps per run |
| Targets | 9 state/thermo variables + 9 lumped species mass fractions |
| Split | 80/20 at run level (no profile leakage) |

Distributed as a [GitHub Release](https://github.com/karefyllidis/PlugFlowML/releases) asset; full details in [docs/SAMPLE_DATASET_CARD.md](docs/SAMPLE_DATASET_CARD.md). The research campaign behind the original project used the same domain at 46 000 runs — the lessons' methods are identical, only budgets differ.

*Note: this dataset was generated with a research-grade mechanism, separate from the smaller open mechanism Lessons 1–2 run live — see [mechanisms/naptha_example.yaml](mechanisms/naptha_example.yaml) for that one's own provenance.*

### Full processed dataset (optional, research-scale)

Don't want to run Lessons 1–3 yourself just to get a processed feature/target set? The full research campaign behind this course — 45 932 runs, ~9.2M rows, same 6-parameter domain and mechanism as the sample above — is available for direct download: [full processed dataset](https://onedrive.live.com/my?id=%2Fpersonal%2F3e111f055e5954cc%2FDocuments%2FPlugFlowML&viewid=53d5a810%2Dc8e2%2D4884%2D9c42%2De2eb4d2f9471). Drop the `.pkl` into `data/processed/` and Lessons 4–10 will pick it up via each lesson's `processed_stem` config. This is far heavier than free-Colab budgets are tuned for — the course itself is designed and tested against the 150-run sample above.

## What's under the hood

The course runs on a research-grade pipeline you can reuse directly:

- **`SimpleNN`** ([src/models/simple_nn.py](src/models/simple_nn.py)) — 3-layer MLP, AdamW + `ReduceLROnPlateau`, optional Optuna search, MC-Dropout uncertainty.
- **`PINNPFR`** ([src/models/pinn.py](src/models/pinn.py)) — same topology plus PFR residual losses ([src/physics/pfr_residuals.py](src/physics/pfr_residuals.py)): ideal-gas EOS, mass conservation (ρuA = ṁ), species sum/positivity, energy ODE via autograd. `L = λ_data·MSE + λ_phys·L_physics` with curriculum warmup.
- **Symbolic regression** (Main_8) — PySR distillation from either NN teacher into closed-form NumPy equations, consumed by Lessons 9–10.
- **Bayesian optimisation** (Main_10) — Optuna `GPSampler` over the SR surrogate, validated against the real simulator.
- **Cantera PFR wrapper** ([src/cantera/pfr_simulator.py](src/cantera/pfr_simulator.py)) and SLURM-ready parallel data generation ([scripts/](scripts/)) for building your own campaigns.

```
PlugFlowML/
├── notebooks/          # the course: Main_1 → Main_10
├── src/                # cantera wrapper, models, physics residuals, ml utils
├── configs/            # per-lesson JSON configs (Colab-scale defaults)
├── scripts/            # cluster/, local/, monitor/, dev/ + CLI inference
├── data/               # sample dataset lands here     (git-ignored)
├── models/             # trained artifacts              (git-ignored)
├── mechanisms/         # kinetic YAML files             (bring your own)
└── docs/               # guides, model cards, course style guide
```

Full tree: [docs/STRUCTURE.md](docs/STRUCTURE.md) · conventions: [docs/PLUGFLOWML_PROJECT_CONVENTIONS.md](docs/PLUGFLOWML_PROJECT_CONVENTIONS.md) · config keys: [docs/ML_CONFIG_GUIDE.md](docs/ML_CONFIG_GUIDE.md)

## Adapting this to your own system

The course is designed to be forked. To point the pipeline at your own simulator: generate runs in the Main_2 format (a tidy table of `run × axial-position × state`), re-run Lesson 3 to export features/targets, and Lessons 4–10 work unchanged. The exercises in each lesson end with prompts for mapping the method to your domain.

## Questions, bugs, contributions

- **Questions** → [GitHub Discussions](https://github.com/karefyllidis/PlugFlowML/discussions)
- **Something broken in a lesson** → [open an issue](https://github.com/karefyllidis/PlugFlowML/issues/new/choose)
- **Fixes and improvements welcome** → see [CONTRIBUTING.md](CONTRIBUTING.md)

## Citing

If this course or pipeline is useful in your teaching or research, please cite it (see [CITATION.cff](CITATION.cff) — GitHub's "Cite this repository" button gives BibTeX/APA).

Parts of the scientific foundation originate from prior doctoral work: [University of Oxford thesis](https://ora.ox.ac.uk/objects/uuid:2479abe8-fefb-4574-b573-a309c278a614).

---

**Nikolas Karefyllidis, PhD** — [Website](https://karefyllidis.github.io/) · [GitHub](https://github.com/karefyllidis) · [LinkedIn](https://www.linkedin.com/in/karefyllidis/) · [Google Scholar](https://scholar.google.co.uk/citations?user=kLGU85cAAAAJ&hl=en)

MIT © Nikolas Karefyllidis
