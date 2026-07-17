---
name: model-card
description: Write or update a PlugFlowML model card (docs/MODEL_CARD.md and docs/HF_MODEL_CARD_TEMPLATE.md) for a trained surrogate. Use after exporting artifacts from Main_4 through Main_8 (tree ensemble, SimpleNN, PINN, or SR) — architecture, training-data lineage, metrics, intended use, and limitations — before sharing a model or publishing to Hugging Face.
license: MIT
---

# Model Card

A model card lets someone decide whether to trust and reuse a trained surrogate without reading the notebook.

## Collect

- **Model family & workflow** — which `Main_N` notebook(s) produced it, algorithm (tree ensemble / SimpleNN / PINNPFR / PySR equations), exit-plane vs full axial profile.
- **Inputs/outputs** — feature list, target list (state/thermo/aero vs species), whether targets are raw `Y_*` or lumped `Y_lump_*`, and whether `relative_position` is an input (full-profile models only).
- **Training data lineage** — source `features_targets_*.pkl` run-stamp (Main_3), feedstock(s), sampling method, run-level train/test split.
- **Physics constraints** (PINN only) — `pinn.loss_weights.*`, curriculum warmup length, which conservation laws are enforced (EOS, mass, species sum/non-negativity, energy ODE).
- **Distillation lineage** (SR only) — teacher stem (`simple_nn_full_profile` or `pinn_pfr`), distillation sample count, resulting equation form.
- **Reported metrics** — R²/MAE/RMSE/MAPE, species-lump NMAE (%), inference speed vs Cantera.
- **Intended use / out-of-scope** — reuse the framing already in `docs/MODEL_CARD.md` (screening/design-exploration, not a substitute for full Cantera validation) unless this model genuinely changes it.
- **Reproducibility pointers** — `configs/ml/mainN_*.json` used, exported artifact paths under `models/`.

## Output structure

Update two documents in lockstep — they serve different audiences and must not drift:

1. **`docs/MODEL_CARD.md`** — internal, full detail. Mirror the existing per-workflow structure (it currently has `### Baseline Exit-Plane Evaluation (Main_4)` and `### Tuned Exit + Full Evolution (Main_5)` under "Training and Evaluation Workflows"); add a new `###` subsection for the model you're documenting rather than a new top-level file.
2. **`docs/HF_MODEL_CARD_TEMPLATE.md`** — external, ready to drop in as a Hugging Face repo `README.md`. Update only if this model is intended for external/HF release.

Cross-reference instead of duplicating:
- Species lumping methodology → `docs/SPECIES_LUMPING_MODEL_CARD.md`
- Data-generation protocol → `docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md`
- Column-level schema of the training data itself → a dataset card for `features_targets_*.pkl`, not this model card.

## PlugFlowML-specific pitfalls

- `docs/MODEL_CARD.md` currently documents only Main_4/Main_5 (tree ensembles). Before assuming it already covers a model, check — it may have gone stale relative to Main_6 (SimpleNN), Main_7 (PINN), or Main_8 (SR), which have no card sections yet.
- Don't restate dataset columns/units/provenance here — that's a dataset-card's job, not a model card's.
- Mass fractions only. State species outputs as `Y_*` / `Y_lump_*`; never describe a mole-fraction (`X_*`) output.
- Pair with `split-audit` before writing "Training Data" (verify the run-level split rather than assuming it), and with `run-comparison` before writing "Reported Metrics" when the card compares this model against a prior version (e.g., Main_4 default vs Main_5 tuned).
- For PINN cards, state which physics losses are enforced vs merely logged (diagnostic) — Main_7 §13 compares trained-vs-untrained residuals but that is a diagnostic, not a training constraint.
