# Hugging Face Model Card Template (HydrAI)

Use this as the `README.md` inside a Hugging Face model repository.

## Model Description

This model is part of **HydrAI**: a physics-grounded steam-cracking simulation framework with machine-learning surrogates for fast reactor screening and design.

- **Model type:** Multi-output regression surrogate
- **Base method:** *(fill in for the specific release — one of: tree ensemble [RF/GBM/XGBoost/AdaBoost, Main_4/5]; `SimpleNN` PyTorch MLP, full axial profile [Main_6]; `PINNPFR` physics-informed MLP [Main_7]; PySR symbolic equations distilled from a NN teacher [Main_8])*
- **Primary domain:** Steam cracking plug-flow reactors

## Intended Uses

- Fast approximation of axial reactor trajectories for screening workflows
- Candidate ranking before higher-cost first-principles simulation

## Out-of-Scope Uses

- Safety-critical or compliance decisions without physics-model verification
- Extrapolation far beyond training conditions

## Training Data

- Source: Synthetic high-fidelity simulations generated with Cantera
- Coverage: Multiple feedstocks and operating conditions
- Notes: Data quality and domain coverage determine surrogate reliability

## Evaluation

- Report project-specific metrics here (R², RMSE, MAPE, MBE).
- Example from HydrAI experiments: mean test **R² ~ 0.97–0.99** on key targets (dataset-dependent).

## Risks and Limitations

- Domain shift can reduce prediction quality
- Mechanism differences or unseen feeds may require retraining
- Should be used with uncertainty checks and periodic full-solver validation

## How to Use

Provide minimal inference code and expected input schema in this section.

```python
# placeholder: load model artifact and predict
# y_pred = predictor.predict(X)
```

## Citation

If you use this model, cite the HydrAI repository:

- <https://github.com/karefyllidis/HydrAI>

## Model Maintainer

- Nikolas Karefyllidis
