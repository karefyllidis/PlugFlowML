# HydrAI Reports

Use this folder for concise run summaries and important findings from the `Main_*`
notebooks. Keep generated figures in `outputs/figures/`, trained artifacts in
`models/`, and human-readable conclusions here.

## Recommended Files

- `Main_1_run_pfr.md` - single-run PFR setup, sanity checks, and physics notes.
- `Main_2_generate_training_data.md` - campaign size, parameter coverage, failed runs, and data quality.
- `Main_3_data_exploration_feature_engineering.md` - target definitions, species lumping choices, and feature/target statistics.
- `Main_4_train_and_evaluate_tree_models_IO.md` - baseline model comparison and exit-plane conclusions.
- `Main_5_train_evaluate_tune_tree_model_evolution.md` - tuned model, axial/full-profile findings, speedup, and export notes.

## Report Template

```markdown
# <Notebook Name> Report

## Run Context
- Date:
- Data file(s):
- Key config:

## Important Findings
- 

## Quality Checks
- 

## Figures / Artifacts
- 

## Decisions / Follow-Up
- 
```
