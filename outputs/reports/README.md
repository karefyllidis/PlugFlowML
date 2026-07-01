# HydrAI Reports

Use this folder for concise run summaries and important findings from the `Main_*`
notebooks. Keep generated figures in `outputs/figures/`, trained artifacts in
`models/`, **Main_6 / Main_6 live-training CSV and Optuna JSON** in
[`data/logs/`](../../data/logs/README.md), and human-readable conclusions here.

## Kinds of files here

1. **Curated reports (`*.md`)** - hand-written summaries (tracked in git).
2. **Run logs (`<NotebookName>.txt`)** - raw terminal output captured automatically
   by each notebook through `src.utils.run_log.start_run_log(...)`. The file path is
   stable (for example `Main_6__train_evaluate_SimpleNN_IO.txt`) and is **overwritten**
   on each run so the folder always reflects the latest execution. These logs are not
   committed (machine-specific, can be large) but live on disk next to the curated
   `.md` reports.

## Recommended report files

- `Main_1_run_pfr.md` - single-run PFR setup, sanity checks, and physics notes.
- `Main_2_generate_training_data.md` - campaign size, parameter coverage, failed runs, and data quality.
- `Main_3_data_exploration_feature_engineering.md` - target definitions, species lumping choices, and feature/target statistics.
- `Main_4_train_and_evaluate_tree_models_IO.md` - baseline model comparison and exit-plane conclusions.
- `Main_5_train_evaluate_tune_tree_model_evolution.md` - tuned model, axial/full-profile findings, speedup, and export notes.
- `Main_6__train_evaluate_SimpleNN_IO.md` / `Main_6_train_evaluate_SimpleNN_full_profile.md` - PyTorch NN training, Optuna, and export notes (optional).

**Not in this folder:** `*_training_progress.csv` and `*_optuna_tuning_plot_data.json` (see `data/logs/`).

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
