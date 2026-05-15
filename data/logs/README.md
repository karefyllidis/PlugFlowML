# NN training logs (`data/logs/`)

Runtime progress files for **Main_6** and **Main_7** PyTorch notebooks. Not committed (see `.gitignore`); paths are stable so monitors and notebooks stay in sync.

## Files

| File | Written by | Contents |
|------|------------|----------|
| `<stem>_training_progress.csv` | Main_6 / Main_7 §8 | Per-epoch train MSE; checkpoint rows add test MSE/R², LR |
| `<stem>_optuna_tuning_plot_data.json` | §6b (when `IF_HYPERPARAM_TUNING=True`) | Incremental Optuna trial snapshot |

Stems (match notebook names):

- `Main_6__train_evaluate_SimpleNN_IO`
- `Main_7_train_evaluate_SimpleNN_full_profile`

Path helpers: `src/utils/training_progress_log.py`.

## Live plots

From repo root:

```bash
python scripts/monitor/monitor_nn_training_progress.py
```

Edit flags at the top of that script:

- `MAIN_6` or `MAIN_7` (exactly one `True`)
- `LIVE=True` — refresh until the log is idle ~90s; `False` — one-shot plot

The monitor **auto-picks whichever log was updated most recently** (Optuna JSON during §6b, training CSV during §8).

## Legacy paths

Older runs may have left `*_training_progress.csv` under `outputs/reports/` or `optuna_tuning_plot_data.json` under `outputs/figures/<stem>/`. Safe to delete; new runs write here only.
