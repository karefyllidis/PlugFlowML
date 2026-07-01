# NN training logs (`data/logs/`)

Runtime progress files for the **Main_6** (SimpleNN, full axial profile) and **Main_7** (PINN) PyTorch notebooks. Not committed (see `.gitignore`); paths are stable so monitors and notebooks stay in sync.

## Files

| File | Written by | Contents |
|------|------------|----------|
| `<stem>_training_progress.csv` | Main_6 §8, Main_7 §7 | Per-epoch train MSE; rows with val/test MSE (Main_7: val rollout); Main_6 checkpoints add R² |
| `<stem>_optuna_tuning_plot_data.json` | §6b (when `IF_HYPERPARAM_TUNING=True`) | Incremental Optuna trial snapshot |

Stems (match notebook names):

- `Main_6_train_evaluate_SimpleNN_full_profile`
- `Main_7_train_evaluate_PINN_full_profile`

Path helpers: `src/utils/training_progress_log.py`.

## External monitor

From repo root:

```bash
python scripts/monitor/monitor_nn_training_progress.py
```

Edit flags at the top of `scripts/monitor/monitor_nn_training_progress.py`:

| Flag | Purpose |
|------|---------|
| `MAIN_6` / `MAIN_7` | Exactly one `True` — selects log filenames |
| `LIVE` | `False` = one-shot plot; `True` = refresh while the notebook runs (recommended for Main_7 §7) |

**Auto-selection:** the monitor plots whichever file was **modified most recently** (Optuna JSON during §6b, training CSV during §7/§8).

**Training view:** **Main_6** — three panels (train MSE, train/test R², gap). **Main_7** — one panel (teacher-forced train MSE + val rollout MSE, every epoch).

**Optuna view (§6b JSON):** trial history + parallel coordinates (validation R²).

**LIVE behaviour:**

- Waits up to **30s** for logs to appear, then exits with a message if none.
- Stops refreshing after **90s** with no size/mtime change and shows a final plot.
- Needs an interactive matplotlib backend (local desktop). On headless SSH, use `LIVE=False` or set `MPLBACKEND=Agg` and extend the script to save PNGs.

## Legacy paths

Older runs may have left `*_training_progress.csv` under `outputs/reports/` or `optuna_tuning_plot_data.json` under `outputs/figures/<stem>/`. Safe to delete; new runs write here only.
