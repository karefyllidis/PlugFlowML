"""Quick smoke test for Main_8 rollout stability (no full dataset)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.profile_gru import ProfileGRU
from src.ml.profile_rollout_eval import eval_rollout_loss
from src.ml.sequence_dataset import ProfileSequenceDataset, collate_profile_sequences


def _fake_sequences(n_runs: int = 12, t: int = 20, n_state: int = 8, n_exog: int = 2, n_design: int = 3):
    seqs = {}
    for rid in range(n_runs):
        state = np.random.randn(t, n_state).astype(np.float32) * 0.5
        seqs[rid] = {
            "design": np.random.randn(n_design).astype(np.float32),
            "exog": np.random.randn(t, n_exog).astype(np.float32),
            "state": state,
        }
    return seqs


def main() -> None:
    device = torch.device("cpu")
    n_state, n_exog, n_design = 8, 2, 3
    seqs = _fake_sequences()
    ds = ProfileSequenceDataset(seqs)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_profile_sequences)
    model = ProfileGRU(n_state, n_exog, n_design, hidden_size=32).to(device)
    loss, bad = eval_rollout_loss(
        model,
        loader,
        device,
        max_delta_scaled=2.0,
        max_state_scaled=10.0,
        log_first_bad=True,
    )
    assert np.isfinite(loss), f"expected finite val loss, got {loss} bad={bad}"
    print(f"OK: val rollout MSE={loss:.6f} (finite)")


if __name__ == "__main__":
    main()
