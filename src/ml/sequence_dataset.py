"""Run-level sequence datasets for axial PFR profile models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceColumnSpec:
    """Column groups for one simulation trajectory."""

    run_cols: list[str]
    design_cols: list[str]
    exog_cols: list[str]
    state_cols: list[str]


def run_id_series(df: pd.DataFrame, run_cols: list[str]) -> pd.Series:
    """Integer run id consistent with Main_7 groupby(run_cols).ngroup()."""
    return df.groupby(run_cols, dropna=False).ngroup()


def build_sequence_arrays(
    df: pd.DataFrame,
    spec: SequenceColumnSpec,
    run_ids: np.ndarray | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """One dict entry per run_id with sorted axial arrays."""
    if run_ids is None:
        run_ids = run_id_series(df, spec.run_cols).to_numpy()

    out: dict[int, dict[str, np.ndarray]] = {}
    sort_col = "z_position_m" if "z_position_m" in df.columns else spec.exog_cols[0]

    for rid in np.unique(run_ids):
        mask = run_ids == rid
        g = df.loc[mask].sort_values(sort_col)
        out[int(rid)] = {
            "design": g[spec.design_cols].iloc[0].to_numpy(dtype=np.float64),
            "exog": g[spec.exog_cols].to_numpy(dtype=np.float64),
            "state": g[spec.state_cols].to_numpy(dtype=np.float64),
            "z_position_m": g["z_position_m"].to_numpy(dtype=np.float64)
            if "z_position_m" in g.columns
            else np.arange(len(g), dtype=np.float64),
            "relative_position": g["relative_position"].to_numpy(dtype=np.float64)
            if "relative_position" in g.columns
            else np.linspace(0, 1, len(g)),
            "relative_tau": g["relative_tau"].to_numpy(dtype=np.float64)
            if "relative_tau" in g.columns
            else np.linspace(0, 1, len(g)),
        }
    return out


class ProfileSequenceDataset(Dataset):
    """One item = full axial profile for a single simulation run."""

    def __init__(self, sequences: dict[int, dict[str, np.ndarray]]):
        self.run_ids = sorted(sequences.keys())
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.run_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        rid = self.run_ids[idx]
        seq = self.sequences[rid]
        return {
            "run_id": rid,
            "design": torch.tensor(seq["design"], dtype=torch.float32),
            "exog": torch.tensor(seq["exog"], dtype=torch.float32),
            "state": torch.tensor(seq["state"], dtype=torch.float32),
            "length": seq["state"].shape[0],
        }


def collate_profile_sequences(
    batch: Sequence[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    """Pad variable-length profiles; ``lengths`` is the true step count per run."""
    lengths = torch.tensor([int(b["length"]) for b in batch], dtype=torch.long)
    design = torch.stack([b["design"] for b in batch], dim=0)
    exog = pad_sequence([b["exog"] for b in batch], batch_first=True, padding_value=0.0)
    state = pad_sequence([b["state"] for b in batch], batch_first=True, padding_value=0.0)
    run_ids = torch.tensor([int(b["run_id"]) for b in batch], dtype=torch.long)
    return {
        "run_id": run_ids,
        "design": design,
        "exog": exog,
        "state": state,
        "lengths": lengths,
    }


def masked_delta_mse(
    pred_delta: torch.Tensor,
    true_state: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """MSE on state increments; mask padded transitions."""
    # true delta: state[t+1]-state[t] for t in 0..L-2
    true_delta = true_state[:, 1:, :] - true_state[:, :-1, :]
    pred = pred_delta
    if pred.size(1) > true_delta.size(1):
        pred = pred[:, : true_delta.size(1), :]
    elif pred.size(1) < true_delta.size(1):
        true_delta = true_delta[:, : pred.size(1), :]

    sq = (pred - true_delta) ** 2
    batch, n_steps, _ = sq.shape
    mask = torch.zeros(batch, n_steps, device=sq.device, dtype=sq.dtype)
    for i, L in enumerate(lengths.tolist()):
        n_trans = max(0, int(L) - 1)
        if n_trans > 0:
            mask[i, :n_trans] = 1.0
    denom = mask.sum().clamp_min(1.0)
    return (sq.mean(dim=-1) * mask).sum() / denom


def masked_rollout_mse(
    pred_states: torch.Tensor,
    true_states: torch.Tensor,
    lengths: torch.Tensor,
    *,
    reduction: str = "none",
) -> torch.Tensor:
    """MSE between rollout states and truth, masked over valid transitions.

    reduction
        ``"none"`` — per-run vector (B,); ``"mean"`` — scalar mean over finite runs.
    """
    n_steps = min(pred_states.size(1), true_states.size(1)) - 1
    if n_steps < 1:
        return torch.tensor(0.0, device=pred_states.device)
    pred = pred_states[:, : n_steps + 1, :]
    true = true_states[:, : n_steps + 1, :]
    err = (pred[:, 1:, :] - true[:, 1:, :]) ** 2
    mask = torch.zeros(err.size(0), err.size(1), device=err.device, dtype=err.dtype)
    for i, L in enumerate(lengths.tolist()):
        n_trans = max(0, int(L) - 1)
        if n_trans > 0:
            mask[i, :n_trans] = 1.0
    per_step = err.mean(dim=-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_run = (per_step * mask).sum(dim=1) / denom
    pred_ok = torch.isfinite(pred[:, 1:, :]).all(dim=(1, 2))
    per_run = torch.where(
        pred_ok & torch.isfinite(per_run),
        per_run,
        torch.full_like(per_run, float("nan")),
    )
    if reduction == "none":
        return per_run
    if reduction != "mean":
        raise ValueError(f"reduction must be 'none' or 'mean', got {reduction!r}")
    ok = torch.isfinite(per_run)
    if ok.any():
        return per_run[ok].mean()
    return torch.tensor(float("nan"), device=per_run.device, dtype=per_run.dtype)
