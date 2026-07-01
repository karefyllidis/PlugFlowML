"""Shared SimpleNN architecture used by Main_6 (full profile), Main_7 (PINN).

3 hidden layers + ReLU + Dropout, linear regression head.
Dropout is active under model.train() and disabled by model.eval(), so test-time
predictions are deterministic. MC-Dropout inference (uncertainty estimation) is
obtained by calling predict_with_uncertainty() with model.train() temporarily
re-enabled.

N. Karefyllidis 2026
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """Multi-output MLP surrogate for PFR state prediction.

    Parameters
    ----------
    in_features  : number of input features
    h1, h2, h3  : hidden layer widths
    out_features : number of output targets
    dropout      : dropout probability (active during training; off at eval)
    """

    def __init__(
        self,
        in_features: int,
        h1: int,
        h2: int,
        h3: int,
        out_features: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1     = nn.Linear(in_features, h1)
        self.fc2     = nn.Linear(h1, h2)
        self.fc3     = nn.Linear(h2, h3)
        self.out     = nn.Linear(h3, out_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.out(x)

    def predict_with_uncertainty(
        self,
        X: torch.Tensor,
        n_samples: int = 50,
        scaler_y=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC-Dropout uncertainty estimate.

        Runs ``n_samples`` stochastic forward passes with dropout active,
        returns mean and std of predictions in physical units.

        Parameters
        ----------
        X        : (N, n_features) scaled input tensor on the correct device
        n_samples: number of MC samples
        scaler_y : fitted StandardScaler for targets (optional; if None, returns scaled units)

        Returns
        -------
        mean : (N, n_outputs) array of mean predictions
        std  : (N, n_outputs) array of per-output standard deviations (epistemic proxy)
        """
        self.train()  # activate dropout
        with torch.no_grad():
            preds = torch.stack([self(X) for _ in range(n_samples)], dim=0)  # (S, N, O)
        self.eval()

        preds_np = preds.cpu().numpy()
        mean_s   = preds_np.mean(axis=0)  # (N, O)
        std_s    = preds_np.std(axis=0)   # (N, O)

        if scaler_y is not None:
            mean = scaler_y.inverse_transform(mean_s)
            std  = std_s * scaler_y.scale_  # propagate scale only (not shift)
        else:
            mean, std = mean_s, std_s

        return mean, std
