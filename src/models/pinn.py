"""Physics-Informed Neural Network for PFR surrogate (Main_7).

Standalone MLP backbone whose forward pass is differentiable with respect to
inputs, enabling torch.autograd.grad-based physics residuals in the training
loop without any coupling to the data-only SimpleNN used in Main_6.

N. Karefyllidis 2026
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNPFR(nn.Module):
    """Physics-Informed surrogate for full axial profile prediction.

    Identical MLP topology to SimpleNN but kept as a separate class so that
    Main_7 is self-contained and architectural changes to the PINN do not
    affect the data-only models.

    Parameters
    ----------
    in_features  : number of input features (inlet conditions + z/L)
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
