"""GRU profile model: predict state increments along reactor residence time."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ProfileGRU(nn.Module):
    """One-layer GRU with design-encoded initial hidden state.

    At each step the GRU sees ``[state_t, exog_t]`` and predicts ``delta_state_t`` in the
    same scaled space as ``state_t``, so ``state_{t+1} = state_t + delta_state_t`` during rollout.
    """

    def __init__(
        self,
        n_state: int,
        n_exog: int,
        n_design: int,
        hidden_size: int = 64,
        design_hidden: int = 32,
        num_gru_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_state = n_state
        self.hidden_size = hidden_size
        self.num_gru_layers = int(num_gru_layers)
        self.gru_input_dim = n_state + n_exog

        enc_layers: list[nn.Module] = [
            nn.Linear(n_design, design_hidden),
            nn.ReLU(),
        ]
        if dropout and dropout > 0:
            enc_layers.append(nn.Dropout(p=float(dropout)))
        enc_layers.append(nn.Linear(design_hidden, hidden_size))
        self.design_encoder = nn.Sequential(*enc_layers)
        gru_dropout = float(dropout) if self.num_gru_layers > 1 and dropout > 0 else 0.0
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_size,
            num_layers=self.num_gru_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.delta_head = nn.Linear(hidden_size, n_state)

    def _encode_h0(self, design: torch.Tensor) -> torch.Tensor:
        # GRU expects h0 shape (num_layers, batch, hidden)
        h = self.design_encoder(design)
        return h.unsqueeze(0).expand(self.num_gru_layers, -1, -1)

    def forward_teacher_forcing(
        self,
        states: torch.Tensor,
        exog: torch.Tensor,
        design: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict deltas for transitions 0..T-2 using true states at each step.

        Parameters
        ----------
        states : (B, T, n_state)  true states along profile
        exog : (B, T, n_exog)     exogenous inputs aligned with states
        design : (B, n_design)

        Returns
        -------
        deltas : (B, T-1, n_state)
        """
        if states.size(1) < 2:
            raise ValueError("Need sequence length >= 2 for delta prediction.")
        # Inputs at steps 0..T-2 (predict transition to next point)
        st_in = states[:, :-1, :]
        ex_in = exog[:, :-1, :]
        gru_in = torch.cat([st_in, ex_in], dim=-1)
        h0 = self._encode_h0(design)
        if lengths is not None:
            # One transition per step 0..L-2; ignore padded tail in batched profiles.
            t_lens = (lengths - 1).clamp(min=1).to(device="cpu", dtype=torch.long)
            packed = pack_padded_sequence(
                gru_in, t_lens, batch_first=True, enforce_sorted=False,
            )
            out_packed, _ = self.gru(packed, h0)
            out, _ = pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.gru(gru_in, h0)
        return self.delta_head(out)

    def forward_rollout(
        self,
        state0: torch.Tensor,
        exog: torch.Tensor,
        design: torch.Tensor,
        lengths: torch.Tensor | None = None,
        max_delta_scaled: float | None = None,
        max_state_scaled: float | None = None,
        max_rollout_steps: int | None = None,
        true_states: torch.Tensor | None = None,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive rollout from inlet state0 through T-1 transitions.

        Parameters
        ----------
        state0 : (B, n_state)
        exog : (B, T, n_exog)  exog[k] used when stepping from k -> k+1
        design : (B, n_design)
        true_states : optional (B, T, n_state) for scheduled sampling (train only)
        scheduled_sampling_prob : per-step prob. to replace rolled state with truth

        Returns
        -------
        states_pred : (B, T, n_state)  including inlet at index 0
        """
        batch, t_steps, _ = exog.shape
        device = exog.device
        if lengths is None:
            lengths = torch.full((batch,), t_steps, dtype=torch.long, device=device)
        use_sched = (
            self.training
            and true_states is not None
            and scheduled_sampling_prob > 0.0
        )
        h = self._encode_h0(design)
        cur = state0
        states_list = [cur]
        max_len = int(lengths.max().item())
        if max_rollout_steps is not None:
            max_len = min(max_len, int(max_rollout_steps) + 1)
            max_len = max(max_len, 2)
        for k in range(max_len - 1):
            if use_sched:
                mask_t = torch.rand(batch, device=device) < scheduled_sampling_prob
                cur = torch.where(mask_t.unsqueeze(1), true_states[:, k, :], cur)
            step_in = torch.cat([cur, exog[:, k, :]], dim=-1).unsqueeze(1)
            out, h = self.gru(step_in, h)
            delta = self.delta_head(out).squeeze(1)
            if max_delta_scaled is not None and max_delta_scaled > 0:
                delta = torch.clamp(delta, -max_delta_scaled, max_delta_scaled)
            next_cur = cur + delta
            if max_state_scaled is not None and max_state_scaled > 0:
                next_cur = torch.clamp(next_cur, -max_state_scaled, max_state_scaled)
            active = (k < (lengths - 1)).unsqueeze(1).to(dtype=cur.dtype)
            cur = active * next_cur + (1.0 - active) * cur
            states_list.append(cur)
        return torch.stack(states_list, dim=1)
