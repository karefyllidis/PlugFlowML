"""Rollout validation loss and non-finite diagnostics for Main_8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.ml.sequence_dataset import masked_rollout_mse


@dataclass(frozen=True)
class FirstNonFiniteReport:
    run_id: int | None
    batch_index: int
    axial_index: int
    stage: str
    variable: str | None
    value: float
    message: str

    def __str__(self) -> str:
        rid = "?" if self.run_id is None else str(self.run_id)
        var = self.variable or "—"
        return (
            f"run_id={rid} batch={self.batch_index} axial={self.axial_index} "
            f"stage={self.stage} variable={var} value={self.value:.6g} | {self.message}"
        )


def _check_tensor_finite(
    t: torch.Tensor,
    stage: str,
    *,
    batch_index: int,
    axial_index: int,
    run_id: int | None,
    state_cols: list[str] | None,
) -> FirstNonFiniteReport | None:
    if torch.isfinite(t).all():
        return None
    bad = ~torch.isfinite(t)
    if t.dim() >= 2:
        bi, ai = int(bad.any(dim=-1).nonzero(as_tuple=False)[0][0]), int(
            bad[0].nonzero(as_tuple=False)[0][0]
        )
    else:
        bi, ai = batch_index, axial_index
    flat_idx = int(bad.reshape(-1).nonzero(as_tuple=False)[0][0])
    var = None
    if state_cols and t.dim() >= 2 and t.size(-1) == len(state_cols):
        j = flat_idx % len(state_cols)
        var = state_cols[j]
    val = float(t.reshape(-1)[flat_idx].detach().cpu())
    return FirstNonFiniteReport(
        run_id=run_id,
        batch_index=bi,
        axial_index=ai,
        stage=stage,
        variable=var,
        value=val,
        message="non-finite tensor",
    )


@torch.no_grad()
def forward_rollout_with_diagnostics(
    model,
    state0: torch.Tensor,
    exog: torch.Tensor,
    design: torch.Tensor,
    lengths: torch.Tensor,
    *,
    max_delta_scaled: float | None,
    max_state_scaled: float | None,
    max_rollout_steps: int | None = None,
    state_cols: list[str] | None = None,
    run_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, FirstNonFiniteReport | None]:
    """Stepwise rollout mirroring ProfileGRU.forward_rollout with finite checks."""
    batch, t_steps, _ = exog.shape
    device = exog.device
    h = model._encode_h0(design)
    cur = state0
    states_list = [cur]
    max_len = int(lengths.max().item())
    if max_rollout_steps is not None:
        max_len = min(max_len, int(max_rollout_steps) + 1)
        max_len = max(max_len, 2)

    for k in range(max_len - 1):
        rid = int(run_ids[0].item()) if run_ids is not None and run_ids.numel() else None
        hit = _check_tensor_finite(
            cur, "scaled_state", batch_index=0, axial_index=k, run_id=rid, state_cols=state_cols,
        )
        if hit is not None:
            return torch.stack(states_list, dim=1), hit

        step_in = torch.cat([cur, exog[:, k, :]], dim=-1).unsqueeze(1)
        hit = _check_tensor_finite(
            step_in, "gru_input", batch_index=0, axial_index=k, run_id=rid, state_cols=None,
        )
        if hit is not None:
            return torch.stack(states_list, dim=1), hit

        out, h = model.gru(step_in, h)
        delta = model.delta_head(out).squeeze(1)
        if max_delta_scaled is not None and max_delta_scaled > 0:
            delta = torch.clamp(delta, -max_delta_scaled, max_delta_scaled)
        hit = _check_tensor_finite(
            delta, "predicted_delta", batch_index=0, axial_index=k, run_id=rid, state_cols=state_cols,
        )
        if hit is not None:
            return torch.stack(states_list, dim=1), hit

        next_cur = cur + delta
        if max_state_scaled is not None and max_state_scaled > 0:
            next_cur = torch.clamp(next_cur, -max_state_scaled, max_state_scaled)
        active = (k < (lengths - 1)).unsqueeze(1).to(dtype=cur.dtype)
        cur = active * next_cur + (1.0 - active) * cur
        states_list.append(cur)

    pred = torch.stack(states_list, dim=1)
    return pred, None


@torch.no_grad()
def eval_rollout_loss(
    model,
    loader,
    device: torch.device,
    *,
    max_delta_scaled: float | None,
    max_state_scaled: float | None,
    max_rollout_steps: int | None = None,
    state_cols: list[str] | None = None,
    log_first_bad: bool = True,
) -> tuple[float, FirstNonFiniteReport | None]:
    """Finite masked rollout MSE; returns (loss, first_nonfinite_report)."""
    model.eval()
    per_run: list[torch.Tensor] = []
    first_bad: FirstNonFiniteReport | None = None

    for batch in loader:
        st = batch["state"].to(device)
        ex = batch["exog"].to(device)
        des = batch["design"].to(device)
        lengths = batch["lengths"].to(device)
        run_ids = batch["run_id"]

        pred = model.forward_rollout(
            st[:, 0, :],
            ex,
            des,
            lengths=lengths,
            max_delta_scaled=max_delta_scaled,
            max_state_scaled=max_state_scaled,
            max_rollout_steps=max_rollout_steps,
        )
        if not torch.isfinite(pred).all() and first_bad is None:
            _, hit = forward_rollout_with_diagnostics(
                model,
                st[:, 0, :],
                ex,
                des,
                lengths,
                max_delta_scaled=max_delta_scaled,
                max_state_scaled=max_state_scaled,
                max_rollout_steps=max_rollout_steps,
                state_cols=state_cols,
                run_ids=run_ids,
            )
            first_bad = hit
            if log_first_bad and hit is not None:
                print(f"[WARN] Rollout non-finite: {hit}")

        if pred.size(1) < st.size(1):
            st_cmp = st[:, : pred.size(1), :]
        else:
            st_cmp = st
            pred = pred[:, : st.size(1), :]

        loss_vec = masked_rollout_mse(pred, st_cmp, lengths.cpu(), reduction="none")
        for i, lv in enumerate(loss_vec.tolist()):
            if np.isfinite(lv):
                per_run.append(torch.tensor(lv, device=device))
            elif first_bad is None:
                rid = int(run_ids[i].item())
                first_bad = FirstNonFiniteReport(
                    run_id=rid,
                    batch_index=i,
                    axial_index=-1,
                    stage="rollout_mse",
                    variable=None,
                    value=float("nan"),
                    message="masked_rollout_mse non-finite",
                )

    if not per_run:
        return float("nan"), first_bad
    out = float(torch.stack(per_run).mean().item())
    if not np.isfinite(out):
        return float("nan"), first_bad
    return out, first_bad
