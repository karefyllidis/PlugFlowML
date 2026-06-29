"""PFR physics residuals for PINN training (Main_8).

Two residual families:

1. Algebraic constraints — do not require Cantera; work with lumped species:
   - Ideal-gas EOS: p = ρ R T / M
   - Mass conservation: ρ u A = ṁ
   - Species sum: Σ Y_k = 1
   - Species non-negativity: Y_k ≥ 0

2. ODE constraints — use torch.autograd.grad on the z/L input:
   - Energy: dT/d(z/L) = L π D q / (ṁ cp)
     (simplified: wall heat only; reaction enthalpy captured through data loss)

3. Cantera ODE residuals — require full (unlumped) species composition:
   - dY_k/dz = W_k ω_k / (ρ u)
   - dT/dz = [π D q - Σ h_k ω_k W_k] / (ρ u cp)
   Gated by USE_CANTERA_RESIDUALS flag in the notebook (default False when using
   lumped species; enable when training on raw species data).

Usage:
    from src.physics import PFRResiduals, compute_algebraic_residuals, build_colloc_input

N. Karefyllidis 2026
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

R_UNIVERSAL = 8314.0  # J / (kmol · K)
PI = torch.pi


# ─── Differentiable scaler helpers ───────────────────────────────────────────

def inv_scale_y(y_scaled: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    """Inverse-transform target tensor; keeps gradient graph intact."""
    return y_scaled * y_std + y_mean


def inv_scale_X(X_scaled: torch.Tensor, X_mean: torch.Tensor, X_std: torch.Tensor) -> torch.Tensor:
    """Inverse-transform feature tensor; keeps gradient graph intact."""
    return X_scaled * X_std + X_mean


# ─── Algebraic residuals ─────────────────────────────────────────────────────

def compute_algebraic_residuals(
    y_pred_phys: torch.Tensor,
    X_phys: torch.Tensor,
    tgt_idx: Dict[str, int],
    feat_idx: Dict[str, int],
    species_indices: List[int],
    *,
    lambda_eos: float = 1.0,
    lambda_mass: float = 1.0,
    lambda_sum: float = 1.0,
    lambda_nonneg: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute four algebraic physics residuals on batches of physical predictions.

    Parameters
    ----------
    y_pred_phys : (N, n_targets) predicted targets in physical units.
    X_phys      : (N, n_features) feature inputs in physical units.
    tgt_idx     : dict mapping target name → column index in y_pred_phys.
    feat_idx    : dict mapping feature name → column index in X_phys.
    species_indices : list of column indices in y_pred_phys for Y_lump_* targets.

    Returns
    -------
    dict with keys 'eos', 'mass', 'species_sum', 'species_nonneg', 'total'.
    All values are scalar tensors (mean squared residuals).
    """
    T   = y_pred_phys[:, tgt_idx["temperature_K"]]
    p   = y_pred_phys[:, tgt_idx["pressure_Pa"]]
    rho = y_pred_phys[:, tgt_idx["density_kgm3"]]
    u   = y_pred_phys[:, tgt_idx["velocity_ms"]]
    M   = y_pred_phys[:, tgt_idx["mean_molecular_weight_kgkmol"]]

    D    = X_phys[:, feat_idx["reactor_diameter_m"]]
    mdot = X_phys[:, feat_idx["mass_flow_rate_kgps"]]

    # 1. EOS: p = ρ R T / M  (relative residual to avoid Pa² scale dominance)
    p_eos  = rho * R_UNIVERSAL * T / M.clamp(min=1.0)
    p_ref  = p.abs().clamp(min=1e3)
    r_eos  = ((p - p_eos) / p_ref).pow(2).mean()

    # 2. Mass conservation: ρ u A = ṁ
    A      = PI * (D * 0.5).pow(2)
    mdot_p = rho * u * A
    m_ref  = mdot.abs().clamp(min=1e-10)
    r_mass = ((mdot_p - mdot) / m_ref).pow(2).mean()

    # 3. Species sum: Σ Y_lump_k = 1
    if species_indices:
        Y = y_pred_phys[:, species_indices]
        r_sum = (Y.sum(dim=-1) - 1.0).pow(2).mean()
        r_neg = F.relu(-Y).pow(2).mean()
    else:
        r_sum = torch.tensor(0.0, device=y_pred_phys.device)
        r_neg = torch.tensor(0.0, device=y_pred_phys.device)

    total = lambda_eos * r_eos + lambda_mass * r_mass + lambda_sum * r_sum + lambda_nonneg * r_neg

    return {
        "eos":            r_eos,
        "mass":           r_mass,
        "species_sum":    r_sum,
        "species_nonneg": r_neg,
        "total":          total,
    }


# ─── ODE residual (energy, via autograd) ─────────────────────────────────────

def compute_energy_ode_residual(
    dT_phys_dz_L: torch.Tensor,
    y_pred_phys: torch.Tensor,
    X_phys: torch.Tensor,
    tgt_idx: Dict[str, int],
    feat_idx: Dict[str, int],
) -> torch.Tensor:
    """Energy ODE residual: dT/d(z/L) - L π D q / (ṁ cp).

    Simplified energy balance (wall heat only; chemistry captured by data loss).
    All inputs are in physical units; dT_phys_dz_L is the autograd-computed
    gradient d(T_pred_phys) / d(z/L_phys) for the collocation batch.

    Returns mean squared residual, scalar tensor.
    """
    cp   = y_pred_phys[:, tgt_idx["heat_capacity_cp_JkgK"]].clamp(min=100.0)
    L    = X_phys[:, feat_idx["reactor_length_m"]]
    D    = X_phys[:, feat_idx["reactor_diameter_m"]]
    q    = X_phys[:, feat_idx["heat_flux_Wm2"]]
    mdot = X_phys[:, feat_idx["mass_flow_rate_kgps"]].abs().clamp(min=1e-10)

    rhs = L * PI * D * q / (mdot * cp)        # [K] per unit z/L
    T_scale = rhs.abs().detach().clamp(min=1.0)
    residual = (dT_phys_dz_L - rhs) / T_scale
    return residual.pow(2).mean()


# ─── Collocation input builder ────────────────────────────────────────────────

def build_colloc_input(
    X_inlet_s: torch.Tensor,
    z_L_phys: torch.Tensor,
    feature_cols: List[str],
    X_mean: torch.Tensor,
    X_std: torch.Tensor,
) -> torch.Tensor:
    """Build a scaled PINN input tensor with random z/L substituted in.

    Parameters
    ----------
    X_inlet_s  : (N, n_feat) scaled inlet features from training data.
                 The z/L and z_position columns will be replaced.
    z_L_phys   : (N, 1) physical z/L values in [0, 1], with requires_grad=True.
    feature_cols: ordered list of feature column names.
    X_mean, X_std: scaler stats as float32 tensors, shape (n_feat,).

    Returns
    -------
    (N, n_feat) scaled input tensor with z_L_phys embedded (grad-enabled).
    """
    z_L_idx   = feature_cols.index("relative_position") if "relative_position" in feature_cols else None
    z_pos_idx = feature_cols.index("z_position_m")      if "z_position_m"       in feature_cols else None
    L_idx     = feature_cols.index("reactor_length_m")  if "reactor_length_m"   in feature_cols else None

    # Scale z/L: differentiable w.r.t. z_L_phys
    z_L_s = (z_L_phys - X_mean[z_L_idx]) / X_std[z_L_idx]  # (N, 1)

    # Build z_position_m scaled (if present)
    if z_pos_idx is not None and L_idx is not None:
        L_phys   = X_inlet_s[:, L_idx:L_idx+1] * X_std[L_idx] + X_mean[L_idx]
        z_pos_phys = L_phys * z_L_phys                   # z_pos_m = L * (z/L)
        z_pos_s  = (z_pos_phys - X_mean[z_pos_idx]) / X_std[z_pos_idx]
    else:
        z_pos_s = None

    # Reconstruct column by column to keep grad flow through z_L_phys
    cols = []
    for i, _name in enumerate(feature_cols):
        if i == z_L_idx:
            cols.append(z_L_s)
        elif i == z_pos_idx and z_pos_s is not None:
            cols.append(z_pos_s)
        else:
            cols.append(X_inlet_s[:, i:i+1])

    return torch.cat(cols, dim=1)


# ─── Cantera ODE wrapper ──────────────────────────────────────────────────────

class PFRResiduals:
    """Cantera oracle for PFR ODE RHS evaluation (full species).

    Used when USE_CANTERA_RESIDUALS=True in Main_8 and the PINN is trained on
    raw (unlumped) species mass fractions. For lumped species, use
    compute_algebraic_residuals instead.

    Parameters
    ----------
    mechanism_file : path to Cantera YAML mechanism file.
    species_names  : ordered list of species names matching the Y_k targets
                     predicted by the model. Must be valid species in the mechanism.
    batch_size     : Cantera loop batch size (default 64).
    """

    def __init__(
        self,
        mechanism_file: str | Path,
        species_names: List[str],
        batch_size: int = 64,
    ):
        try:
            import cantera as ct
        except ImportError as exc:
            raise ImportError(
                "Cantera is required for PFRResiduals. "
                "Install with: pip install cantera>=3.1.0"
            ) from exc

        self.mechanism_file = str(mechanism_file)
        self.species_names  = species_names
        self.batch_size     = batch_size
        self._ct = ct

        gas = ct.Solution(self.mechanism_file)
        self.W_k = np.array([gas.molecular_weights[gas.species_index(s)] for s in species_names])
        self._n_species = len(species_names)

    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        T_batch: np.ndarray,
        p_batch: np.ndarray,
        Y_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Cantera at (T, p, Y_k) states.

        Parameters
        ----------
        T_batch : (N,) temperatures [K]
        p_batch : (N,) pressures   [Pa]
        Y_batch : (N, n_species) mass fractions for self.species_names
                  (rows will be clipped to [0,1] and renormalised)

        Returns
        -------
        omega_k : (N, n_species) net molar production rates [kmol / m³ / s]
        rho     : (N,) density [kg/m³]
        cp      : (N,) specific heat at const pressure [J/(kg·K)]
        """
        ct    = self._ct
        gas   = ct.Solution(self.mechanism_file)
        N     = T_batch.shape[0]
        omega = np.zeros((N, self._n_species), dtype=np.float64)
        rho   = np.zeros(N,                   dtype=np.float64)
        cp    = np.zeros(N,                   dtype=np.float64)

        for i in range(N):
            Y_row = np.clip(Y_batch[i], 0.0, 1.0)
            s     = Y_row.sum()
            if s > 0:
                Y_row /= s

            # Build full composition dict (zeros for non-listed species)
            comp = {name: float(Y_row[j]) for j, name in enumerate(self.species_names)}
            try:
                gas.TPY = float(T_batch[i]), float(p_batch[i]), comp
                species_idx = [gas.species_index(s) for s in self.species_names]
                omega[i] = gas.net_production_rates[species_idx]
                rho[i]   = gas.density
                cp[i]    = gas.cp_mass
            except Exception:
                # Cantera failed at this state — leave zero (residual contribution = 0)
                pass

        return omega, rho, cp

    # ------------------------------------------------------------------
    def ode_residuals(
        self,
        T_pred: torch.Tensor,
        p_pred: torch.Tensor,
        Y_pred: torch.Tensor,
        dY_dz:  torch.Tensor,
        u_pred: torch.Tensor,
        reactor_length: torch.Tensor,
    ) -> torch.Tensor:
        """Full ODE residuals: dY_k/d(z/L) - L W_k omega_k / (rho u).

        Parameters
        ----------
        T_pred, p_pred, Y_pred : physical predictions (detached OK for Cantera call).
        dY_dz   : (N, n_species) autograd-computed d(Y_k)/d(z/L) — MUST stay in graph.
        u_pred  : (N,) predicted velocity [m/s].
        reactor_length : (N,) reactor lengths [m].

        Returns mean squared normalised residual (scalar tensor in graph).
        """
        # Cantera call on detached numpy arrays
        T_np  = T_pred.detach().cpu().numpy()
        p_np  = p_pred.detach().cpu().numpy()
        Y_np  = Y_pred.detach().cpu().numpy()

        omega_np, rho_np, _cp_np = self.evaluate_batch(T_np, p_np, Y_np)

        dev = dY_dz.device
        W_t     = torch.tensor(self.W_k,   dtype=torch.float32, device=dev)
        omega_t = torch.tensor(omega_np,   dtype=torch.float32, device=dev)
        rho_t   = torch.tensor(rho_np,     dtype=torch.float32, device=dev).unsqueeze(1)

        u_safe = u_pred.unsqueeze(1).clamp(min=0.01)          # [N, 1]
        L_exp  = reactor_length.unsqueeze(1)                   # [N, 1]

        # RHS: L W_k omega_k / (rho u)
        rhs = L_exp * W_t * omega_t / (rho_t * u_safe)        # [N, n_species]

        # Normalise by Y_k scale to balance large/small fractions
        Y_ref = Y_pred.abs().clamp(min=1e-4)
        res   = (dY_dz - rhs) / Y_ref
        return res.pow(2).mean()
