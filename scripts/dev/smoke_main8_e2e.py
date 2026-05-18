#!/usr/bin/env python3
"""Smoke Main_8 GRU path: synthetic pkl, 3 epochs, anchor + metrics checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.dataframe_pickle import save_pickle_portable
from src.ml.profile_gru import ProfileGRU
from src.ml.profile_rollout_stability import clamp_state_physical, rollout_minmax_table
from src.ml.sequence_dataset import (
    ProfileSequenceDataset,
    SequenceColumnSpec,
    build_sequence_arrays,
    collate_profile_sequences,
    masked_delta_mse,
    run_id_series,
)
from src.utils.profile_predictions import anchor_inlet_profile_predictions
from src.utils.residence_time import (
    add_residence_time_columns,
    compute_residence_time_for_run,
    validate_residence_time_on_df,
)


def _make_synthetic(n_runs: int = 12, n_steps: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run in range(n_runs):
        z = np.linspace(0, 10.0, n_steps)
        u = 40.0 + 2.0 * z + 0.5 * run
        t = 300.0 + 50.0 * z / z[-1]
        for i in range(n_steps):
            rows.append({
                "initial_temperature_K": 850.0 + run,
                "initial_pressure_Pa": 2e5,
                "reactor_length_m": 10.0,
                "reactor_diameter_m": 0.03,
                "mass_flow_rate_kgps": 0.07,
                "heat_flux_Wm2": 150000.0,
                "reactant_type": "n-Hexane",
                "z_position_m": z[i],
                "relative_position": z[i] / z[-1],
                "temperature_K": t[i],
                "pressure_Pa": 2e5 - 1e3 * z[i],
                "density_kgm3": 0.4 + 0.01 * i,
                "velocity_ms": u[i],
                "Y_lump_chem_feedstock": max(0.1, 1.0 - 0.03 * i),
                "Y_lump_chem_olefins": 0.02 * i,
            })
    df_f = pd.DataFrame(rows)
    df_t = df_f[
        ["temperature_K", "pressure_Pa", "density_kgm3", "velocity_ms",
         "Y_lump_chem_feedstock", "Y_lump_chem_olefins"]
    ].copy()
    return df_f, df_t


def main() -> int:
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    pkl_path = proc / "features_targets_smoke_main8.pkl"
    df_features, df_target = _make_synthetic()
    save_pickle_portable(
        {"df_features": df_features, "df_target": df_target},
        pkl_path,
    )
    print(f"Wrote {pkl_path}")

    design_cols = [
        "initial_temperature_K", "initial_pressure_Pa",
        "reactor_length_m", "reactor_diameter_m",
        "mass_flow_rate_kgps", "heat_flux_Wm2", "reactant_type",
    ]
    run_cols = design_cols.copy()
    state_cols = list(df_target.columns)
    df = df_features[
        ["z_position_m", "relative_position"] + design_cols
    ].join(df_target, how="inner")
    le = LabelEncoder()
    df["reactant_type"] = le.fit_transform(df["reactant_type"].astype(str))
    df = add_residence_time_columns(df, run_cols=run_cols)
    assert not validate_residence_time_on_df(df, run_cols=run_cols)

    z = np.linspace(0, 10, 25)
    u = 40 + 2 * z
    rt = compute_residence_time_for_run(z, u)
    assert rt["tau_s"][0] == 0 and np.all(np.diff(rt["tau_s"]) > 0)

    spec = SequenceColumnSpec(
        run_cols=run_cols,
        design_cols=design_cols,
        exog_cols=["relative_position", "relative_tau", "log_dt_s"],
        state_cols=state_cols,
    )
    full_run_id = run_id_series(df, run_cols)
    unique_runs = np.array(sorted(pd.unique(full_run_id)))
    trainval, test_runs = train_test_split(unique_runs, test_size=0.2, random_state=42)
    train_runs, val_runs = train_test_split(trainval, test_size=0.15, random_state=42)

    def mask(runs):
        return full_run_id.isin(runs).to_numpy()

    train_mask, val_mask, test_mask = mask(train_runs), mask(val_runs), mask(test_runs)
    scaler_design = StandardScaler().fit(
        df.loc[train_mask].groupby(run_cols, dropna=False)[design_cols].first()
    )
    scaler_state = StandardScaler().fit(df.loc[train_mask, state_cols])
    scaler_exog = StandardScaler().fit(df.loc[train_mask, spec.exog_cols])

    def seq_dict(run_mask):
        arrays = build_sequence_arrays(df.loc[run_mask], spec, full_run_id[run_mask].to_numpy())
        for seq in arrays.values():
            seq["design"] = scaler_design.transform(seq["design"].reshape(1, -1)).ravel()
            seq["state"] = scaler_state.transform(seq["state"])
            seq["exog"] = scaler_exog.transform(seq["exog"])
        return arrays

    train_loader = DataLoader(
        ProfileSequenceDataset(seq_dict(train_mask)),
        batch_size=4,
        shuffle=True,
        collate_fn=collate_profile_sequences,
    )
    device = torch.device("cpu")
    model = ProfileGRU(
        len(state_cols),
        len(spec.exog_cols),
        len(design_cols),
        hidden_size=32,
        design_hidden=16,
        num_gru_layers=1,
        dropout=0.0,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        for batch in train_loader:
            st = batch["state"].to(device)
            ex = batch["exog"].to(device)
            des = batch["design"].to(device)
            lengths = batch["lengths"]
            pred_d = model.forward_teacher_forcing(st, ex, des, lengths=lengths.to(device))
            loss = masked_delta_mse(pred_d, st, lengths)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # rollout + anchor
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in DataLoader(
            ProfileSequenceDataset(seq_dict(test_mask)),
            batch_size=4,
            collate_fn=collate_profile_sequences,
        ):
            for i in range(batch["state"].size(0)):
                L = int(batch["lengths"][i])
                st_i = batch["state"][i, :L].numpy()
                ex_i = batch["exog"][i, :L].unsqueeze(0).to(device)
                des_i = batch["design"][i].unsqueeze(0).to(device)
                pred = model.forward_rollout(
                    torch.tensor(st_i[0], device=device).unsqueeze(0),
                    ex_i,
                    des_i,
                    lengths=torch.tensor([L], device=device),
                    max_delta_scaled=3.0,
                ).cpu().numpy()[0]
                true_p = scaler_state.inverse_transform(st_i)
                pred_p = clamp_state_physical(
                    scaler_state.inverse_transform(pred),
                    state_cols,
                    [c for c in state_cols if c.startswith("Y_")],
                )
                rows.append({
                    "run_id": int(batch["run_id"][i]),
                    "true": true_p,
                    "pred": pred_p,
                })

    all_true = np.concatenate([r["true"] for r in rows])
    all_pred = np.concatenate([r["pred"] for r in rows])
    run_ids_out = np.concatenate([np.full(r["true"].shape[0], r["run_id"]) for r in rows])
    rel_pos = np.concatenate([
        df.loc[full_run_id == r["run_id"]].sort_values("z_position_m")["relative_position"].to_numpy()
        for r in rows
    ])
    all_pred, n_in = anchor_inlet_profile_predictions(
        all_pred, all_true, run_ids_out, rel_pos,
    )
    assert n_in > 0, "expected inlet rows anchored"
    np.testing.assert_array_equal(all_pred[rel_pos <= rel_pos.min() + 1e-12], all_true[rel_pos <= rel_pos.min() + 1e-12])

    mm = rollout_minmax_table(rows, state_cols)
    metrics = {
        "n_anchored": int(n_in),
        "n_test_rows": int(len(all_true)),
        "pred_T_max": float(mm.loc[mm["variable"] == "temperature_K", "pred_max"].iloc[0]),
    }
    print("smoke_main8_e2e OK:", json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
