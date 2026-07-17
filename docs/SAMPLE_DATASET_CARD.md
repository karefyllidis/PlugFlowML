# Sample Dataset Card — `sample150`

The dataset every hands-on lesson (3–10) of the HydrAI course uses. Small
enough to download in seconds, real enough to teach honest surrogate modelling.

## Files (GitHub Release `sample-data-v1`)

| File | Contents | Destination |
|---|---|---|
| `training_data_complete_sample150.pkl` | Raw axial profiles, 27 661 rows × 328 cols (portable pandas pickle) | `data/training/` |
| `metadata_sample150.json` | Generation metadata (per-task provenance, parameter ranges) | `data/training/` |
| `training_data_sample150.parquet` | Same table as the raw pickle, Parquet/zstd — for use outside this pipeline | anywhere |
| `features_targets_training_data_complete_sample150.pkl` | Lesson 3's processed export (lumped-chemistry features/targets) consumed by Lessons 4–10 | `data/processed/` |
| `models_pretrained_sample.zip` | Pretrained artifacts (SimpleNN, PINN, SR equations) so Lessons 8–10 run standalone | `models/` |

Each lesson's bootstrap cell downloads exactly what it needs; the table above
is for manual setup.

## Provenance

- **Generator:** Cantera 3.2 plug-flow reactor with wall heat flux
  (`src/cantera/pfr_simulator.py`), n-hexane feed, detailed kinetic mechanism
  (153 species / 2 146 reactions). The mechanism itself is **not
  redistributable**; this dataset is simulation *output* and is released under
  the licence below.
- **Sampling:** Latin Hypercube over 6 inlet parameters, seed 7
  (`configs/ml/main2_sample_dataset_config.json` — the exact config, committed).
  The seed differs from the research campaign's, so this is an independent
  draw from the same domain, not a subset.
- **Campaign:** 150 simulations, 138 successful (92.0 % — failures are stiff
  integrations that did not complete, the same QC reality as the full
  campaign), 200 axial steps per run → 27 661 rows.
- **Generated:** 2026-07-17, single 20-core workstation, ~10 min wall clock
  (8 parallel workers).

## Schema (raw table)

One row = one axial position of one simulation.

- **Design columns (constant within a run):** `reactant_type`,
  `initial_temperature_K` (800–900), `initial_pressure_Pa` (1.5–3.5 bar),
  `reactor_length_m` (10–15), `reactor_diameter_m` (0.025–0.040),
  `mass_flow_rate_kgps` (0.05–0.10), `heat_flux_Wm2` (100–250 k).
- **Axial coordinate:** `z_position_m` (plus derived `relative_position` z/L
  after Lesson 3).
- **State/thermo:** temperature, pressure, density, velocity, residence time,
  enthalpy, etc. along the reactor.
- **Composition:** mass fractions `Y_<species>` for all 153 species (and mole
  fractions `X_*`, which the ML pipeline deliberately never uses — see course
  data rules). Lesson 3 lumps these into 9 chemistry groups
  (`Y_lump_hydrogen`, `Y_lump_olefins`, `Y_lump_paraffins`, …).

The processed export keeps 6 design columns + `relative_position` as features
and 9 state/thermo + 9 lumped mass-fraction targets.

## Splits

No split is baked into the files. Lessons split **at run level** (80/20,
`random_state=42`) so that no axial profile leaks across the train/test
boundary — a core teaching point of the course (Lesson 6).

## Known limitations & appropriate use

- 138 runs is deliberately small: models trained on it are *teaching-quality*,
  not research-quality. The research campaign used the same domain at 46 000
  runs.
- Single feedstock (n-hexane), fixed mechanism, fixed heat-flux shape —
  surrogates trained here do not generalise beyond the sampled domain, and the
  course treats that as a feature (Lesson 10 discusses extrapolation risk).
- Failed-run bias: the 12 failed simulations are absent; if failures cluster
  in a domain corner, that corner is under-represented (Lesson 2 discusses
  this).

## Licence & citation

Data: released with the repository (MIT-licensed project; data usable for any
purpose with attribution appreciated). Cite via the repository's
`CITATION.cff`.
