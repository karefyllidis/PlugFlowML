# Model card: Species separation, grouping, and lumping (HydrAI)

This document describes the **methodology** used in HydrAI to reduce hundreds of tracked species to a smaller set of **mass-fraction-based** features and targets for ML. It is **not** a surrogate model card for a trained predictor; it documents the **preprocessing taxonomy and aggregation rules**.

**Source of truth (implementation):** `notebooks/Main_3_data_exploration_feature_engineering.ipynb` (functions `_extract_carbon_count`, `_classify_species_chemistry`, `organize_data_columns`, `_build_lumped_species_columns`, export merge).

---

## Summary

| Item | Description |
|------|-------------|
| **Goal** | Dimensionality reduction: many `Y_<species>` columns → fewer interpretable groups for EDA and ML. |
| **Quantity preserved** | **Mass fraction:** lump values are **sums** of constituent species mass fractions within a group (same row / same axial location). |
| **Mole fractions** | **`X_*` are not ML targets.** Raw training pickles may still contain `X_*`; Main_3 `df_target` and lump export use **`Y_*` only.** |
| **Two schemes** | (1) **Carbon-number lumps** — `C1`, `C2`, …, plus **`inert`**. (2) **Chemistry / process-role lumps** — e.g. `olefins`, `coke_precursors`, `hydrogen`. |
| **Export modes** | `individual` (keep all `Y_*`) vs `lumped_chemistry` / `lumped_carbon` (replace with **`Y_lump_*`** columns). |

---

## Motivation

- Detailed steam-cracking mechanisms can list **150+ species** as separate mass-fraction columns.
- Predicting every minor species often **overfits** and is weakly constrained by data.
- Many engineering questions concern **aggregate yields** (total olefins, aromatics, precursors to coking) and **carbon routing** (distribution across C1, C2, …).
- Fewer outputs improve **training cost**, **stability**, and **interpretability** of tree and other multi-output models.

---

## Inputs and naming convention

- Species mass fractions in tabular data use the prefix **`Y_`** (e.g. `Y_C2H4(8)` after the mechanism naming convention).
- Parsing strips optional leading `Y_` / `X_` for classification helpers; **lumping for ML uses `Y_*` columns only.**
- The **base name** used for rules is the substring before `(` if present (handles suffixes like grid indices).

---

## Scheme A: Carbon-number grouping (`IF_SEPARATE_SPECIES_BY_CARBON`)

**Function:** `_extract_carbon_count(species_name) -> int | None`

- Returns a **carbon count** for hydrocarbon-like names; returns **`None` for inerts** (then routed to **`species_mass_fractions_inert`**).
- **Inert set (carbon lump):** `Water`, `Ar`, `He`, `Ne`, `N2`, `H2`, `H`, `S` → **`None`** (inert bucket).
- **Special carbon mapping:** `Benzene` → 6, `Toluene` → 7, `Styrene` → 8.
- **Patterns:** `C{n}H…` → `n`; names starting with `CH…` → 1; `CC`, `CCC`, `C#C…` → carbon count from string structure; otherwise `None`.

**Category keys in `data_categories`:**

- `species_mass_fractions` — all `Y_*` columns.
- When carbon split is on: `species_mass_fractions_C1`, `species_mass_fractions_C2`, … and **`species_mass_fractions_inert`** (plus other `species_mass_fractions_*` split keys as produced).

**Exported lump columns (`EXPORT_SPECIES_AS = 'lumped_carbon'`):**

- One column per non-root split key, e.g. **`Y_lump_carbon_C2`**, **`Y_lump_carbon_inert`**, etc.:  
  **`Y_lump_carbon_<suffix> = sum(Y_i)`** over all `Y_*` in that carbon group.

---

## Scheme B: Chemistry / process-role grouping (`IF_CATEGORIZE_BY_CHEMISTRY`)

**Function:** `_classify_species_chemistry(species_name) -> str`

**Ordered rule list (first match wins):**

1. **`hydrogen`** — `H2` only (explicitly **not** grouped with diluent).
2. **`diluent`** — `Water`, `Ar`, `He`, `Ne`, `N2`, `H`.
3. **`feedstock`** — `C6H14` (n-hexane in the project’s default feed labeling).
4. **`olefins`** — `C2H4`, `C3H6`, `C4H6`, `C4H8`.
5. **`aromatics`** — `Benzene`, `Toluene`, `Styrene`, `C8H10`.
6. **`paraffins`** — `CH4`, `CC`, `CCC` (methane, ethane, propane naming in mechanism).
7. **`coke_precursors`** — includes:
   - Acetylene / alkyne motifs: `C2H2`, names starting with `C#C`;
   - `C5H6`, `C5H5`, `C6H6`, `C6H8`, `C3H4`, `C4H4`, `C4H5`;
   - **Heavy unsaturates:** for `C{n}H{m}` with **n ≥ 6** and **H/C < 1.5**.
8. **`radicals`** — fixed set including e.g. `CH3`, `C2H3`, `C2H5`, …, `C3H3` (see notebook for full list).
9. **`other`** — `S` and everything not caught above.

**Category keys:** `species_Y_<role>` (e.g. `species_Y_olefins`).

**Exported lump columns (`EXPORT_SPECIES_AS = 'lumped_chemistry'`):**

- **`Y_lump_chem_<role> = sum(Y_i)`** over all `Y_*` in `species_Y_<role>`.

---

## Using both schemes

- **Carbon** and **chemistry** flags can both be **True** for **EDA** (extra keys in `data_categories`).
- **`EXPORT_SPECIES_AS`** must be **either** `lumped_carbon` **or** `lumped_chemistry` when you want a lumped pickle — not both in one export path. Pick the scheme that matches the question (carbon balance vs product/coking roles).

---

## Aggregation: sum of mass fractions

For each lump (group) \(G\) at a given row (spatial location):

\[
Y_{\text{lump},G} = \sum_{k \in G} Y_k
\]

where each \(Y_k\) is a species **mass fraction**. This is appropriate because mass fractions of components **add linearly** within a mixture subset at a point.

**Caveats:**

- Lumps are **not** new pseudo-species with a single molecular weight; they are **reporting aggregates**.
- If the underlying mechanism or column set changes (different species names), **group membership** changes — **retrain** or **rebuild** lumps.

---

## Configuration flags (Main_3)

| Flag | Role |
|------|------|
| `IF_SEPARATE_SPECIES_BY_CARBON` | Enables carbon-based keys in `organize_data_columns`. |
| `IF_CATEGORIZE_BY_CHEMISTRY` | Enables `species_Y_*` keys. |
| `EXPORT_SPECIES_AS` | `'individual'` \| `'lumped_chemistry'` \| `'lumped_carbon'`. Lumped modes require the matching flag; otherwise the notebook warns and falls back to individual `Y_*`. |

**Artifact:** `data/processed/features_targets_<run_stem>.pkl` containing `df_features` and `df_target`.

---

## Downstream ML (`Main_4`)

- **`Main_4_train_and_evaluate_tree_models.ipynb`** loads the pickle and detects **`Y_lump_chem_*`** vs **`Y_lump_carbon_*`** vs individual **`Y_*`** to build chemistry-group diagnostics and training targets.
- **Mass fraction only:** species targets are **`Y_*` or `Y_lump_*`**; mole fractions are out of scope for this pathway.

---

## Limitations and known gaps

- **Heuristic taxonomy:** Chemistry classes use **name-based rules**, not graph-theoretic structure from the mechanism file. Renamed species or different feeds may require rule updates.
- **Feedstock scope:** `feedstock` is tied to **`C6H14`** in the current rule set; other feeds need explicit extensions.
- **`other` bucket:** Species that miss all rules fall into **`other`**; monitor its size when changing mechanisms.
- **No uncertainty:** Lumps inherit model and data uncertainty; they do not add statistical error bars by themselves.

---

## Related documentation

- `docs/ML_CONFIG_GUIDE.md` — notebook flags and export behavior.
- `docs/TRAINING_DATA_GENERATION_PROTOCOL_MODEL_CARD.md` — how consolidated training tables are produced before Main_3.

---

## Maintainer

HydrAI project maintainers; keep this card synchronized when editing `_classify_species_chemistry` or `_extract_carbon_count` in Main_3.
