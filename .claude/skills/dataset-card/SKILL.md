---
name: dataset-card
description: Write or update a dataset card (README) for a scientific/ML dataset. Use when documenting columns, units, provenance, splits, license, and known biases before sharing or training.
license: MIT
---

# Dataset Card

A dataset card lets another researcher use the data without a tour.

## Collect

- **Purpose** — what question this dataset supports.
- **Provenance** — source simulations/experiments, dates, creators, processing pipeline version.
- **Files** — paths, formats, sizes, checksums if available.
- **Schema** — each column/field: name, dtype, unit, description, allowed values.
- **Splits** — how train/val/test were built; manifest location.
- **License / access** — who may use it and restrictions.
- **Known issues** — missing data, outliers, label noise, selection bias.

## Output structure

1. Overview (2–3 sentences)
2. Composition (what each file contains)
3. Schema table
4. Collection & processing
5. Splits and recommended use
6. Metrics baseline (optional)
7. Citation / contact
8. Changelog

See `references/card-template.md`. Pair with `split-audit` before publishing splits.
