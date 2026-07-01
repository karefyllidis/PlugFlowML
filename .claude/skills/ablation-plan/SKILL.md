---
name: ablation-plan
description: Plan a scientific ablation study (one-factor-at-a-time or structured grid). Use before spending GPU hours to isolate which hypothesis matters.
license: MIT
---

# Ablation Plan

## Define

- **Baseline** run (frozen config + data version).
- **Hypotheses** — each maps to one config change (or small coupled set with justification).
- **Primary metric** on **val** split; holdout untouched until final confirmation.
- **Budget** — max runs, seeds per condition.

## Design

- Prefer **one change per run** unless interaction is the research question.
- Fix **seeds** and data version across compared runs.
- Order runs: cheap smoke → full training for promising deltas only.
- Pre-register **stop rule** (e.g. no improvement on val for 3 conditions).

## Deliverable

Table: `run_id | change from baseline | expected signal | priority | est. cost`

After results, hand off to `run-comparison`.
