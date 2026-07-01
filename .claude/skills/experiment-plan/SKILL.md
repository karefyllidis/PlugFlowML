---
name: experiment-plan
description: Design an experiment or study. Use when the user wants to plan experiments, ablations, benchmarks, or a study design — defining hypotheses, variables, metrics, controls, sample size, and expected failure modes before running anything.
license: MIT
---

# Experiment Plan

Design the experiment before running it, so the results will actually answer the question.

## Define up front

- **Question / hypothesis:** a falsifiable statement. What result would confirm vs. refute it?
- **Variables:** independent (what you vary), dependent (what you measure), and controlled (held fixed). Name confounders.
- **Conditions & ablations:** the comparison set — baseline, treatment(s), and ablations that isolate each factor.
- **Metrics:** primary metric tied to the question, plus secondary/diagnostic metrics. Define exactly how each is computed.
- **Controls:** baselines, control groups, randomization, seeds; what makes the comparison fair.
- **Sample size / runs:** how many seeds/replicates to distinguish signal from noise; note variance expectations.
- **Protocol:** the exact procedure, in enough detail to reproduce, including data splits.
- **Expected outcomes & failure modes:** predict the plausible results and what each would mean; list ways the experiment could mislead (leakage, confounds, underpowered, metric gaming).
- **Decision rule:** what result leads to what next action.

## Discipline

- One question per experiment where possible; don't entangle variables.
- Pre-register the analysis: decide how you'll interpret results before seeing them.
- Make it reproducible: seeds, versions, data provenance.

See `references/plan-template.md`. Pairs with `results-interpretation` and `modeling-assumption-review`.
