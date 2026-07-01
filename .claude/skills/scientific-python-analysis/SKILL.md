---
name: scientific-python-analysis
description: Perform data analysis in scientific Python. Use when the user wants to explore, clean, analyze, or compute over a dataset with numpy/pandas/scipy — EDA, statistics, transformations, and correctness-focused numerical work.
license: MIT
---

# Scientific Python Analysis

Goal: correct, reproducible analysis where each step is checked, not assumed.

## Workflow

1. **Understand the data first.** Load it, then inspect: shape, dtypes, ranges, missing values, units, obvious anomalies. State what you observe before transforming.
2. **State the question.** What are we computing and what would a sane answer look like (order of magnitude, sign, units)?
3. **Clean deliberately.** Handle missing/invalid values explicitly and say how (drop, impute, flag). Don't silently coerce.
4. **Transform vectorized.** Use numpy/pandas operations; avoid loops over rows. Keep track of shapes and axes.
5. **Compute with care.** Tolerances not `==` on floats; guard div-by-zero and log domains; watch dtype overflow; handle NaN/inf intentionally.
6. **Sanity-check every result.** Does the magnitude/sign/units make sense? Cross-check with a simple independent calculation where possible.
7. **Make it reproducible.** Seed RNGs; record data source/version and library versions; keep the pipeline runnable end to end.

## Discipline

- Show the intermediate numbers you relied on, not just the final figure.
- Distinguish what the data shows from what you infer (that's `results-interpretation`'s job).
- Don't fabricate or assume values — if a column/stat is needed and absent, say so.
- Verify any nontrivial library call's behavior rather than trusting recall of its API.

Pairs with `results-interpretation` and `plot-and-figure-review`.
