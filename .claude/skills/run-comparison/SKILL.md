---
name: run-comparison
description: Compare ML or simulation experiment runs fairly. Use when choosing between runs, writing results, or explaining metric differences — requires run metadata and what changed.
license: MIT
---

# Run Comparison

## Gather per run

Run ID, git commit, config file, data version, seeds, hardware, metric table (same metric names), training steps/epochs.

## Compare

1. **Align metrics** — same split, same definition; reject apples-to-oranges.
2. **Diff config** — list only changed keys (hypothesis table).
3. **Uncertainty** — multiple seeds or error bars if available; say if single seed.
4. **Cost** — wall time, GPU hours if relevant.
5. **Conclusion** — which run wins on which criterion; what is inconclusive.

## Output

Short table + narrative recommendation. Flag **leakage risks** or **tuned-on-test** if detected.

Pairs with `ablation-plan`.
