---
name: results-interpretation
description: Turn raw experimental or analytical results into findings. Use when the user has numbers, metrics, or model outputs and wants conclusions, limitations, and next steps — interpreting what the results mean without overclaiming.
license: MIT
---

# Results Interpretation

Translate raw results into honest, useful findings. The discipline is separating what the data shows from what you infer.

## Steps

1. **Restate what was measured** and under what conditions (so the interpretation is anchored).
2. **Report the result plainly** with units, uncertainty, and the relevant comparison (vs. baseline/target/prior).
3. **State the finding** — what the result implies for the question. One claim at a time.
4. **Quantify confidence** — is the effect larger than the noise/variance? Statistically and practically significant? Don't dress up noise as signal.
5. **List limitations** — sample size, confounders, assumptions, generalization limits, anything that could explain the result other than the hypothesis.
6. **Next steps** — what to test or do next given the finding.

## Discipline

- Don't overclaim: correlation ≠ causation; a single run ≠ a robust result; in-distribution ≠ general.
- Report negative and null results honestly; they're informative.
- Show the numbers behind each claim; flag any result you couldn't fully verify.
- Distinguish "the data shows X" (fact) from "this suggests Y" (interpretation) from "we should Z" (recommendation).
- Watch for leakage, cherry-picked metrics, and survivorship bias.

See `references/checklist.md`. Pairs with `experiment-plan`, `plot-and-figure-review`, `red-team-review`.
