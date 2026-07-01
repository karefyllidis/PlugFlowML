---
name: split-audit
description: Audit train/validation/test splits for leakage and documentation gaps. Use before training, before publication, or when metrics look "too good."
license: MIT
---

# Split Audit

## Steps

1. **Locate** split definition (code, manifest, random seed, filters).
2. **Entity key** — what must not appear in two splits (sample, run, batch, trajectory, site).
3. **Check duplicates** — hash IDs across splits; flag near-duplicates (same parent sim, relabeled rows).
4. **Temporal / group leakage** — future data in train, entire groups split across sets incorrectly.
5. **Label leakage** — features that encode the label or post-outcome information.
6. **Holdout usage** — confirm tuning did not use the final test set.
7. **Report** — pass/fail per check with fixes.

Output a short **audit memo**: findings, severity, recommended fix. See `references/checklist.md`.
