---
name: pr-author
description: Draft a pull request title and description with testing notes and risk section. Use when the user is about to open a PR or wants review-ready text from a diff summary.
license: MIT
---

# PR Author

## Inputs

Summary of changes, issue link, test commands run, breaking changes, screenshots if UI.

## Produce

- **Title:** imperative, scoped (`Fix val split leakage in dataset loader`)
- **Body sections:** Summary | Changes | How tested | Risks/rollout | Related issues
- **Reviewer hints:** files needing careful look, migration steps

## Quality bar

Every claim in "How tested" must match what was actually run or say "not run — reason."

Pairs with `spec-first-planning` when the PR closes a spec'd change.
