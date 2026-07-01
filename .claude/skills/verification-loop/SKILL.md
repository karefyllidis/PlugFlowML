---
name: verification-loop
description: Goal-driven execution for coding tasks. Use to turn vague requests into verifiable success criteria, reproduce bugs with a failing test before fixing, and loop until checks pass.
license: MIT
---

# Verification Loop

Define success criteria first, then loop until they're met. Strong criteria let you work independently; weak criteria ("make it work") force constant back-and-forth.

## Turn the task into a goal

- "Fix the bug" → "Write a test that reproduces it (fails), then make it pass."
- "Add validation" → "Tests for each invalid input, then implement until green."
- "Refactor X" → "Suite passes before and after; behavior identical."
- "Make it faster" → "Define the metric and target (e.g. p95 < 100ms), measure before/after."

## Plan with checks

For anything multi-step, write the plan with an observable check per step:

```
1. <step> → verify: <what you'll observe>
2. <step> → verify: <what you'll observe>
3. <step> → verify: <what you'll observe>
```

Each step should be independently verifiable.

## Loop

1. Make the smallest change toward the next unmet criterion.
2. Run the check (tests, a script, a manual command).
3. If it fails, read the actual error and adjust. Don't guess-and-rerun blindly.
4. Repeat until all criteria pass.

## Discipline

- Run the tests; don't infer success from reading code.
- Keep tests deterministic: seed RNGs, freeze time, no live network unless intended.
- Don't declare done while any check fails or the work is partial.
- If you genuinely can't run a check, state exactly what remains unverified and why.
