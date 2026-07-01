---
name: systematic-debugging
description: Debug methodically instead of guess-and-check. Use when something fails, errors, or produces wrong output — to reproduce, isolate, form and test hypotheses, and fix the root cause rather than the symptom.
license: MIT
---

# Systematic Debugging

Don't shotgun edits and rerun hoping it works. Find the root cause, then fix once.

## The loop

1. **Reproduce reliably.** Get a minimal, deterministic repro. If you can't trigger it on demand, you can't confirm a fix. Capture the exact input, environment, and the full error/stack.
2. **Read the actual error.** The message and traceback usually name the file, line, and cause. Read it before theorizing.
3. **Localize.** Narrow where it goes wrong: bisect the code path, add targeted logging/prints of real values, check the boundary between "right" and "wrong" state. Binary-search the history (`git bisect`) if it's a regression.
4. **Form a hypothesis.** State what you think is wrong and *why*, as a falsifiable claim. "X is null here because the loader skips empty rows."
5. **Test the hypothesis cheaply.** One change that confirms or refutes it. Don't bundle a fix with the diagnosis.
6. **Fix the root cause.** Not the symptom. If you're catching an exception to hide it, ask why it's thrown.
7. **Verify and guard.** Confirm the repro now passes, run the surrounding tests for regressions, and add a test that locks in the fix.

## Discipline

- Change one thing at a time; revert changes that didn't help.
- Trust observed values over assumptions — print/inspect rather than guess what a variable holds.
- Check the obvious first: stale cache/build, wrong env or version, uninstalled change, wrong file.
- If stuck after a few hypotheses, step back and question an assumption you've been treating as fact.
- Don't claim it's fixed until the original repro passes.

Pairs with `verification-loop` (lock the fix with a test) and `self-review`.
