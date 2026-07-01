---
name: subagent-orchestration
description: Coordinate subagents and parallel work on large tasks. Use when a job is big enough to split into independent units, needs fan-out research, or benefits from a separate agent for verification — to delegate cleanly and integrate results.
license: MIT
---

# Subagent Orchestration

Use additional agents to parallelize independent work and to isolate context — not to offload thinking you should do yourself. Each spawned agent starts cold and re-derives context, so delegation must pay for that cost.

## When to delegate

- **Fan-out search/research** — many files, sources, or naming conventions to sweep; you only need the conclusions.
- **Independent parallel units** — sub-tasks with no shared mutable state that can run at once.
- **Isolated verification** — a fresh agent reviews/tests work without the implementer's bias.
- **Context isolation** — a noisy sub-task (large logs, big dumps) you don't want polluting the main thread.

Don't delegate: small tasks, tightly coupled steps, or anything where the briefing costs more than doing it.

## How to delegate well

1. **Write a self-contained brief.** The subagent has none of your context. State the goal, the inputs/paths, the constraints, and exactly what to return.
2. **Define the contract.** Specify the output shape ("return the file paths and the one-line conclusion, not full dumps").
3. **Bound it.** Scope the search/work; cap depth. Autonomous agents must be bounded so they don't run away.
4. **Parallelize only independent work.** If B needs A's result, run them in sequence.
5. **Integrate and verify.** Treat returned claims as unverified until checked; reconcile conflicts; you own the final result.

## Verification pattern

For high-stakes work, use a separate agent to review or test the output. The reviewer should not be the author. Give it the success criteria and have it report pass/fail with evidence.

Pairs with `spec-first-planning` (decompose first) and `self-review` (the reviewer's checklist).
