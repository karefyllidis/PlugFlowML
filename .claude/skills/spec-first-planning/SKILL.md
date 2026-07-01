---
name: spec-first-planning
description: Plan before building on non-trivial work. Use when a task is large, ambiguous, or multi-file — to produce a short written spec (goal, scope, approach, risks, checks) and get alignment before writing code.
license: MIT
---

# Spec-First Planning

For anything beyond a small, obvious change, write a short spec and confirm it before implementing. A wrong plan caught in 5 lines is cheaper than a wrong implementation caught in 500.

## When to use

Use when any of these are true: touches multiple files/modules, the requirements are ambiguous, there's more than one reasonable design, it changes a public interface or data format, or it's risky/hard to reverse. Skip for typos, one-line fixes, and obvious local changes.

## The spec (keep it to one screen)

```
## Goal
<one sentence: the outcome and who it's for>

## In scope / Out of scope
- In: <what this change does>
- Out: <explicitly what it does NOT do>

## Approach
<the chosen design in 2–5 bullets; name key files/functions>

## Alternatives considered
<1–2 other options and why not — only if a real choice exists>

## Risks & unknowns
<what could break; what you're unsure about; assumptions made>

## Verification
<the checks that prove it works: tests, commands, manual steps>
```

## Rules

- Surface assumptions explicitly; if one would change the design, ask before coding.
- Present real alternatives when they exist — don't silently pick one.
- Right-size the spec to the task. Three bullets is fine for medium work.
- Once approved, implement in verifiable steps; if reality diverges from the spec, stop and update it rather than improvising silently.

Pairs with `verification-loop` (executing the plan) and `self-review` (checking the result).
