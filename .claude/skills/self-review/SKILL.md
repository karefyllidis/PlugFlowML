---
name: self-review
description: Red-team your own work before delivering. Use after writing code, a document, or an analysis to catch bugs, hidden assumptions, hallucinated facts, scope creep, and unverified claims before the user sees them.
license: MIT
---

# Self-Review

Before you deliver, switch from author to adversarial reviewer. Assume the work is wrong and try to prove it. The cheapest bug to fix is the one you catch yourself.

## Run the checklist

**Correctness**
- Did I run it / check the output, or am I assuming it works? Run it.
- Edge cases: empty, null, zero, negative, huge, duplicate, unicode, concurrent?
- Off-by-one, wrong sign, wrong units, swapped arguments?

**Honesty & hallucination**
- Every API/function/import I used — does it actually exist in this version? Verify the ones I'm not certain of.
- Every factual or numeric claim — is it backed by something I observed or cited? Label anything I couldn't verify.
- Did I claim "done/tested/works" beyond what I checked?

**Scope**
- Does every changed line trace to the request? Revert drive-by edits, reformatting, and unrequested features.
- Did I introduce abstractions or config nobody asked for? Simplify.

**Clarity (for writing)**
- Does the opening state the conclusion? Can a section be cut without loss?
- Are claims specific and supported? Any filler, clichés, or hedging to remove?

**Failure modes**
- What's the worst input? What happens on error — clear message or silent corruption?
- Security: secrets, untrusted input, dangerous calls?

## Then

List what you actually verified vs. what remains unverified, and say so in the delivery. If review surfaced a real problem, fix it before sending — don't ship with a footnote.

For high-stakes work, have a *separate* agent run this checklist (see `subagent-orchestration`); an author reviewing their own work has blind spots.
