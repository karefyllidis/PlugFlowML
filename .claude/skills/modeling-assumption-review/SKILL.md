---
name: modeling-assumption-review
description: Review the assumptions behind a model or simulation. Use when checking equations, physical/mathematical assumptions, units, boundary/initial conditions, and validation logic — common for scientific, engineering, and ML modeling work.
license: MIT
---

# Modeling Assumption Review

Pressure-test the assumptions a model rests on before trusting its output.

## What to review

- **Governing equations:** are they the right ones for this regime? Derivation sound? Terms dropped/kept justified?
- **Assumptions:** every simplifying assumption made explicit (steady-state, incompressible, ideal, linear, equilibrium, well-mixed, etc.) — and is each valid in the operating range?
- **Units & dimensions:** dimensional consistency of every equation; unit conversions explicit and correct.
- **Parameters & constants:** sources and uncertainties of rate constants, coefficients, material properties; sensible values.
- **Boundary & initial conditions:** physically meaningful, complete, and consistent with the equations.
- **Domain of validity:** where does the model break down? Are inputs within that domain?
- **Numerics:** discretization/solver choices, stability (e.g. CFL), convergence, tolerances, mesh/step sensitivity.
- **Validation logic:** what is the model checked against — analytical limits, conservation laws, experimental data, a trusted reference? Does it actually pass?

## Output

For each issue: the assumption, why it may not hold here, the likely consequence, and how to test or relax it. Separate "invalidates results" from "limits scope". Flag any claim of validity that isn't backed by a check.

See `references/checklist.md`.
