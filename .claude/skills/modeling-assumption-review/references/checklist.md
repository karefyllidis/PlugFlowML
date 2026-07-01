# Modeling Assumption Review Checklist

Equations
- [ ] Correct governing equations for the regime
- [ ] Derivation sound; dropped terms justified

Assumptions
- [ ] Every simplifying assumption stated explicitly
- [ ] Each assumption valid across the operating range
- [ ] Linearity/equilibrium/steady-state/ideal claims checked

Units & parameters
- [ ] Dimensional consistency of every equation
- [ ] Unit conversions explicit and correct
- [ ] Parameter/constant sources and uncertainties documented
- [ ] Values physically plausible

Conditions
- [ ] Boundary conditions complete and physical
- [ ] Initial conditions consistent with equations
- [ ] Inputs within the model's domain of validity

Numerics
- [ ] Discretization/solver appropriate
- [ ] Stability criteria met (e.g. CFL)
- [ ] Convergence / mesh / step-size sensitivity checked
- [ ] Tolerances reasonable

Validation
- [ ] Checked against analytical limit, conservation law, data, or trusted reference
- [ ] Actually passes the check (not just asserted)
- [ ] Limits of validity stated
