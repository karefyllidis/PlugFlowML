---
name: optuna-study-card
description: Document or audit an Optuna BO/MOO optimization study. Use when setting up a new study config, reviewing an existing one, or writing the inline model card header for a study YAML. Covers score function design, search space rationale, metric boundaries, constraints, and GPSampler pitfalls.
license: MIT
---

# Optuna Study Card

A study card documents every decision in an optimization config so a future user can reproduce, tune, or critique the study without reading the code.

## Collect

- **Study type** — BO (single-score scalar), MOO (Pareto multi-objective), or DOE (space-filling LHS).
- **Baseline** — the reference case all bounds are relative to; note the key values.
- **Sampler** — GPSampler, TPE, or NSGA-II; note why that sampler was chosen.
- **Score / objectives** — exact formula, metric names, sense (max/min), weights, scales. For BO: `S = Σ senseᵢ · wᵢ · (metricᵢ / scaleᵢ)`; state what scale values represent (characteristic magnitude at baseline) and what the weight priority ordering is.
- **Metric boundaries** — exactly where each output metric is evaluated (e.g., "sn=5 diffuser TE exit, not including downstream exhaust duct"). Ambiguous boundaries cause silent measurement inconsistencies.
- **Search space** — for each free parameter: type (scalar/integer/ramp/per-stage), bounds, physical meaning, and why those bounds bracket the baseline.
- **Constraints** — metric, limit value, enforced vs. logged only, physical rationale.
- **Fixed parameters** — what is held constant and why (e.g., thermochemistry fixed so optimizer explores geometry only).

## Output structure (inline YAML header)

```yaml
# ============================================================
# MODEL CARD — <study_id>
# ============================================================
# Study type : <BO single-score | MOO Pareto | DOE LHS>
# Baseline   : <path>  (<short description>)
# Optimizer  : <Optuna sampler> → <objective description>
#
# INTENT
#   <1–3 sentences: what is being optimized and why>
#
# SCORE FUNCTION  (BO only)
#   + w * (metric / scale)  for sense: maximize
#   - w * (metric / scale)  for sense: minimize
#   Scale = characteristic value near baseline.
#   Infeasible trials receive ±1e12 penalty.
#
# METRIC BOUNDARIES
#   <metric_name> : <exact evaluation plane / definition>
#   ...
#
# SEARCH SPACE DESIGN
#   <param>  [lo, hi]  type  — <physical meaning; why these bounds>
#   ...
#
# CONSTRAINTS (hard | optional)
#   <metric> ≤ <value> — <physical rationale>
#   ...
#
# RUN INSTRUCTIONS
#   Run:  python run_bo.py
#   Data: output/<group>/<run_id>/
# ============================================================
```

## GPSampler static search space rule

**GPSampler requires every trial to suggest the exact same set of parameter names.**
If a parameter name appears in some trials but not others, Optuna falls back to `RandomSampler` for those parameters and logs `[W] ... sampled independently using RandomSampler`.

This fires when an array parameter's length is itself a free integer — e.g., `n_stage ∈ [8, 13]` driving per-stage array `time_pass_0 … time_pass_{n_stage-1}`.

**Fix:** Always suggest `n_stage_max` flat keys regardless of the sampled `n_stage`, then truncate in `expand_params`:

```python
# In suggest_params:
n_stage_max = int(n_stage_spec.bounds[1])   # e.g. 13
for i in range(n_stage_max):               # always full set — static space
    lo, hi = _per_stage_ms_bounds(spec, i, n_stage_max)
    params[f"{name}_{i}"] = trial.suggest_float(f"{name}_{i}", lo, hi)
# Do NOT set params[name] = list — that bypasses truncation

# In expand_params:
_tp_keys = sorted([k for k in trial_params if k.startswith("time_pass_") ...])
expanded["time_pass"] = [float(trial_params[k]) for k in _tp_keys[:n_stage]]  # truncate
```

Check: if `param_csv_fieldnames` packs per-stage params as a JSON list, update it to emit flat columns `param_time_pass_0 … param_time_pass_{n_stage_max-1}` to match.

## DOE warm-start injection

Pre-evaluated DOE samples can seed the GP prior before BO trials begin, replacing random startup trials with real evaluations. Pattern:

1. Config: `warm_start_doe: true`, `doe_run_id: <doe_run_id>`
2. On fresh study (`len(study.trials) == 0`), load `doe_results_*.csv`.
3. Build `distributions` dict mirroring `suggest_params` (same keys, same bounds).
4. For each row: extract params as flat keys, score via the BO loss function, inject with `study.add_trial(create_trial(params=..., distributions=..., value=score))`.
5. Infeasible DOE rows get the penalty score so the GP learns their region is bad.

Guard: skip injection if the study already has trials (resume safety).

## Plot coverage checklist

Every optimization pipeline should have at minimum:

| Plot | BO | MOO | DOE |
|------|----|-----|-----|
| Convergence / improvement trace | ✓ | — | — |
| Feasibility history + violations | ✓ | ✓ | — |
| Parallel coordinates (params + objectives) | ✓ | ✓ | ✓ |
| Per-stage residence time heatmap (best/Pareto trials) | ✓ | ✓ | ✓ |
| Parameter sensitivity proxy (correlation) | ✓ | — | — |
| Objective scatter matrix | — | ✓ | — |
| Constraint margin distribution (feasible trials) | ✓ | ✓ | — |
| Output metric distributions (post-simulation) | — | — | ✓ |
| Input → output correlation heatmap | — | — | ✓ |
| Seed / best trial radar | — | ✓ | — |
