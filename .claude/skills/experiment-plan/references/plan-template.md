# Experiment Plan — {{title}}

## Question / Hypothesis
{{Falsifiable statement. What confirms it? What refutes it?}}

## Variables
- Independent (varied): {{…}}
- Dependent (measured): {{…}}
- Controlled (fixed): {{…}}
- Possible confounders: {{…}}

## Conditions / Ablations
| Condition | Description | Isolates |
|---|---|---|
| Baseline | {{…}} | — |
| {{Treatment}} | {{…}} | {{factor}} |
| {{Ablation}} | {{…}} | {{factor}} |

## Metrics
- Primary: {{name — exact definition / formula}}
- Secondary / diagnostic: {{…}}

## Controls & Setup
- Baselines / control: {{…}}
- Randomization & seeds: {{…}}
- Data / splits: {{source, version, train/val/test}}
- Replicates / seeds per condition: {{n}}

## Expected Outcomes & Failure Modes
- Predicted result and meaning: {{…}}
- Ways this could mislead: {{leakage, confounds, underpowered, metric gaming…}}

## Decision Rule
{{If result X → do A; if Y → do B.}}

## Reproducibility
Seeds: {{…}} · Library versions: {{…}} · Data provenance: {{…}}
