---
name: plot-and-figure-review
description: Review or improve a scientific plot or figure. Use when checking a chart/figure for correctness, clarity, and honesty — axes, units, scales, labels, color, and whether the visual supports the claim it's making.
license: MIT
---

# Plot & Figure Review

A figure should be correct, self-explanatory, and honest. Review against these.

## Correctness
- Right chart type for the data (don't use lines for unordered categories; don't pie-chart many slices).
- Axes show the right variables; transformations (log, normalized) stated.
- Data mapped correctly — no swapped series, wrong aggregation, or mislabeled groups.

## Clarity
- Both axes labeled **with units**; readable tick density.
- Title/caption makes the figure stand alone (what, conditions, takeaway).
- Legend present when multiple series; series distinguishable.
- Color: colorblind-safe palette; color carries meaning, isn't decorative; consistent across related figures.
- Uncertainty shown where relevant (error bars, CI bands) and defined.

## Honesty
- Axis ranges not truncated to exaggerate; baselines sensible (bar charts start at 0 unless justified).
- Linear vs. log scale chosen to inform, not mislead; dual axes avoided unless necessary and clearly marked.
- No cherry-picked window hiding contrary data.
- Sample size / n visible where it matters.

## Output
List issues by severity with a concrete fix for each, then confirm whether the figure actually supports the claim it's used for. If reviewing code that generates the plot, point to the lines to change.

See `references/checklist.md`. Pairs with `results-interpretation`.
