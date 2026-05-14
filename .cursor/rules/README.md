# HydrAI Cursor Rules

This directory contains Cursor AI agent rules for the HydrAI project. These rules help maintain consistency across the codebase and document architectural decisions.

## Files

- **`HYDRAI_NOTEBOOK_PLOT_COLORS.mdc`**: Cursor rule — preferred **`k`/`b`/`r`/`m`/`lime`**, **white bars + hatch**; **extension ladder** (extra series, colormaps, many categories); legacy catalogue.
- **`HYDRAI_PROJECT_CONVENTIONS.md`**: Comprehensive project standards covering:
  - Species data handling (mass fractions, chemistry-based lumping)
  - ML pipeline architecture (Main_3, Main_4, Main_5 workflows)
  - Plotting standards (aesthetics, scatter plots, chemistry diagnostics)
  - Unit conventions (pressure in bar, SI units)
  - Performance metrics (NMAE, R², MAE, RMSE)
  - Documentation standards (model cards, README maintenance)
  - Code style (imports, comments, variable naming)
  - HPC and parallel computing (SLURM, cross-platform)
  - Data compatibility (pickle versioning, pandas dtypes)
  - Matplotlib styling (setup_matplotlib function)
  - Feature scaling rationale
  - Portfolio/professional standards

## Usage

These rules are automatically loaded by Cursor AI agents when working in this repository. They serve as a "source of truth" for:

1. **Consistency**: Ensuring all code follows the same patterns
2. **Onboarding**: Helping new team members understand project conventions
3. **AI Assistance**: Guiding Cursor agents to make appropriate decisions
4. **Documentation**: Capturing tribal knowledge and architectural rationale

## Version Control

This directory is **hidden from git** (via `.gitignore`) to keep development artifacts separate from the main codebase.

## Updates

Rules are maintained as the project evolves. When updating:
- Document the rationale for changes
- Update version history in the rule file
- Consider impact on existing notebooks and documentation

## References

For detailed project documentation, see:
- `README.md` - Main project overview
- `docs/ML_CONFIG_GUIDE.md` - Configuration reference
- `docs/MODEL_CARD.md` - High-level model card
- `docs/SPECIES_LUMPING_MODEL_CARD.md` - Species methodology

---

**Created**: 2026-05-08  
**Last Updated**: 2026-05-14
