---
name: environment-audit
description: Audit a Python/scientific environment for reproducibility — lockfiles, pins, CUDA match, CI vs local gaps. Use before release, paper submission, or "works on my machine" incidents.
license: MIT
---

# Environment Audit

## Check

- Manifest files present (`pyproject.toml`, `requirements.txt`, `environment.yml`, lockfile)
- Install instructions in README
- Python version documented
- GPU/CUDA packages consistent with hardware target
- Dev vs runtime dependency separation
- CI install path matches developer docs
- Known unpinned packages listed

## Output

| Item | Status | Risk | Fix |
|------|--------|------|-----|

Priority-ordered action list.
