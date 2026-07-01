---
name: batch-job-template
description: Draft a SLURM/sbatch (or similar) batch script for scientific or ML jobs. Use when the user needs a cluster job with sane resources, logging, modules, and checkpoint hooks.
license: MIT
---

# Batch Job Template

## Ask first

Scheduler (SLURM/PBS/LSF), partition/QOS, GPUs, expected runtime, input/output paths, conda/module env, array vs single job.

## Script skeleton

- Shebang and `set -euo pipefail`
- `#SBATCH` directives: job name, time, memory, cpus, gpus, partition, output/error paths with `%j`
- Comment block: purpose, author, date, git commit if known
- Module load / env activate (pinned)
- `cd` to project dir or use `$SLURM_SUBMIT_DIR`
- Run command with explicit config path
- Optional: checkpoint path env var, array index handling

## Deliver

Complete script + 3-line "how to submit" (`sbatch ...`) + smoke-test suggestion (1 node, short time).

Pairs with `failed-job-triage`.
