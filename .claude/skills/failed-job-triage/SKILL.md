---
name: failed-job-triage
description: Triage a failed HPC/batch job from logs and exit codes. Use when SLURM/PBS jobs fail, OOM, timeout, or missing modules — produce likely cause and next commands.
license: MIT
---

# Failed Job Triage

## Inputs

Exit code, `.out`/`.err` tail (last ~100 lines), `#SBATCH` resource lines, node/GPU type, recent code/data changes.

## Diagnosis order

1. **Immediate error** — import error, file not found, permission, segfault message.
2. **OOM / cgroup kill** — increase memory or batch size; checkpoint more often.
3. **Timeout** — wall time vs actual progress; checkpoint/resume or partition with longer limit.
4. **Quota / disk** — scratch full, inode limit, home quota.
5. **Environment** — wrong module, CUDA mismatch, missing bind mount in containers.
6. **Preemption** — requeue policy; need shorter chunks + checkpoints.

## Output

- **Likely cause** (ranked)
- **Evidence** (log lines)
- **Fix** (concrete edits to script or resources)
- **Verify** (minimal resubmit command)

Do not guess cluster policy — state assumptions.
