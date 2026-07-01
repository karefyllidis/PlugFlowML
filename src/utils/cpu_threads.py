"""Cap CPU threads for PyTorch / BLAS (Main_6 notebook)."""

from __future__ import annotations

import os
from typing import Any

import torch


def available_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def resolve_n_cpu_cores(n_cpu_cores: int | None) -> int:
    """Return effective core budget (1 .. logical CPUs). None = all available."""
    if n_cpu_cores is None:
        return available_cpu_count()
    return max(1, min(int(n_cpu_cores), available_cpu_count()))


def resolve_optuna_n_jobs(
    optuna_n_jobs: int | None,
    device_type: str,
    n_cpu_cores: int | None,
) -> int:
    """Default parallel Optuna trials: 1 on GPU/MPS; modest split on CPU."""
    if optuna_n_jobs is not None:
        n = max(1, int(optuna_n_jobs))
        if device_type == "cpu":
            return min(n, resolve_n_cpu_cores(n_cpu_cores))
        return 1 if device_type in ("cuda", "mps") else n
    if device_type in ("cuda", "mps"):
        return 1
    n = resolve_n_cpu_cores(n_cpu_cores)
    return max(1, min(4, n // 2))


def configure_cpu_threads(
    n_cpu_cores: int | None,
    parallel_jobs: int = 1,
    device_type: str = "cpu",
) -> dict[str, Any]:
    """
    Set PyTorch and common BLAS thread env vars.

    *parallel_jobs* — Optuna ``n_jobs`` (or 1 for §8 training). Thread budget is
    ``n_cores // parallel_jobs`` per process on CPU; on CUDA/MPS host work uses
  ``min(n_cores, 8)`` threads so the GPU is not starved by CPU oversubscription.
    """
    n_cores = resolve_n_cpu_cores(n_cpu_cores)
    jobs = max(1, int(parallel_jobs))
    if device_type == "cpu":
        per_job = max(1, n_cores // jobs)
    else:
        jobs = 1
        per_job = max(1, min(n_cores, 8))

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(per_job)

    torch.set_num_threads(per_job)
    try:
        torch.set_num_interop_threads(max(1, min(4, per_job)))
    except RuntimeError:
        pass

    return {
        "n_cores": n_cores,
        "parallel_jobs": jobs,
        "torch_threads_per_job": per_job,
        "device_type": device_type,
    }
