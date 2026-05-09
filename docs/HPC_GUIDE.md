# HPC Guide (CSD3 / SLURM)

The cluster scripts under `scripts/cluster/` are currently tuned for the **University of Cambridge CSD3** SLURM environment (accounts, partitions, and module names).

On CSD3 **ampere**, GPU jobs cap CPUs per GPU (for example, 32 CPUs per 1 GPU). Parallel `srun` workers must use **`--ntasks=N --cpus-per-task=1`**, not one task with `N` CPUs, or `srun` will fail with "More processors requested than permitted."

GPU smoke jobs use **`--time=00:10:00`** and **`--qos=INTR`** (interactive-style short runs). For multi-hour production sweeps, use a non-interactive QoS and longer `--time` in `run_training_mul_CPUs.sh` or a custom `#SBATCH` header.

For other clusters, edit `#SBATCH` directives and `module load` commands before submission.
