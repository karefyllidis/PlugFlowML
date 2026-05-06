#!/bin/bash
# Smoke test on a GPU partition: many CPU workers, GPUs idle (Cantera is CPU-only).
# Uses configs/ml/ml_data_generation_config.smoke.json (tiny LHS sample).
#
# Submit from repo root (batch-style on interactive QoS):
#   sbatch scripts/cluster/run_training_smoke_gpu_partition.sh
#
# Or get an interactive allocation first, then from repo root:
#   salloc -J hydrai-smoke -A YOUR-SLURM-ACCOUNT-GPU -p ampere \
#     --nodes=1 --ntasks=32 --cpus-per-task=1 --gres=gpu:1 \
#     --time=00:10:00 --qos=INTR
#   ./scripts/cluster/run_training_smoke_gpu_partition.sh
#
# Monitor one task's progress (replace 0 with SLURM_PROCID):
#   tail -f logs/data_generation_progress_task_0.json
#
# Tiny smokes: most ranks get 0 assigned simulations (expected). Optional: lower
# #SBATCH --ntasks (e.g. 4) for smokes so more ranks do real work; keep total
# CPUs ≤ 32 per GPU on CSD3 and match srun task count.
#
# Production CPU runs: scripts/cluster/run_training_mul_CPUs.sh

#SBATCH -J hydrai-smoke
#SBATCH -A YOUR-SLURM-ACCOUNT-GPU
#SBATCH -p ampere
# CSD3 ampere: max 32 CPUs per 1 GPU. Use 32 Slurm tasks × 1 CPU so
# `srun --ntasks=32` matches the allocation (not 1 task × 32 CPUs).
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
# CSD3 interactive smoke: 10 min cap + QoS INTR (interactive / debug partition policy).
#SBATCH --time=00:10:00
#SBATCH --qos=INTR
#SBATCH --mail-type=NONE

set -euo pipefail

workdir="${SLURM_SUBMIT_DIR:-.}"
cd "$workdir" || exit 1
run_root="$(pwd -P)"
export HYDRAI_RUN_ROOT="$run_root"

export HYDRAI_ML_CONFIG="${workdir}/configs/ml/ml_data_generation_config.smoke.json"

# Must match Slurm task count (SLURM_NTASKS), not CPUs on one pseudo-task.
# Override: export HYDRAI_NTASKS=16
numtasks=${HYDRAI_NTASKS:-${SLURM_NTASKS:-1}}

module load rhel7/default-ccl 2>/dev/null || true

# ---------------------------------------------------------------------------
# Python interpreter: respect HYDRAI_PYTHON if set, otherwise use the python3
# that is active in the current environment (conda/venv/system).
# To pin a specific interpreter, set before submitting:
#   export HYDRAI_PYTHON=/path/to/your/python3
# ---------------------------------------------------------------------------
PYTHON="${HYDRAI_PYTHON:-$(which python3)}"
echo "Python: $PYTHON  ($(${PYTHON} --version 2>&1))"

echo "JobID: ${SLURM_JOB_ID:-local}"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Dir:  ${run_root}"
echo "Tasks: $numtasks"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unknown}"
echo "HYDRAI_ML_CONFIG=$HYDRAI_ML_CONFIG"
echo "Per-task progress: logs/data_generation_progress_task_*.json"
echo ""

mkdir -p logs
{
  echo "time=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-.}"
  echo "run_root=${run_root}"
  echo "python=${PYTHON}"
  echo "config=${HYDRAI_ML_CONFIG}"
  echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
  echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
  echo "numtasks_for_srun=$numtasks"
} > logs/RUN_ROOT.txt

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  : > logs/srun_step.err
  if ! srun --ntasks="$numtasks" bash -c "
    export TASK_ID=\$SLURM_PROCID
    export NTASKS=\$SLURM_NTASKS
    export HYDRAI_RUN_ROOT='${HYDRAI_RUN_ROOT}'
    export HYDRAI_ML_CONFIG='${HYDRAI_ML_CONFIG}'
    ${PYTHON} scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_\${SLURM_PROCID}.log 2>&1
  " 2>> logs/srun_step.err; then
    srun_ec=$?
    echo "srun_exit_code=$srun_ec" >> logs/RUN_ROOT.txt
    echo "hint=see logs/srun_step.err and slurm-${SLURM_JOB_ID}.out" >> logs/RUN_ROOT.txt
    exit "$srun_ec"
  fi
else
  echo "No active SLURM allocation detected; running one local task."
  export TASK_ID=0
  export NTASKS=1
  "${PYTHON}" scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_0.log 2>&1
fi

echo "Finished at $(date)"
