#!/bin/bash
# Canonical GPU smoke script name.

#SBATCH -J hydrai-smoke
#SBATCH -A YOUR-SLURM-ACCOUNT-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1
#SBATCH --qos=INTR
#SBATCH --time=00:45:00
#SBATCH --mail-type=NONE
#SBATCH --exclusive

set -euo pipefail

workdir="${SLURM_SUBMIT_DIR:-.}"
cd "$workdir" || exit 1
run_root="$(pwd -P)"
export HYDRAI_RUN_ROOT="$run_root"
export HYDRAI_ML_CONFIG="${workdir}/configs/ml/ml_data_generation_config.smoke.json"
numtasks=${HYDRAI_NTASKS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-1}}}}

module load rhel7/default-ccl 2>/dev/null || true

# ---------------------------------------------------------------------------
# Python interpreter: respect HYDRAI_PYTHON if set, otherwise use the python3
# active in the current environment (conda/venv/system).
# To pin a specific interpreter:
#   export HYDRAI_PYTHON=/path/to/your/python3
# ---------------------------------------------------------------------------
PYTHON="${HYDRAI_PYTHON:-$(which python3)}"

echo "JobID: ${SLURM_JOB_ID:-local}  (smoke test — see run_training_smoke_gpu_partition.sh)"
echo "HYDRAI_ML_CONFIG=$HYDRAI_ML_CONFIG"
echo "Tasks: $numtasks"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unknown}"
echo "Python: $PYTHON  ($(${PYTHON} --version 2>&1))"
echo "Dir: ${run_root}"
mkdir -p logs
{
  echo "time=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-.}"
  echo "run_root=${run_root}"
  echo "python=${PYTHON}"
  echo "config=${HYDRAI_ML_CONFIG}"
} > logs/RUN_ROOT.txt

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks="$numtasks" --exclusive bash -c "
    export TASK_ID=\$SLURM_PROCID
    export NTASKS=\$SLURM_NTASKS
    export HYDRAI_RUN_ROOT='${HYDRAI_RUN_ROOT}'
    export HYDRAI_ML_CONFIG='${HYDRAI_ML_CONFIG}'
    ${PYTHON} scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_\${SLURM_PROCID}.log 2>&1
  "
else
  echo "No active SLURM allocation detected; running one local task."
  export TASK_ID=0
  export NTASKS=1
  "${PYTHON}" scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_0.log 2>&1
fi

echo "Finished at $(date)"
