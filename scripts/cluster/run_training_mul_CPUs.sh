#!/bin/bash
#SBATCH -J training
#SBATCH -A YOUR-SLURM-ACCOUNT-CPU
#SBATCH -p cclake
#SBATCH --nodes=4
#SBATCH --ntasks=224
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE

# Get number of tasks
numtasks=${SLURM_NTASKS:-1}

# Work directory (where you submitted from)
workdir="${SLURM_SUBMIT_DIR:-.}"

# Load only the required module
module load rhel7/default-ccl 2>/dev/null || true

# Go to project directory
cd "$workdir" || exit 1
run_root="$(pwd -P)"
export HYDRAI_RUN_ROOT="$run_root"

# ---------------------------------------------------------------------------
# Python interpreter: respect HYDRAI_PYTHON if set, otherwise use the python3
# active in the current environment (conda/venv/system).
# To pin a specific interpreter:
#   export HYDRAI_PYTHON=/path/to/your/python3
# ---------------------------------------------------------------------------
PYTHON="${HYDRAI_PYTHON:-$(which python3)}"

# Print info
echo "JobID: ${SLURM_JOB_ID:-local}"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Directory: ${run_root}"
echo "Tasks: $numtasks"
echo "Python: $PYTHON  ($(${PYTHON} --version 2>&1))"
echo ""

# Create logs directory
mkdir -p logs
{
  echo "time=$(date -Is)"
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-.}"
  echo "run_root=${run_root}"
  echo "python=${PYTHON}"
  echo "config=${HYDRAI_ML_CONFIG:-configs/ml/ml_data_generation_config.json}"
} > logs/RUN_ROOT.txt

# Launch tasks - each runs the Python script with its own task ID
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks="$numtasks" --exclusive bash -c "
    export TASK_ID=\$SLURM_PROCID
    export NTASKS=\$SLURM_NTASKS
    export HYDRAI_RUN_ROOT='${HYDRAI_RUN_ROOT}'
    ${PYTHON} scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_\${SLURM_PROCID}.log 2>&1
  "
else
  echo "No active SLURM allocation detected; running one local task."
  export TASK_ID=0
  export NTASKS=1
  "${PYTHON}" scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_0.log 2>&1
fi

echo ""
echo "Finished at $(date)"
echo "Check logs in: logs/main2_task_*.log"
