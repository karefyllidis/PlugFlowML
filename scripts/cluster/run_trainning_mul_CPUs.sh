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
module load rhel7/default-ccl

# Go to project directory
cd "$workdir" || exit 1

# Print info
echo "JobID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Directory: $(pwd)"
echo "Tasks: $numtasks"
echo ""

# Create logs directory
mkdir -p logs

# Launch tasks - each runs the Python script with its own task ID
srun --ntasks="$numtasks" --exclusive bash -c '
  export TASK_ID=$SLURM_PROCID
  export NTASKS=$SLURM_NTASKS
  python3 scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_${SLURM_PROCID}.log 2>&1
'

echo ""
echo "Finished at $(date)"
echo "Check logs in: logs/main2_task_*.log"
