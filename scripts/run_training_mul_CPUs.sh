#!/bin/bash
#
# SLURM job script: parallel Main_2 (training data generation) on multiple CPUs
# Runs notebooks/Main_2_generate_training_data logic via scripts/run_main2_slurm_chunk.py
# One process per task; each task handles a chunk of the simulation list.
#
# Submit from project root:
#   sbatch scripts/run_training_mul_CPUs.sh
#
# If used on top of any other environments such as TS - remember to "deactivate" first

#############################################################
#### Modify the options in this section as appropriate ######
#############################################################

#SBATCH -J Training
#SBATCH -A YOUR-SLURM-ACCOUNT-CPU
#SBATCH -p cclake
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE

# Get number of tasks
numtasks=${SLURM_NTASKS:-1}

# Work directory (where you submitted from)
workdir="${SLURM_SUBMIT_DIR:-.}"

# Load only the required module(s)
module purge
module load rhel7/default-ccl
# module load python/3.10  # Uncomment if your cluster requires explicit python module

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
srun --ntasks=${numtasks} --exclusive bash -c '
  export TASK_ID=${SLURM_PROCID}
  export NTASKS=${SLURM_NTASKS}
  python3 scripts/run_main2_slurm_chunk.py \
    >> logs/main2_task_${SLURM_PROCID}.log 2>&1
'

echo ""
echo "Finished at $(date)"
echo "Check logs in: logs/main2_task_*.log"
