#!/bin/bash
#!
#! SLURM job script: parallel Main_2 (training data generation) on multiple CPUs
#! Runs notebooks/Main_2_generate_training_data logic via scripts/run_main2_slurm_chunk.py
#! with one process per task; each task handles a chunk of the simulation list.
#!
#! Submit from project root:  sbatch scripts/run_trainning_mul_CPUs.sh
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J HydrAI_Main2_trainning_mul_CPUs
#! Which project should be charged:
#SBATCH -A YOUR-SLURM-ACCOUNT
#SBATCH -p cclake
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many parallel tasks (one Python process per task)?
#SBATCH --ntasks=11
#! How much wallclock time will be required?
#SBATCH --time=01:10:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! - This job runs Main_2 training data generation in parallel: NTASKS processes
#!   each run a chunk of the (reactant × parameter) simulations.
#! - Config: configs/ml_data_generation_config.json
#! - Output: data/training/task_0, data/training/task_1, ... (merge later if needed).
#! - No MPI: we use bash to launch NTASKS background processes.

#! Number of tasks allocated by SLURM (do not change):
numtasks=${SLURM_NTASKS:-1}

#! Work directory: must be project root (submit with: sbatch scripts/run_trainning_mul_CPUs.sh from project root)
workdir="${SLURM_SUBMIT_DIR:-.}"

#! Environment (reproduced at submission; adjust if needed):
. /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load rhel7/default-ccl 2>/dev/null || true
#! Load Python/miniconda if your cluster uses modules for Python:
# module load python/3.9  # uncomment and set version as on your system

#! Python interpreter (use system or env):
PYTHON="${PYTHON:-python3}"
RUNNER_SCRIPT="${workdir}/scripts/run_main2_slurm_chunk.py"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd "$workdir" || exit 1
echo "Changed directory to $(pwd)."
echo ""

JOBID="${SLURM_JOB_ID:-$$}"
echo "JobID: $JOBID"
echo "======="
echo "Time: $(date)"
echo "Running on: $(hostname)"
echo "Current directory: $(pwd)"
echo "Parallel tasks: $numtasks"
echo "Runner: $RUNNER_SCRIPT"
echo ""

if [ ! -f "$RUNNER_SCRIPT" ]; then
    echo "ERROR: Runner script not found: $RUNNER_SCRIPT"
    exit 1
fi

mkdir -p "${workdir}/logs"

# Launch one Python process per task in parallel (background), each with its TASK_ID
for (( task=0; task<numtasks; task++ )); do
    export TASK_ID=$task
    export NTASKS=$numtasks
    echo "Starting TASK_ID=$task NTASKS=$numtasks ..."
    $PYTHON "$RUNNER_SCRIPT" >> "${workdir}/logs/main2_task_${task}.log" 2>&1 &
done

# Wait for all background jobs (exit non-zero if any task failed)
wait
exitcode=$?
echo ""
echo "All $numtasks tasks finished at $(date). Logs: ${workdir}/logs/main2_task_*.log"
exit $exitcode
