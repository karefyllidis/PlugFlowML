#!/bin/bash
# GPU production run — uses configs/ml/main2_data_generation_config.json (full config).
# For smoke tests use: scripts/cluster/run_training_smoke_gpu_partition.sh
#
# Submit from repo root:
#   sbatch scripts/cluster/run_training_mul_GPUs.sh

#SBATCH -J hrgpu
#SBATCH -A YOUR-SLURM-ACCOUNT-GPU
#SBATCH -p ampere
# CSD3 ampere: 32 CPUs per GPU — use 32 tasks × 1 CPU for srun fan-out.
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail

workdir="${SLURM_SUBMIT_DIR:-.}"
cd "$workdir" || exit 1
run_root="$(pwd -P)"
export PLUGFLOWML_RUN_ROOT="$run_root"
export PLUGFLOWML_ML_CONFIG="${workdir}/configs/ml/main2_data_generation_config.json"
numtasks=${PLUGFLOWML_NTASKS:-${SLURM_NTASKS:-1}}

module load rhel7/default-ccl 2>/dev/null || true

# Safety guard: hard-stop srun if it hangs, so nodes are released.
# Override if needed, e.g. export PLUGFLOWML_SRUN_TIMEOUT=43000s
SRUN_TIMEOUT="${PLUGFLOWML_SRUN_TIMEOUT:-42600s}"  # 11h50m for a 12h job

# ---------------------------------------------------------------------------
# Python interpreter: respect PLUGFLOWML_PYTHON if set, otherwise use the python3
# active in the current environment (conda/venv/system).
# To pin a specific interpreter:
#   export PLUGFLOWML_PYTHON=/path/to/your/python3
# ---------------------------------------------------------------------------
PYTHON="${PLUGFLOWML_PYTHON:-$(which python3)}"

echo "JobID: ${SLURM_JOB_ID:-local}"
echo "PLUGFLOWML_ML_CONFIG=$PLUGFLOWML_ML_CONFIG"
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
  echo "config=${PLUGFLOWML_ML_CONFIG}"
  echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
  echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
  echo "numtasks_for_srun=$numtasks"
} > logs/RUN_ROOT.txt

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  : > logs/srun_step.err
  if ! timeout "$SRUN_TIMEOUT" srun --ntasks="$numtasks" bash -c "
    export TASK_ID=\$SLURM_PROCID
    export NTASKS=\$SLURM_NTASKS
    export PLUGFLOWML_RUN_ROOT='${PLUGFLOWML_RUN_ROOT}'
    export PLUGFLOWML_ML_CONFIG='${PLUGFLOWML_ML_CONFIG}'
    ${PYTHON} scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_\${SLURM_PROCID}.log 2>&1
  " 2>> logs/srun_step.err; then
    srun_ec=$?
    echo "srun_exit_code=$srun_ec" >> logs/RUN_ROOT.txt
    echo "srun_timeout=$SRUN_TIMEOUT" >> logs/RUN_ROOT.txt
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
