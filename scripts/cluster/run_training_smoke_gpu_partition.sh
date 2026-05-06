#!/bin/bash
# Smoke test on a GPU partition: many CPU workers, GPUs idle (Cantera is CPU-only).
# Uses configs/ml/ml_data_generation_config.smoke.json (tiny LHS sample).
#
# Submit from repo root:
#   sbatch scripts/cluster/run_training_smoke_gpu_partition.sh
#
# Monitor one task's progress (replace 0 with SLURM_PROCID):
#   tail -f logs/data_generation_progress_task_0.json
#
# Production CPU runs: scripts/cluster/run_training_mul_CPUs.sh

#SBATCH -J hydrai-smoke
#SBATCH -A YOUR-SLURM-ACCOUNT-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --qos=INTR
#SBATCH --time=00:45:00
#SBATCH --mail-type=NONE

set -euo pipefail

workdir="${SLURM_SUBMIT_DIR:-.}"
cd "$workdir" || exit 1

export HYDRAI_ML_CONFIG="${workdir}/configs/ml/ml_data_generation_config.smoke.json"

numtasks=${SLURM_NTASKS:-1}

module load rhel7/default-ccl 2>/dev/null || true

echo "JobID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Dir:  $(pwd)"
echo "Tasks: $numtasks"
echo "HYDRAI_ML_CONFIG=$HYDRAI_ML_CONFIG"
echo "Per-task progress: logs/data_generation_progress_task_*.json"
echo ""

mkdir -p logs

srun --ntasks="$numtasks" --exclusive bash -c '
  export TASK_ID=$SLURM_PROCID
  export NTASKS=$SLURM_NTASKS
  python3 scripts/cluster/run_main2_slurm_chunk.py >> logs/main2_task_${SLURM_PROCID}.log 2>&1
'

echo "Finished at $(date)"
