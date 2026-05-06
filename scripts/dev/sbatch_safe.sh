#!/usr/bin/env bash
set -euo pipefail

# Normalizes line endings to LF before calling sbatch.
# Usage:
#   bash scripts/dev/sbatch_safe.sh scripts/cluster/run_training_mul_GPUs.sh
#   bash scripts/dev/sbatch_safe.sh scripts/cluster/run_training_mul_CPUs.sh --account=...

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <batch-script> [sbatch args...]"
  exit 1
fi

batch_script="$1"
shift || true

if [[ ! -f "$batch_script" ]]; then
  echo "[ERROR] Batch script not found: $batch_script"
  exit 1
fi

# Enforce Unix line endings for this script before submission.
sed -i 's/\r$//' "$batch_script"

echo "[INFO] Normalized line endings (LF): $batch_script"
echo "[INFO] Submitting via sbatch..."
sbatch "$batch_script" "$@"
