#!/bin/bash
#SBATCH --job-name=cosmosr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --output=cosmosr-%j.out

# Set partition/account/GPU resources for your site through sbatch.
set -eo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the prepared field root}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh run directory}"
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
cd "$REPO_ROOT"
args=(--config configs/state56.json --data-root "$DATA_ROOT" --output "$OUTPUT_DIR")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    args+=(--resume "$RESUME_CHECKPOINT")
fi
"$PYTHON" train.py "${args[@]}"
