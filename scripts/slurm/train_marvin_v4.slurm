#!/bin/bash
# Slurm wrapper for scripts/train_marvin_v4.sh. Submit from the repository root:
#   sbatch scripts/slurm/train_marvin_v4.slurm
# Edit --partition / --gres for your cluster (released run: 1x NVIDIA A100 80GB PCIe).
#SBATCH --job-name=marvin_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=marvin_v4-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv
OUTPUT="runs/marvin_v4_${SLURM_JOB_ID}" bash scripts/train_marvin_v4.sh "$@"
