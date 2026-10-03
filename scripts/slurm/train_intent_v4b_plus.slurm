#!/bin/bash
# Slurm wrapper for scripts/train_intent_v4b_plus.sh. Submit from the repository root:
#   sbatch scripts/slurm/train_intent_v4b_plus.slurm
# Edit --partition / --gres for your cluster (released run: 1x NVIDIA A100 80GB PCIe).
#SBATCH --job-name=intent_v4b_plus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=intent_v4b_plus-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv
OUTPUT="runs/intent_v4b_plus_${SLURM_JOB_ID}" bash scripts/train_intent_v4b_plus.sh "$@"
