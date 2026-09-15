#!/usr/bin/env bash
#SBATCH --job-name=curation-quality-retry
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/curation/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
nvidia-smi
exec bash scripts/curation/run_first_batch.sh --quality-retry
