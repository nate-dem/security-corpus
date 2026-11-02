#!/usr/bin/env bash
#SBATCH --job-name=corpus-production
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/curation/%x-%A_%a.log
set -euo pipefail
cd "${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
nvidia-smi
exec bash scripts/curation/run_first_batch.sh --production
