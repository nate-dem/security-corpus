#!/usr/bin/env bash
#SBATCH --job-name=curation-v3
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/curation/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
export CURATION_RUBRIC_VERSION=v3
export CURATION_OUTPUT_DIR="${CURATION_OUTPUT_DIR:-/scratch/m000091/${USER}/curation/development-v3}"
nvidia-smi
exec bash scripts/curation/run_benchmark.sh
