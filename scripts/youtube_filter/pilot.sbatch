#!/usr/bin/env bash
#SBATCH --job-name=youtube-qwen-pilot
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/youtube-filter/%x-%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
cd "${PROJECT_DIR}"
nvidia-smi
exec bash scripts/youtube_filter/run.sh score "$@"
