#!/usr/bin/env bash
# Repair evidence for failed v3 spans only; preserve all original decisions.
#SBATCH --job-name=youtube-evidence-v3
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/youtube-filter/%x-%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
export YOUTUBE_DATA_DIR="${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}"
export YOUTUBE_SCORE_DIR="${YOUTUBE_32B_V3_SCORE_DIR:-${YOUTUBE_DATA_DIR}/pilot-qwen32b-v3}"
export YOUTUBE_REPAIR_DIR="${YOUTUBE_V3_REPAIR_DIR:-${YOUTUBE_DATA_DIR}/pilot-qwen32b-v3-evidence-repair}"
cd "${PROJECT_DIR}"
nvidia-smi
exec bash scripts/youtube_filter/run.sh repair-evidence "$@"
