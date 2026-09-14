#!/usr/bin/env bash
# Reuse prepared pilot-v1; preserve pilot-qwen8b-v1 decisions for comparison.
#SBATCH --job-name=youtube-qwen-v2
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
export YOUTUBE_DATA_DIR="${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}"
# Pin this job's output separately even if a previous score directory is exported.
export YOUTUBE_SCORE_DIR="${YOUTUBE_V2_SCORE_DIR:-${YOUTUBE_DATA_DIR}/pilot-qwen8b-v2}"
cd "${PROJECT_DIR}"
nvidia-smi
exec bash scripts/youtube_filter/run.sh score --evidence-format segment-ids "$@"
