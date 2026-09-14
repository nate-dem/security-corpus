#!/usr/bin/env bash
# Same prepared texts and v2 rubric as the 8B run; only the model/TP change.
# This is a bounded comparison, not production filtering or a new keep policy.
#SBATCH --job-name=youtube-qwen32b-v2
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/youtube-filter/%x-%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
export YOUTUBE_DATA_DIR="${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}"
export YOUTUBE_SCORE_DIR="${YOUTUBE_32B_V2_SCORE_DIR:-${YOUTUBE_DATA_DIR}/pilot-qwen32b-v2}"
cd "${PROJECT_DIR}"
nvidia-smi
exec bash scripts/youtube_filter/run.sh score \
  --evidence-format segment-ids \
  --model Qwen/Qwen3-32B \
  --model-revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --tensor-parallel-size 2 "$@"
