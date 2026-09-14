#!/usr/bin/env bash
# Full English candidate preparation; CPU only, no quality selection.
#SBATCH --job-name=youtube-english
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/youtube/%x-%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
cd "${PROJECT_DIR}"
exec bash scripts/youtube/run_prepare_english.sh "$@"
