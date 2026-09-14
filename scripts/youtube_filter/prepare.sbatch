#!/usr/bin/env bash
#SBATCH --job-name=youtube-pilot-prepare
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/youtube-filter/%x-%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
cd "${PROJECT_DIR}"
exec bash scripts/youtube_filter/run.sh prepare "$@"
