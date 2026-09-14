#!/usr/bin/env bash
# Submit from the project root. CPU-only; no GPU allocation is needed.
#SBATCH --job-name=web-corpora
#SBATCH --account=marlowe-m000091
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/web-corpora/%x-%j.log

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
cd "${PROJECT_DIR}"
if [[ $# -eq 0 ]]; then set -- full; fi
exec bash scripts/web_corpora/run.sh "$@"
