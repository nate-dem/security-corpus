#!/usr/bin/env bash
# Reuse all downloads and the completed RedSage profile-v1.
#SBATCH --job-name=web-primus-recovery
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/web-corpora/%x-%j.log
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
export WEB_DATA_DIR="${WEB_DATA_DIR:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/web-corpora}"
cd "${PROJECT_DIR}"
bash scripts/web_corpora/run.sh profile --source primus-fineweb --profile-name profile-v2
bash scripts/web_corpora/run.sh compare --source-profile primus-fineweb=profile-v2 \
  --output "${WEB_DATA_DIR}/comparison-v2.json"
exec bash scripts/web_corpora/run.sh compare-baseline --source-profile primus-fineweb=profile-v2 \
  --output "${WEB_DATA_DIR}/comparison-baseline-v2.json"
