#!/usr/bin/env bash
# Sync the bounded recovery inputs to Marlowe without deleting remote files.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-marlowe}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/m000091-pm05/natedem/security-corpus}"
INCLUDE_LEGACY_PAPERS=0

if [[ "${1:-}" == "--include-legacy-papers" ]]; then
  INCLUDE_LEGACY_PAPERS=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--include-legacy-papers]" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

required=(
  "data/filtering/v4/qa_to_score"
  "data/filtering/v4/citation_abstract_universe.parquet"
  "data/checkpoints/structured-v1"
  "data/arxiv/raw/metadata"
  "reports/recovery/arxiv"
  "reports/recovery/structured-v1"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required recovery input is missing: ${PROJECT_ROOT}/${path}" >&2
    exit 1
  fi
done

echo "Creating bounded destination on ${REMOTE_HOST}: ${REMOTE_ROOT}"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}'"

echo "Syncing corpus code (data, reports, Git history, and environments excluded)"
rsync -az --progress \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude 'data/' \
  --exclude 'reports/' \
  --exclude 'recovery/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  "${PROJECT_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"

inputs=(
  "./data/filtering/v4/qa_to_score"
  "./data/filtering/v4/qa_exact_duplicates.parquet"
  "./data/filtering/v4/manifest.json"
  "./data/filtering/v4/citation_abstract_universe.parquet"
  "./data/filtering/v4/citation_abstract_exact_duplicates.parquet"
  "./data/filtering/v4/citation_abstract_manifest.json"
  "./data/filtering/v4/sigma_structural_quality.parquet"
  "./data/filtering/v4/cloudtrail_structural_quality.parquet"
  "./data/checkpoints/structured-v1"
  "./data/arxiv/raw/metadata"
  "./reports/recovery/arxiv"
  "./reports/recovery/structured-v1"
)

if [[ ${INCLUDE_LEGACY_PAPERS} -eq 1 ]]; then
  inputs+=(
    "./data/arxiv/raw/source/normalized"
    "./data/training-clean-v1/normalized/source_id=arxiv/part-00000.parquet"
    "./data/filtering/v3/qwen_citation_abstract_kept_full.parquet"
  )
fi

echo "Syncing recovery inputs"
rsync -azR --progress --partial \
  "${inputs[@]}" "${REMOTE_HOST}:${REMOTE_ROOT}/"

echo "Input sync complete: ${REMOTE_HOST}:${REMOTE_ROOT}"
