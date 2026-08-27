#!/usr/bin/env bash
# Checkpoint completed Marlowe filtering and paper outputs on the laptop.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-marlowe}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/m000091-pm05/natedem/security-corpus}"
INCLUDE_ARXIV_DOWNLOADS=0

if [[ "${1:-}" == "--include-arxiv-downloads" ]]; then
  INCLUDE_ARXIV_DOWNLOADS=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--include-arxiv-downloads]" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

outputs=(
  "data/filtering/v4/qwen_qa_shards"
  "data/filtering/v4/qwen_qa_decisions.parquet"
  "data/filtering/v4/qwen_qa_decisions.manifest.json"
  "data/filtering/v4/qwen_citation_abstract_shards"
  "data/filtering/v4/qwen_citation_abstract_decisions.parquet"
  "data/filtering/v4/qwen_citation_abstract_decisions.manifest.json"
  "data/filtering/v4/citation_accepted_ids.txt"
  "data/filtering/v4/citation_accepted_ids.manifest.json"
  "data/arxiv/raw/metadata/citations"
  "data/arxiv/raw/source/normalized"
  "data/rebuilt/arxiv-normalized"
  "reports/release"
)

if [[ ${INCLUDE_ARXIV_DOWNLOADS} -eq 1 ]]; then
  outputs+=("data/arxiv/raw/source/downloads")
fi

for path in "${outputs[@]}"; do
  echo "Checkpointing ${path}"
  mkdir -p "${PROJECT_ROOT}/$(dirname "${path}")"
  rsync -az --progress --partial \
    "${REMOTE_HOST}:${REMOTE_ROOT}/${path}" "${PROJECT_ROOT}/$(dirname "${path}")/"
done

echo "Output checkpoint sync complete"
