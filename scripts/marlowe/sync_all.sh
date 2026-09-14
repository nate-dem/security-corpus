#!/usr/bin/env bash
# Copy the project, local corpus, and reports to an empty or existing Marlowe
# directory. Rerunning resumes transfers; remote files are never deleted.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-natedem@login.marlowe.stanford.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/m000091/natedem/security-corpus}"

if [[ $# -ne 0 ]]; then
  echo "Usage: $0 (optional environment: REMOTE_HOST, REMOTE_ROOT)" >&2
  exit 2
fi

echo "Copying project code, corpus data, recovery inventories, and reports"
echo "Destination: ${REMOTE_HOST}:${REMOTE_ROOT}/"
printf -v QUOTED_REMOTE_ROOT '%q' "${REMOTE_ROOT}"
ssh "${REMOTE_HOST}" "mkdir -p -- ${QUOTED_REMOTE_ROOT}"

rsync -a --progress --partial \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.claude/' \
  --exclude '.codex/' \
  --exclude '.agents/' \
  --exclude '.idea/' \
  --exclude '.vscode/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '*.egg-info/' \
  --exclude '/build/' \
  --exclude '/dist/' \
  --exclude '/logs/' \
  --exclude '.DS_Store' \
  --exclude '._*' \
  "${PROJECT_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"

echo "Transfer complete. Recreate the Python environment on Marlowe."
