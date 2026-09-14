#!/usr/bin/env bash
# Refresh implementation and tokenizer assets without retransferring the corpus.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-natedem@login.marlowe.stanford.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/m000091/natedem/security-corpus}"
if [[ $# -ne 0 ]]; then echo "Usage: $0 (optional REMOTE_HOST, REMOTE_ROOT)" >&2; exit 2; fi
printf -v QUOTED_REMOTE_ROOT '%q' "${REMOTE_ROOT}"
ssh "${REMOTE_HOST}" "mkdir -p -- ${QUOTED_REMOTE_ROOT}"
rsync -a --progress --partial \
  --exclude '.venv/' --exclude 'venv/' --exclude '__pycache__/' \
  --exclude '*.pyc' --exclude '.DS_Store' --exclude '._*' \
  --exclude '.env' --exclude '.env.*' --exclude '.hf-home/' --exclude '.cache/' \
  --exclude '*.egg-info/' \
  "${PROJECT_ROOT}/scripts" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/docs" \
  "${PROJECT_ROOT}/tests" "${PROJECT_ROOT}/pyproject.toml" "${PROJECT_ROOT}/README.md" \
  "${PROJECT_ROOT}/config" \
  "${PROJECT_ROOT}/AGENTS.md" "${REMOTE_HOST}:${REMOTE_ROOT}/"
echo "Code and tokenizer assets copied. Remote data and environments preserved."
