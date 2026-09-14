#!/usr/bin/env bash
# Run this on an allocated CPU node. Install requirements.txt into .venv first.
set -euo pipefail
YOUTUBE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
YOUTUBE_PYTHON="${YOUTUBE_PYTHON:-${YOUTUBE_SCRIPT_DIR}/.venv/bin/python}"
YOUTUBE_DATA_DIR="${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}"

if [[ ! -x "${YOUTUBE_PYTHON}" ]]; then
  echo "Python environment missing: ${YOUTUBE_PYTHON}" >&2
  echo "Follow the setup steps in ${YOUTUBE_SCRIPT_DIR}/README.md" >&2
  exit 1
fi

exec "${YOUTUBE_PYTHON}" -u "${YOUTUBE_SCRIPT_DIR}/download.py" \
  --output-dir "${YOUTUBE_DATA_DIR}" "$@"
