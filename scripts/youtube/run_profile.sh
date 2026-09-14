#!/usr/bin/env bash
set -euo pipefail
YOUTUBE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
YOUTUBE_PYTHON="${YOUTUBE_PYTHON:-${YOUTUBE_SCRIPT_DIR}/.venv/bin/python}"
YOUTUBE_DATA_DIR="${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}"
# Run this with --check-env on the login node before requesting compute time.
if ! command -v "${YOUTUBE_PYTHON}" >/dev/null 2>&1; then
  printf 'Python interpreter unavailable: %s\nSet YOUTUBE_PYTHON to your profiling environment.\n' "${YOUTUBE_PYTHON}" >&2
  exit 1
fi
"${YOUTUBE_PYTHON}" - "${YOUTUBE_SCRIPT_DIR}/requirements-profile.txt" <<'PY'
import shlex
import sys

try:
    import duckdb
    import pyarrow.parquet
except (ImportError, OSError) as exc:
    print(f"Profiling dependencies unavailable in {sys.executable}: {exc}", file=sys.stderr)
    print("Install them before submitting the job:\n  " + shlex.join([
        sys.executable, "-m", "pip", "install", "-r", sys.argv[1],
    ]), file=sys.stderr)
    raise SystemExit(1)
print(f"Environment OK: {sys.executable}; DuckDB {duckdb.__version__}; PyArrow {pyarrow.__version__}")
PY
if [[ "${1:-}" == "--check-env" ]]; then
  exit 0
fi
exec "${YOUTUBE_PYTHON}" -u "${YOUTUBE_SCRIPT_DIR}/profile.py" \
  --data-dir "${YOUTUBE_DATA_DIR}" "$@"
