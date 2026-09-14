#!/usr/bin/env bash
set -euo pipefail
ENGLISH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENGLISH_PROJECT_ROOT="$(cd -- "${ENGLISH_SCRIPT_DIR}/../.." && pwd)"
ENGLISH_PYTHON="${ENGLISH_PYTHON:-${ENGLISH_PROJECT_ROOT}/scripts/web_corpora/.venv/bin/python}"
export PYTHONPATH="${ENGLISH_PROJECT_ROOT}/src:${ENGLISH_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"${ENGLISH_PYTHON}" - <<'PY'
import importlib.metadata
from ingest.utils import compute_token_count
for name, expected in {'duckdb':'1.5.2', 'pyarrow':'23.0.1', 'tiktoken':'0.12.0'}.items():
    if importlib.metadata.version(name) != expected:
        raise RuntimeError(f'Expected {name}=={expected}; use the existing web CPU environment')
assert compute_token_count('Offline tokenizer check.') > 0
print('English preparation environment OK; CPU only', flush=True)
PY
if [[ "${1:-}" == '--check-env' ]]; then exit 0; fi
cd "${ENGLISH_PROJECT_ROOT}"
exec "${ENGLISH_PYTHON}" -u -m scripts.youtube.prepare_english \
  --data-dir "${YOUTUBE_DATA_DIR:-/scratch/m000091/${USER}/youtube-transcripts}" \
  --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
