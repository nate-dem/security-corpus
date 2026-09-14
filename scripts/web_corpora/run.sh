#!/usr/bin/env bash
# Run from the source checkout, not Slurm's temporary copy of job.sbatch.
set -euo pipefail
WEB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEB_PROJECT_ROOT="$(cd -- "${WEB_SCRIPT_DIR}/../.." && pwd)"
WEB_PYTHON="${WEB_PYTHON:-${WEB_SCRIPT_DIR}/.venv/bin/python}"
WEB_DATA_DIR="${WEB_DATA_DIR:-/scratch/m000091/${USER}/web-corpora}"
export PYTHONPATH="${WEB_PROJECT_ROOT}/src:${WEB_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if ! command -v "${WEB_PYTHON}" >/dev/null 2>&1; then
  printf 'Python unavailable: %s\nCreate scripts/web_corpora/.venv and install requirements.txt first.\n' "${WEB_PYTHON}" >&2
  exit 1
fi
"${WEB_PYTHON}" - "${WEB_SCRIPT_DIR}/requirements.txt" <<'PY'
import importlib.metadata
from pathlib import Path
import shlex
import sys

try:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    for line in Path(sys.argv[1]).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, version = line.split('==')
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Expected {name}=={version}")
    import duckdb
    import pyarrow.parquet
    from ingest.utils import compute_token_count
    assert compute_token_count("Environment check.") > 0
except Exception as exc:
    print(f"Environment check failed in {sys.executable}: {exc}", file=sys.stderr)
    print("Install before requesting compute:\n  " + shlex.join([
        sys.executable, '-m', 'pip', 'install', '-r', sys.argv[1],
    ]), file=sys.stderr)
    raise SystemExit(1)
print(f"Environment OK: {sys.executable}; packaged cl100k_base available offline", flush=True)
PY

WEB_PHASE="${1:---check-env}"
if [[ $# -gt 0 ]]; then shift; fi
cd "${WEB_PROJECT_ROOT}"
case "${WEB_PHASE}" in
  --check-env) exit 0 ;;
  download)
    exec "${WEB_PYTHON}" -u -m scripts.web_corpora.download --data-dir "${WEB_DATA_DIR}" "$@" ;;
  profile)
    exec "${WEB_PYTHON}" -u -m scripts.web_corpora.profile --data-dir "${WEB_DATA_DIR}" \
      --workers "${SLURM_CPUS_PER_TASK:-4}" "$@" ;;
  compare)
    exec "${WEB_PYTHON}" -u -m scripts.web_corpora.compare --data-dir "${WEB_DATA_DIR}" \
      --workers "${SLURM_CPUS_PER_TASK:-4}" "$@" ;;
  compare-baseline)
    exec bash scripts/web_corpora/compare_baseline.sh "$@" ;;
  full)
    if [[ $# -ne 0 ]]; then echo 'full takes no extra arguments; use download/profile/compare for overrides.' >&2; exit 2; fi
    "${WEB_PYTHON}" -u -m scripts.web_corpora.download --data-dir "${WEB_DATA_DIR}"
    "${WEB_PYTHON}" -u -m scripts.web_corpora.profile --data-dir "${WEB_DATA_DIR}" \
      --workers "${SLURM_CPUS_PER_TASK:-4}"
    exec bash scripts/web_corpora/compare_baseline.sh ;;
  *) echo "Usage: $0 {--check-env|download|profile|compare|compare-baseline|full} [arguments]" >&2; exit 2 ;;
esac
