#!/usr/bin/env bash
#SBATCH --job-name=curation-inputs
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/curation/%x-%j.log
set -euo pipefail
cd "${PROJECT_DIR:?Set PROJECT_DIR to the repository root}"
export PYTHONPATH="${PWD}/src:${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
"${YOUTUBE_FILTER_CPU_PYTHON:-${PWD}/scripts/web_corpora/.venv/bin/python}" - <<'PYTHON'
import importlib.metadata
from ingest.utils import compute_token_count
for name, version in {"duckdb": "1.5.2", "pyarrow": "23.0.1", "tiktoken": "0.12.0"}.items():
    assert importlib.metadata.version(name) == version, f"Expected {name}=={version}"
assert compute_token_count("Offline tokenizer check.") > 0
PYTHON
exec "${YOUTUBE_FILTER_CPU_PYTHON:-${PWD}/scripts/web_corpora/.venv/bin/python}" -u -m scripts.curation.prepare \
  --youtube "${YOUTUBE_DATA_DIR:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/youtube-transcripts}/english-candidates-v1" \
  --web-root "${WEB_DATA_DIR:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/web-corpora}" \
  --output-dir "${CURATION_PREPARED_DIR:-${CYBEROPEN_WORK_DIR:-${PWD}/data}/curation/inputs-v1}" \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
