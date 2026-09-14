#!/bin/bash
#SBATCH --job-name=qwen-recovery-smoke
#SBATCH --account=marlowe-m000091-pm05
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/qwen_smoke/%j.out
#SBATCH --error=logs/qwen_smoke/%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/m000091/natedem/security-corpus}"
cd "${PROJECT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PROJECT_DIR}/src"
SMOKE_ROOT="data/filtering/smoke-${SLURM_JOB_ID}"
mkdir -p "${SMOKE_ROOT}"
nvidia-smi
python -m pip check

# One short and one longest document per source exercises real input shapes.
# This is an inference smoke sample, not the researcher's human quality audit.
python - "${SMOKE_ROOT}" <<'PY'
from pathlib import Path
import sys
import duckdb

root = Path(sys.argv[1])
with duckdb.connect() as con:
    for name, source in (
        ('qa', 'data/filtering/v4/qa_to_score/**/*.parquet'),
        ('citations', 'data/filtering/v4/citation_abstract_universe.parquet'),
    ):
        con.read_parquet(source, union_by_name=True, hive_partitioning=False).create_view('input_records', replace=True)
        order = 'content_length' if name == 'qa' else 'length(abstract)'
        con.sql(f'''
            SELECT * FROM input_records
            QUALIFY row_number() OVER (PARTITION BY source_id ORDER BY {order}, record_id) = 1
                 OR row_number() OVER (PARTITION BY source_id ORDER BY {order} DESC, record_id) = 1
        ''').write_parquet(str(root / f'{name}.parquet'), compression='zstd')
PY

MODEL_REVISION="${MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
for name in qa citations; do
  task=qa
  dtype=bfloat16
  if [[ "${name}" == citations ]]; then task=arxiv-abstract; dtype=float16; fi
  python scripts/classify/score_qwen_vllm.py \
    --input "${SMOKE_ROOT}/${name}.parquet" --task "${task}" \
    --output-dir "${SMOKE_ROOT}/${name}-decisions" \
    --model Qwen/Qwen3-8B --model-revision "${MODEL_REVISION}" --dtype "${dtype}"
  python scripts/release/audit_qwen_coverage.py \
    --corpus "${SMOKE_ROOT}/${name}.parquet" \
    --decisions "${SMOKE_ROOT}/${name}-decisions" \
    --expected-model Qwen/Qwen3-8B --expected-revision "${MODEL_REVISION}" \
    --output "${SMOKE_ROOT}/${name}-coverage.json"
done

echo "Smoke outputs and coverage reports: ${SMOKE_ROOT}"
