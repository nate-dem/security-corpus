#!/usr/bin/env bash
# Explicit retained baseline only; never glob the larger recovery working set.
set -euo pipefail
WEB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEB_PROJECT_ROOT="$(cd -- "${WEB_SCRIPT_DIR}/../.." && pwd)"
cd "${WEB_PROJECT_ROOT}"
exec bash scripts/web_corpora/run.sh compare \
  --baseline \
    data/filtering/v3/qwen_qa_kept \
    data/final-no-chunk/academic_papers/arxiv_cs_cr_full.parquet \
    data/final-no-chunk/academic_papers/arxiv_citation_qwen_kept_full.parquet \
    data/final-no-chunk/normalized/source_id=cloudtrail-flaws \
    data/final-no-chunk/normalized/source_id=sigma \
    data/training-clean-v2/normalized/source_id=nvd \
    data/training-clean-v2/normalized/source_id=cisa-kev \
    data/training-clean-v2/normalized/source_id=mitre-attack \
    data/training-clean-v2/normalized/source_id=mitre-cwe \
    data/training-clean-v2/normalized/source_id=capec \
  "$@"
