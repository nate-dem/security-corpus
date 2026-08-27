# Security Corpus

Reproducible ingestion, normalization, and filtering code for a cybersecurity
continued-pretraining corpus. Records are stored as Parquet documents with
stable IDs, SHA-256 content hashes, `cl100k_base` token counts, source URLs,
and per-record license labels.

This recovery branch contains only corpus work. SecurityClip, benchmarks, and
YouTube have been archived outside the repository. RegMix, FineWeb, the unused
logistic-regression filter, BRON, and GitHub Advisory code are not part of the
current release.

## Recovery status

The laptop checkpoints are preserved, but the publishable corpus is not yet
finished. The current reproducible restart points are:

| Checkpoint | Records | Tokens | Status |
|---|---:|---:|---|
| QA/social exact-deduplicated universe | 1,617,344 | 1,128,832,782 | Full Qwen re-score required |
| Recovered full papers | 63,340 | 1,299,306,873 | Legacy checkpoint; re-extract and re-score citation abstracts |
| Structured/artifact sources | 408,596 | 275,145,829 | Structurally cleaned; source-policy decisions remain |

These counts are inputs, not a final corpus total. They must not be quoted as a
released token count until Qwen filtering, paper re-extraction, artifact
transformation, exact deduplication, and license review are complete.

Machine-readable recovery reports live under `reports/recovery/`:

- `arxiv/audit.json` inventories paper checkpoints and restart IDs.
- `structured-v1/manifest.json` records structural cleaning and source counts.
- `data/filtering/v4/manifest.json` records the QA universe and Qwen queue.

## Sources in scope

- NVD and CISA Known Exploited Vulnerabilities
- MITRE ATT&CK, CWE, and CAPEC
- Sigma detection rules
- flaws.cloud CloudTrail sessions
- Stack Overflow and four Stack Exchange sites
- security-focused Reddit communities
- arXiv cs.CR papers and citation-expanded arXiv papers

YouTube is deliberately deferred. BRON and GitHub Advisory are also deferred.

## Install and test

Python 3.11 or newer is required. Runtime dependencies are pinned in
`pyproject.toml`; the immutable `cl100k_base` vocabulary is packaged locally so
token counts do not depend on a runtime download.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For local Qwen/vLLM scoring on a compatible GPU environment:

```bash
pip install -e ".[qwen]"
```

## Core workflows

List and run registered ingestion sources:

```bash
python scripts/ingest.py list
python scripts/ingest.py nvd
python scripts/ingest.py stackexchange infosec
```

Rebuild the policy-neutral structured checkpoint:

```bash
python scripts/build_structured_checkpoint.py --overwrite
```

Rebuild the exact-deduplicated Qwen inputs:

```bash
python scripts/classify/build_qa_qwen_universe.py --overwrite
python scripts/classify/build_citation_qwen_universe.py --overwrite
```

Audit recovered paper work and generate Marlowe restart lists:

```bash
python scripts/arxiv/audit_recovery.py --overwrite
```

The filtering protocol and Slurm commands are documented in
[`docs/filtering.md`](docs/filtering.md). The bounded transfer, scoring,
re-extraction, and laptop-checkpoint procedure is in
[`docs/marlowe_recovery.md`](docs/marlowe_recovery.md). No script submits a
cluster job.

Audit the exact Parquet files proposed for publication:

```bash
python scripts/release/audit_source_licenses.py PATH [PATH ...] \
  --output reports/release/source_license_audit.json
```

The command exits nonzero when any records need permission, institutional
review, or license-metadata repair. See
[`docs/source_licenses.md`](docs/source_licenses.md).

## Data contract

Every normalized record has these common fields:

- `source_id`, `source_record_id`, `record_id`
- `content`, `title`, `content_length`, `content_hash`
- `ingested_at`, `published_at`, `source_url`, `license`

Source-family subclasses add queryable fields for vulnerabilities, knowledge
bases, Q&A threads, academic papers, detection rules, and CloudTrail sessions.
Quality decisions are sidecars keyed by `(source_id, record_id, content_hash)`;
normalized source Parquet is immutable.

## Publication boundary

Code correctness and data-redistribution permission are separate questions.
The `license` column preserves source-level terms, but it is not legal advice or
a grant of redistribution rights. A Hugging Face release must include a source
card, attribution/provenance, license grouping, and any exclusions required by
the final license review. A repository code license must also be chosen before
the public GitHub release.
