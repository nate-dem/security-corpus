# CyberOpen

<p align="center"><img src="assets/cyberopen-logo.png" alt="CyberOpen logo" width="150"></p>

CyberOpen is a cybersecurity corpus for continued pretraining of language models.
It combines vulnerability descriptions, detection rules, practitioner discussions,
academic papers, operational logs, web documents, and video transcripts, including
supporting knowledge in operating systems, networking, cloud infrastructure, and
software engineering.

The corpus contains **1,474,511 documents** and **2,808,027,544 tokens**, counted
with `tiktoken`'s `cl100k_base` encoding. Documents have common identity,
provenance, license, hash, and token-count fields, together with source-specific
metadata for constructing different training mixtures.

- [Technical report](docs/technical_report/CyberOpen.pdf)
- [Report LaTeX and figures](docs/technical_report/)
- [Dataset card and source statistics](docs/dataset_card.md)
- [Dataset on Hugging Face](https://huggingface.co/datasets/cyberopen-research/CyberOpen)
- [Filtering methods](docs/filtering.md)
- [Filtering metadata and selection policy](docs/filtering_metadata.md)
- [Dataset packaging and loading](docs/dataset_packaging.md)

## Corpus composition

| Source group | Documents | Tokens |
|---|---:|---:|
| arXiv papers | 63,246 | 908,999,231 |
| Primus-FineWeb | 536,414 | 739,086,525 |
| YouTube transcripts | 94,116 | 346,597,347 |
| RedSage-CFW | 184,930 | 255,090,274 |
| CloudTrail sessions | 56,043 | 246,135,664 |
| Stack Overflow | 98,174 | 126,140,592 |
| Reddit | 62,074 | 99,803,122 |
| Stack Exchange | 57,031 | 58,676,087 |
| NVD, CISA KEV and MITRE | 318,777 | 25,820,691 |
| Sigma rules | 3,706 | 1,678,011 |
| **Total** | **1,474,511** | **2,808,027,544** |

The repository contains the code and documentation. Training Parquet files are
distributed separately; they are not stored in Git.

## Document construction

Questions and answers are combined into complete discussion documents while
preserving code blocks. Reddit discussions retain nested comment structure.
Sigma documents include complete rule YAML, and CloudTrail sessions contain
ordered JSON events. Structured knowledge sources retain normalized descriptions,
source identifiers, and metadata.

Normalization and training-data selection are separate. Selection uses source
rules, model assessments, and duplicate checks; scores and decisions are stored
separately from normalized documents. Details and limitations are described in
the technical report.

## Installation

Python 3.11 or newer is required. Install the ingestion package and test tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

CPU web filtering and near-duplicate detection have separate pinned dependencies:

```bash
python -m pip install -r scripts/web_semantic/requirements.txt
python -m pip install -r scripts/near_dedup/requirements.txt
```

GPU classification uses separate environments: the Q&A classifier dependencies
are in the `qwen` extra, and transcript runtime dependencies are in
`scripts/curation/requirements-next.txt`. Hugging Face packaging dependencies are
isolated in `scripts/release/requirements-huggingface.txt` so they do not replace
the ingestion or model-runtime versions.

## Code organization

| Location | Purpose |
|---|---|
| `src/ingest/` | Streaming readers, source connectors, document assembly, and Parquet writers. |
| `src/classify/`, `scripts/classify/` | Q&A and abstract classification, response parsing, and score sidecars. |
| `scripts/arxiv/` | Paper metadata, citation expansion, downloads, and full-text normalization. |
| `scripts/youtube/`, `scripts/web_corpora/` | Source acquisition, profiling, and preparation. |
| `scripts/curation_single/` | Transcript relevance, utility, and language assessment. |
| `scripts/curation/`, `scripts/youtube_filter/` | Shared scoring, parser, model-runtime, and preparation dependencies. |
| `scripts/web_filter/`, `scripts/web_semantic/` | Web text-quality checks and logistic regression scoring. |
| `scripts/near_dedup/` | MinHash candidate retrieval, verified Jaccard comparisons, and duplicate lineage. |
| `scripts/youtube_finalize/`, `scripts/youtube_snapshot/` | Assemble completed transcript outputs, audit, and deduplicate additions. |
| `scripts/release/` | Canonical export, metadata normalization, source-term records, integrity checks, and Hub packaging. |
| `tests/` | Fixture-based tests for assembly, filtering, deduplication, finalization, and upload configuration. |

List the available ingestion commands:

```bash
python scripts/ingest.py list
```

Pipeline stages expose their arguments through `--help`, for example:

```bash
python -m scripts.curation_single.run --help
python -m scripts.web_semantic.train --help
python -m scripts.near_dedup.run --help
python -m scripts.youtube_finalize.finalize --help
```

Configure input/output locations and scheduler settings for your environment.
Set `PROJECT_DIR` to the repository root and `CYBEROPEN_WORK_DIR` to a storage
directory for large intermediate outputs; the latter defaults to `data/` under
the repository. Set your Slurm account with `SBATCH_ACCOUNT` and adapt partition
and resource requests to your cluster. Evidence restoration accepts
`CYBEROPEN_WEB_AUDIT` and `CYBEROPEN_FINALIZED` for the completed audit directories.
The source inventories and frozen model/selection configurations must correspond
to the artifacts being processed; checksum mismatches intentionally stop a run.

## Validation

```bash
python -m pytest -m 'not data_quality'
```

Tests marked `data_quality` require local corpus data. Packaging validation checks
file hashes, schema fields, source totals, and the files selected by Hugging Face
Datasets; see [dataset packaging](docs/dataset_packaging.md).

## Source terms

Each document carries a source-specific `license` value. The corpus combines
materials with different terms; a source's license label does not apply to the
whole collection. See [source-license metadata](docs/source_licenses.md).
