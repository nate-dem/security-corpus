# Open Security Corpus

This repo contains a data ingestion pipeline for building a security-domain mid-training corpus. It pulls from sources such as vulnerability databases, knowledge bases, detection rules, Q&A archives, academic papers, security blog posts, and security conference transcripts. It then normalizes every record into a canonical schema, and writes Parquet output partitioned by source.

## Sources

| Source | Schema | Connector |
|---|---|---|
| NVD (CVEs) | `VulnerabilityData` | `vulnerability/nvd.py` |
| CISA KEV | `VulnerabilityData` | `vulnerability/cisa_kev.py` |
| GitHub Advisory Database | `VulnerabilityData` | `vulnerability/github_advisory.py` |
| MITRE ATT&CK | `MitreData` | `knowledge/mitre_attack.py` |
| MITRE CWE | `MitreData` | `knowledge/mitre_cwe.py` |
| CAPEC | `MitreData` | `knowledge/capec.py` |
| BRON | `NormalizedData` | `knowledge/bron.py` |
| Sigma Rules | `DetectionRuleData` | `detection/sigma.py` |
| Stack Exchange (InfoSec, RE, Crypto, Tor) | `QAThreadData` | `stackexchange/` |
| Stack Overflow (security tags) | `QAThreadData` | `stackexchange/stackoverflow.py` |
| Reddit (22 security subreddits) | `QAThreadData` | `reddit/` |
| CloudTrail (flaws.cloud) | `CloudTrailSessionData` | `logs/cloudtrail.py` |
| YouTube transcripts | `TranscriptData` | `transcripts/youtube_transcripts.py` |
| arXiv papers | `AcademicPaperData` | `arxiv/` |

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Usage

Ingest a source with the repo-local wrapper:

```bash
python scripts/ingest.py list
python scripts/ingest.py nvd
python scripts/ingest.py cisa-kev
python scripts/ingest.py stackexchange infosec   # infosec, reverseengineering, crypto, tor
python scripts/ingest.py stackoverflow           # streams from .7z archive
python scripts/ingest.py reddit netsec           # or: python scripts/ingest.py reddit --all
python scripts/ingest.py cloudtrail-flaws
```

After `pip install -e ".[dev]"`, the same commands are available as:

```bash
security-corpus-ingest list
security-corpus-ingest nvd
```

The command implementation lives in `src/ingest/commands.py`; `scripts/ingest.py` is only a thin wrapper so direct repo usage and installed CLI usage share the same paths and behavior.

Run tests:

```bash
pytest                      # unit tests
pytest -m data_quality      # validate ingested Parquet output
```

## Output

Parquet files written to `data/{source}/normalized/`, Hive-partitioned by `source_id`. Each record includes content, metadata, quality signals, content hash (for dedup), and token count.
