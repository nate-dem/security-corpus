# Open Security Corpus

This repo contains a data ingestion pipeline for building a security-domain mid-training corpus. It pulls from sources such as vulnerability databases, knowledge bases, detection rules, Q&A archives, academic papers, security blog posts, and security conference transcripts. It then normalizes every record into a canonical schema, and writes Parquet output partitioned by source.

## Sources

| Source | Schema | Connector |
|---|---|---|
| NVD (CVEs) | `VulnerabilityData` | `nvd.py` |
| CISA KEV | `VulnerabilityData` | `cisa_kev.py` |
| MITRE ATT&CK | `MitreData` | `mitre_attack.py` |
| MITRE CWE | `MitreData` | `mitre_cwe.py` |
| CAPEC | `MitreData` | `capec.py` |
| BRON | `NormalizedData` | `bron.py` |
| Sigma Rules | `DetectionRuleData` | `sigma.py` |
| Stack Exchange (InfoSec, RE, Crypto, Tor) | `QAThreadData` | `stackexchange/` |
| Stack Overflow (security tags) | `QAThreadData` | `stackexchange/stackoverflow.py` |
| Reddit (22 security subreddits) | `QAThreadData` | `reddit/` |

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Usage

Ingest a source:

```bash
python scripts/ingest_{source}.py
python scripts/ingest_stackexchange.py {site}    # infosec, reverseengineering, crypto, tor
python scripts/ingest_stackoverflow.py           # streams from .7z archive
python scripts/ingest_reddit.py {subreddit}      # or --all for all 22 subreddits
```

Run tests:

```bash
pytest                      # unit tests
pytest -m data_quality      # validate ingested Parquet output
```

## Output

Parquet files written to `data/{source}/normalized/`, Hive-partitioned by `source_id`. Each record includes content, metadata, quality signals, content hash (for dedup), and token count.
