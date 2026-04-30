# Open Security Corpus

This repo contains a data ingestion pipeline for building a security-domain mid-training corpus. It pulls from sources such as vulnerability databases, knowledge bases, detection rules, Q&A archives, academic papers, security blog posts, and security conference transcripts. It then normalizes every record into a canonical schema, and writes Parquet output partitioned by source.

## Sources

| Source | Schema | Connector |
|---|---|---|
| NVD (CVEs) | `VulnerabilityData` | `nvd.py` |
| CISA KEV | `VulnerabilityData` | `cisa_kev.py` |
| MITRE ATT&CK | `KnowledgeBaseData` | `mitre_attack.py` |
| MITRE CWE | `KnowledgeBaseData` | `mitre_cwe.py` |
| CAPEC | `KnowledgeBaseData` | `capec.py` |
| BRON | `KnowledgeBaseData` | `bron.py` |
| Sigma Rules | `DetectionRuleData` | `sigma.py` |
| Stack Exchange (InfoSec, RE, Crypto) | `QAThreadData` | `stackexchange/` |

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Usage

Ingest a source:

```bash
python scripts/ingest_{source}.py
```

Run tests:

```bash
pytest                      # unit tests
pytest -m data_quality      # validate ingested Parquet output
```

## Output

Parquet files written to `data/{source}/normalized/`, partitioned by `source_id`. Each record includes content, metadata, quality signals, content hash (for dedup), and token count.
