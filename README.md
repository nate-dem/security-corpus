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

Audit normalized output before making downstream filtering decisions:

```bash
python scripts/audit_normalized_corpus.py
python scripts/audit_normalized_corpus.py --output-dir reports/normalized_audit
```

The audit writes a Markdown report and CSV tables covering per-source token counts, length distributions, missing required fields, exact duplicates, Q&A quality signals, vulnerability fields, CloudTrail session outliers, and license totals.

## Security Scope

Security Scope is the research interface over the final cleaned corpus. It provides a local web UI plus shell-native retrieval commands: `search`, `ls`, `cat`, `head`, `grep`, result handles, and an optional MCP wrapper for agent retrieval.

The historical package and command names remain available as compatibility aliases. You can use either `security-scope` or `securityclip` for the CLI.

Set the index once:

```bash
export SECURITYCLIP_INDEX=/scratch/m000091-pm05/natedem/securityclip-index/v1
```

Then use it without repeating `--index`:

```bash
security-scope ls /
security-scope search "CVE-2021-44228" -n 10
security-scope grep -i "alphamissense" /papers/ --limit 5
```

See [docs/securityscope.md](docs/securityscope.md) for the full usage guide.

The optional web UI runs on top of the same index:

```bash
python -m pip install -e ".[web]"
cd web && npm install && npm run build && cd ..
security-scope-web --host 127.0.0.1 --port 8765
```

Build the first downstream training-clean export:

```bash
python scripts/build_training_clean_v1.py
```

This writes filtered Parquet to `data/training-clean-v1/normalized/` and summary reports to `reports/training-clean-v1/`. The v1 policy drops invalid/empty records, Q&A records with no answers/comments plus non-positive score, and exact duplicate content.

Run tests:

```bash
pytest                      # unit tests
pytest -m data_quality      # validate ingested Parquet output
```

## Output

Parquet files written to `data/{source}/normalized/`, Hive-partitioned by `source_id`. Each record includes content, metadata, quality signals, content hash (for dedup), and token count.
