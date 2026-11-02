---
pretty_name: CyberOpen
task_categories:
- text-generation
tags:
- cybersecurity
- continued-pretraining
- research-corpus
size_categories:
- 1M<n<10M
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: data/*/*.parquet
dataset_info:
  features:
  - name: source_id
    dtype: string
  - name: source_record_id
    dtype: string
  - name: record_id
    dtype: string
  - name: content
    dtype: large_string
  - name: title
    dtype: string
  - name: content_length
    dtype: int64
  - name: content_hash
    dtype: string
  - name: ingested_at
    dtype: timestamp[us, tz=UTC]
  - name: published_at
    dtype: timestamp[us, tz=UTC]
  - name: source_url
    dtype: string
  - name: license
    dtype: string
  - name: raw
    dtype: string
  - name: score
    dtype: int64
  - name: answer_count
    dtype: int64
  - name: has_accepted_answer
    dtype: bool
  - name: closed
    dtype: bool
  - name: tags
    list: string
  - name: qwen_security_relevance
    dtype: int16
  - name: qwen_quality
    dtype: int16
  - name: qwen_model_should_keep
    dtype: bool
  - name: qwen_should_keep
    dtype: bool
  - name: qwen_reason
    dtype: string
  - name: qwen_parse_status
    dtype: string
  - name: qwen_keep_policy
    dtype: string
  - name: qwen_keep_policy_passed
    dtype: bool
  - name: qwen_keep_policy_reason
    dtype: string
  - name: qwen_model
    dtype: string
  - name: qwen_prompt_version
    dtype: string
  - name: qwen_scored_at
    dtype: string
  - name: qwen_shard_id
    dtype: string
  - name: qwen_task
    dtype: string
  - name: qwen_input_kind
    dtype: string
  - name: qwen_raw_response
    dtype: string
  - name: arxiv_id
    dtype: string
  - name: source_format
    dtype: string
  - name: authors
    list: string
  - name: abstract
    dtype: string
  - name: categories
    list: string
  - name: primary_category
    dtype: string
  - name: doi
    dtype: string
  - name: journal_ref
    dtype: string
  - name: event_count
    dtype: int64
  - name: session_duration_seconds
    dtype: int64
  - name: source_ip
    dtype: string
  - name: principals
    list: string
  - name: actions
    list: string
  - name: aws_services
    list: string
  - name: regions
    list: string
  - name: has_errors
    dtype: bool
  - name: rule_id
    dtype: string
  - name: rule_format
    dtype: string
  - name: rule_level
    dtype: string
  - name: rule_source
    dtype: string
  - name: framework
    dtype: string
  - name: category_id
    dtype: string
  - name: cve_id
    dtype: string
  - name: severity
    dtype: string
  - name: cvss_score
    dtype: float64
  - name: cwe_ids
    list: string
  - name: exploited_in_wild
    dtype: bool
  - name: kind
    dtype: string
  - name: dataset_revision
    dtype: string
  - name: source_shard
    dtype: string
  - name: source_row
    dtype: int64
  - name: source_copies
    dtype: int64
---

# CyberOpen

CyberOpen is a cybersecurity corpus for continued pretraining of language models.
It brings together vulnerability descriptions, detection rules, technical
discussions, academic papers, operational logs, web documents, and video
transcripts. The corpus also includes supporting knowledge in operating systems,
networking, cloud infrastructure, and software engineering.

The corpus contains **1,474,511 documents** and **2,808,027,544 tokens**, counted
with `tiktoken`'s `cl100k_base` encoding. Source-specific processing assembles
related material into self-contained documents with common metadata and
source-specific fields. Researchers can use these fields to inspect the corpus
and construct different training mixtures.

The [technical report](technical_report/CyberOpen.pdf) describes source selection, filtering, deduplication, and dataset analysis.

## Load the dataset

Authenticate with
Hugging Face before loading a private repository.

```python
from datasets import load_dataset

corpus = load_dataset(
    "cyberopen-research/CyberOpen",
    split="train",
    streaming=True,
)

document = next(iter(corpus))
print(document["source_id"])
print(document["content"][:500])
```

CyberOpen has one configuration (`default`) and one split (`train`). No validation
or test partitions are defined. Documents are plain text with Markdown, code,
YAML, or JSON where appropriate; they are not formatted as instruction-response
pairs. To reproduce a specific dataset version, pass its Hub commit hash as
`revision` when loading.

## Source composition

| Source | Documents | cl100k_base tokens |
|---|---:|---:|
| `arxiv` | 63,246 | 908,999,231 |
| `capec` | 556 | 68,938 |
| `cisa-kev` | 1,481 | 52,045 |
| `cloudtrail-flaws` | 56,043 | 246,135,664 |
| `mitre-attack` | 2,176 | 389,635 |
| `mitre-cwe` | 1,286 | 103,783 |
| `nvd` | 313,278 | 25,206,290 |
| `primus-fineweb` | 536,414 | 739,086,525 |
| `reddit-antivirus` | 4,774 | 5,841,338 |
| `reddit-asknetsec` | 9,660 | 14,210,930 |
| `reddit-blueteamsec` | 299 | 341,016 |
| `reddit-bugbounty` | 1,532 | 1,657,943 |
| `reddit-cloudflare` | 601 | 771,414 |
| `reddit-computersecurity` | 606 | 677,185 |
| `reddit-computerviruses` | 1,359 | 1,468,016 |
| `reddit-cryptography` | 3,309 | 4,764,171 |
| `reddit-cyberlaws` | 44 | 85,818 |
| `reddit-cybersecurity` | 10,753 | 25,833,577 |
| `reddit-cybersecurity_help` | 4,880 | 7,140,953 |
| `reddit-cybersecurityadvice` | 621 | 798,219 |
| `reddit-hacking` | 6,993 | 12,271,418 |
| `reddit-malware` | 1,174 | 1,444,121 |
| `reddit-netsec` | 1,153 | 3,731,059 |
| `reddit-netsecstudents` | 1,344 | 1,831,192 |
| `reddit-phishing` | 1,211 | 1,521,497 |
| `reddit-reverseengineering` | 518 | 1,132,953 |
| `reddit-security` | 2,061 | 2,564,342 |
| `reddit-tor` | 5,907 | 7,554,522 |
| `reddit-vpn` | 3,275 | 4,161,438 |
| `redsage-cfw` | 184,930 | 255,090,274 |
| `sigma` | 3,706 | 1,678,011 |
| `stackexchange-crypto` | 16,456 | 16,722,872 |
| `stackexchange-infosec` | 35,394 | 36,096,276 |
| `stackexchange-reverseengineering` | 3,369 | 4,571,347 |
| `stackexchange-tor` | 1,812 | 1,285,592 |
| `stackoverflow` | 98,174 | 126,140,592 |
| `youtube-commons` | 94,116 | 346,597,347 |
| **Total** | **1,474,511** | **2,808,027,544** |

## Document structure

Stack Exchange questions include their answers and code blocks. Reddit
discussions retain nested comments. Sigma documents contain rule descriptions
and complete YAML, while CloudTrail documents contain ordered JSON events
grouped into sessions. Structured knowledge sources retain normalized
descriptions, source identifiers, and metadata.

| Field | Description |
|---|---|
| `content` | Assembled document text. |
| `source_id` | Source collection. |
| `source_record_id`, `record_id` | Source-level and namespaced document identifiers. |
| `content_length` | Number of `cl100k_base` tokens in the document. |
| `content_hash` | SHA-256 of the UTF-8 document content. |
| `title`, `source_url`, `published_at` | Document title, source URL, and publication time, where available. |
| `ingested_at` | Recorded normalization time. |
| `license` | Source-specific license declaration or terms label. |
| `raw` | Serialized source material, where retained. |

Additional fields contain information specific to each source, such as Q&A
scores and answer counts, filtering assessments, vulnerability severity, rule
YAML, academic-paper metadata, and CloudTrail session features. The dataset
configuration declares all 65 fields. Fields that do not apply to
a source, or whose values are unavailable, are returned as null.

## Files

Training documents are stored in 10,621 Parquet files under
`data/`. The dataset configuration loads only these files. `SHA256SUMS` contains
file checksums for verifying downloads.

## Filtering metadata

The [filtering metadata supplement](https://huggingface.co/datasets/cyberopen-research/CyberOpen/blob/main/metadata/filtering/README.md) provides
original automated assessments for 721,344 retained web documents and 94,116
retained YouTube transcripts, including 138,208 passage assessments. The tables
contain scores, labels, model-generated explanations and research provenance,
linked to training documents by `record_id` and `content_hash`.

The supplement includes a field guide, selection rules and the transcript
classification rubric and inference settings. It covers retained documents;
it does not contain the full rejected candidate population or human-review
labels. These files are stored under `metadata/filtering/` and are separate
from the default training-data configuration.

## Use and limitations

CyberOpen supports research on how training-data composition affects
cybersecurity capabilities. Corpus construction and dataset statistics alone
do not establish improvements in trained-model performance. Automated filtering
can retain weak material or exclude useful material, and near duplicates may
remain. Filtering scores do not measure factual correctness.

License information is recorded per document in the `license` field. Terms vary
across sources; a source's license label should not be treated as applying to
the entire corpus.
