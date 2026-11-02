# CyberOpen filtering methods

The [technical report](technical_report/CyberOpen.pdf) describes corpus
construction and dataset statistics. This document maps the implemented stages
to their code. Filtering operates on normalized documents and preserves the
identity and content hash associated with each decision.

## Normalization

Source connectors assemble related records into complete documents and remove
structurally invalid material, such as empty or deleted records. Normalization
computes a SHA-256 content hash and a `cl100k_base` token count. Source-specific
features remain queryable columns rather than implicit selection rules.

Implementation: `src/ingest/connectors/`, `src/ingest/utils.py`, and
`src/ingest/writers.py`.

## Discussions and academic papers

Qwen3-8B assesses candidate discussion threads for security relevance and
technical quality. Candidate preparation limits which records are presented to
the model; an absent assessment must not be interpreted as a rejection.
Citation-expanded academic papers use title/abstract selection before full-text
acquisition and normalization.

Implementation: `src/classify/`, `scripts/classify/`, and `scripts/arxiv/`.

## Transcripts

Qwen3.8-27B-FP8 jointly assesses relevance, utility, and language across transcript
passages. Whole-document candidates require useful, in-scope English labels
throughout their passages. Supporting computing topics qualify without an
explicit security reference. Mixed, uncertain, and unparseable results are
represented separately from eligible documents.

Implementation: `scripts/curation_single/rubric.py`, `score.py`, and `run.py`.
The model revision and inference settings are recorded in each run configuration.

## Web documents

Primus-FineWeb and the explicitly relevant portion of RedSage-CFW undergo exact
baseline-overlap checks and text-damage checks. Repeated nonblank-line character
fractions of at least 0.50 or replacement/control-character fractions of at
least 0.05 trigger recoverable quarantine. Tabs and newlines are excluded from
the latter measure. These parameters are in `scripts/web_filter/policy.json`.

Separate logistic regression models predict relevance and technical substance
from hashed word and adjacent word-pair features. They are trained on 3,249
Qwen-labeled excerpts from 3,141 documents. Document-grouped held-out evaluation
measures agreement with these labels; the production scorers are refit on all
labeled excerpts. Retention requires both scores to be at least 0.80. The
usability predictor is diagnostic and does not determine retention. The scores
do not measure factual correctness.

Implementation: `scripts/web_filter/run.py` and `scripts/web_semantic/`.

## Duplicate handling

Exact-match checks compare content hashes. For web and transcript additions,
MinHash retrieves potential near duplicates, then whole-document sets of
five-lexical-token shingles are compared using Jaccard similarity. A score of at
least 0.90 is required before removing a copy. Existing baseline documents take
precedence, and longer additions take precedence over shorter copies. Every
excluded document must match a retained representative directly.

Transcript additions are checked against the baseline and retained web
collection. MinHash can miss candidate pairs, so the procedure does not guarantee
that all near duplicates have been removed.

Implementation: `scripts/near_dedup/` and `scripts/youtube_finalize/finalize.py`.

## Final assembly and validation

Canonical export normalizes field names and types while retaining source-specific
columns. Explicit exclusion or repair plans bind each affected record to its
identifier and content hash. Final checks validate file inventories, schema,
record identities, content hashes, token counts, and source totals. Evidence
restoration joins already computed assessments without rerunning inference.

Implementation: `scripts/release/export_canonical.py`, `finalize_local.py`,
`normalize_final_schema.py`, `repair_final_metadata.py`, `audit_snapshot.py`,
`restore_filter_evidence.py`, and `attach_filter_evidence.py`.
