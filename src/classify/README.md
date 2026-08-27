# V3 Filtering Sidecars

This package is for downstream filtering scores, not ingestion. It never
modifies normalized corpus Parquet. Every output is a sidecar keyed by:

- `source_id`
- `record_id`
- `content_hash`

The expected flow is:

1. Read labeled examples from a CSV or Parquet file.
2. Train a TF-IDF plus logistic regression model for one label task.
3. Score normalized corpus Parquet into sidecar Parquet.
4. Use those sidecars in DuckDB to select Qwen candidates or final mixtures.

The first tasks are:

- `qa_quality`
- `qa_quality_binary`
- `security_relevance`
- `quality`

For V3 production prefiltering, use `qa_quality_binary`. It answers whether a
QA/social record is high enough quality to spend Qwen inference on. Its labels
must be numeric `qa_quality_binary_label` values in `0..1`, and its score is
the expected binary quality from class probabilities. The older 4-class
`qa_quality` task remains useful for analysis and auditing. QA quality scoring
is restricted to `stackoverflow`, `stackexchange-*`, and `reddit-*` by default.
Do not use the generic `quality` task for Sigma or CloudTrail.

Qwen support lives in `classify.qwen` and `scripts/classify/score_qwen_vllm.py`.
The script uses local vLLM only when run without `--dry-run`; importing the
module and rendering dry-run prompts does not import vLLM or load model weights.
The scorer filters sources before prompt validation; use `--qa-sources` for
QA/social runs and `--source-id arxiv` for arXiv runs against combined
normalized data. On Hive-partitioned normalized directories it selects matching
`source_id=...` partition files before validating the task schema. It fails fast
when task-required prompt fields are missing or blank in the selected rows.

Structural artifact checks for Sigma and CloudTrail live in
`classify.artifact_quality` and write sidecars with explicit review flags. They
are not generic prose classifiers.
