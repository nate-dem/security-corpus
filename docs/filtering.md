# Filtering and release protocol

## Invariants

Filtering never overwrites normalized source Parquet. Every semantic decision
is a sidecar keyed by:

```text
source_id, record_id, content_hash
```

Exact model name, immutable model revision, prompt version, timestamp, shard,
raw response, parsed scores, keep decision, and parse status are recorded for
every Qwen result. Parse failures have `qwen_should_keep = null` and must be
re-scored; they are never silently treated as keep or drop decisions.

The packaged tokenizer is OpenAI `cl100k_base`, SHA-256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.

## Source routing

| Family | Semantic filtering | Structural processing |
|---|---|---|
| Stack Overflow, Stack Exchange, Reddit | Qwen on every exact-unique record | Valid IDs/content/hash; exact dedup |
| arXiv cs.CR | Source-selected | Hardened source extraction and document validation |
| Citation-expanded arXiv | Qwen title/abstract pass | Exact abstract dedup, then hardened full-text extraction |
| NVD, CISA KEV, MITRE | None by default | Schema validation, entity-ID repair, source-specific audit |
| Sigma | None by default | YAML/detection validation and outlier review |
| CloudTrail | None by default | Session validation, repetition audit, lossless event chunking |

Numeric thresholds and source exclusions are researcher decisions. Scripts
either require them explicitly or emit audit data without choosing them.

## QA recovery and Qwen scoring

Build the complete universe from original normalized Stack Overflow, Stack
Exchange, Reddit, and recovered 2026 Reddit Parquet:

```bash
python scripts/classify/build_qa_qwen_universe.py --overwrite
```

The generated `data/filtering/v4/qa_to_score/` currently equals the full
1,617,344-record universe because all legacy decisions lack immutable model
revision hashes. The historical decisions remain checkpoints only.

Render a bounded prompt sample without loading vLLM:

```bash
python scripts/classify/score_qwen_vllm.py \
  --input data/filtering/v4/qa_to_score \
  --task qa \
  --qa-sources \
  --dry-run \
  --dry-run-prompts reports/filtering/v4/qa_prompt_sample.jsonl \
  --max-records 100
```

The templates pin `Qwen/Qwen3-8B` to Hugging Face commit
`b968826d9c46dd6066d109eabc6255188de91218`. Override `MODEL_REVISION` only to
start and document a deliberately new scoring run. Submit the reviewed template
manually:

```bash
mkdir -p logs/qwen_qa
sbatch scripts/classify/slurm/qwen_qa_array.sbatch
```

Merge completed Parquet shards:

```bash
python scripts/classify/merge_qwen_sidecars.py \
  --input-dir data/filtering/v4/qwen_qa_shards \
  --output data/filtering/v4/qwen_qa_decisions.parquet
```

The merge fails if any row lacks a model revision or score timestamp.
Each shard also writes `run-config.json` with the exact inference arguments,
fixed seed, and installed vLLM/Transformers versions. A resumed shard refuses
to mix incompatible settings.

Prove complete one-to-one coverage before selecting kept rows:

```bash
python scripts/release/audit_qwen_coverage.py \
  --corpus data/filtering/v4/qa_universe \
  --decisions data/filtering/v4/qwen_qa_decisions.parquet \
  --expected-model Qwen/Qwen3-8B \
  --expected-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --output reports/release/qwen_qa_coverage.json
```

The audit fails on missing, duplicate, orphaned, undecided, parse-failed, or
wrong-model decisions.

Create the deterministic source-by-decision sample required for human review.
The sample size is an explicit researcher decision:

```bash
python scripts/release/sample_qwen_audit.py \
  --corpus data/filtering/v4/qa_universe \
  --decisions data/filtering/v4/qwen_qa_decisions.parquet \
  --per-stratum YOUR_REVIEWED_SAMPLE_SIZE \
  --seed 0 \
  --output reports/release/qwen_qa_human_audit.csv
```

After labeling the review columns, run
`scripts/release/audit_qwen_human_labels.py`. It blocks incomplete reviews and
reports per-source agreement without inventing an acceptance threshold.

## Citation-paper Qwen scoring

Build the exact-deduplicated abstract universe:

```bash
python scripts/classify/build_citation_qwen_universe.py --overwrite
```

This produces 118,664 unique metadata records. Score them with the same pinned
Qwen revision:

```bash
mkdir -p logs/qwen_citations
sbatch scripts/classify/slurm/qwen_citation_abstract_array.sbatch
```

Merge using `merge_qwen_sidecars.py`, audit kept and dropped samples, then
write the accepted paper IDs:

```bash
python scripts/arxiv/select_accepted_citations.py \
  --decisions data/filtering/v4/qwen_citation_abstract_decisions.parquet
```

Download and normalize full text only for this coverage-checked accepted set.

## arXiv extraction

The v2 normalizer:

- rejects archive traversal, links/devices, and excessive expansion;
- resolves `input`, `include`, `subfile`, `import`, and related commands;
- supports nested paths relative to the including file with a root fallback;
- records missing, circular, and outside-project includes in `status.json`;
- preserves percent signs inside common code environments;
- writes content and status atomically.

Restart IDs are generated by:

```bash
python scripts/arxiv/audit_recovery.py --overwrite
```

Use `reports/recovery/arxiv/seed_reextract_ids.txt` for the cs.CR checkpoint.
After the new citation Qwen pass, create a new accepted-ID list rather than
assuming the legacy 17,067-paper selection is final.

## Structured and artifact sources

Build the checkpoint and audit exact duplicates:

```bash
python scripts/build_structured_checkpoint.py --overwrite
```

Structural artifact sidecars are threshold-free unless explicit values are
passed:

```bash
python scripts/classify/report_artifact_quality.py \
  --input data/checkpoints/structured-v1/source_id=sigma \
  --source sigma \
  --output data/filtering/v4/sigma_structural_quality.parquet

python scripts/classify/report_artifact_quality.py \
  --input data/checkpoints/structured-v1/source_id=cloudtrail-flaws \
  --source cloudtrail-flaws \
  --output data/filtering/v4/cloudtrail_structural_quality.parquet
```

Before final assembly, the researcher must resolve the NVD `Rejected` policy,
Sigma outliers, the training context limit used for lossless CloudTrail
chunking, and cross-record exact-deduplication precedence.

## Release gates

A release candidate is ready only when all of these pass:

1. Every record has nonblank IDs, content, content hash, token count, and license.
2. Record IDs are unique within each source.
3. Content hashes and token counts are recomputed and match stored values.
4. Every semantic decision has a pinned model revision and accepted parse status.
5. Keep/drop samples are manually audited by source and decision stratum.
6. Chunking preserves every source event/document segment and records lineage.
7. Exact duplicates and cross-source precedence are documented.
8. Source licenses, attribution, provenance, and redistribution terms are documented.
9. Per-source and total record/token counts are generated from the final files.
10. The Hugging Face data card matches the immutable release manifest.

The final candidate must also pass:

```bash
python scripts/release/audit_source_licenses.py PATH [PATH ...]
```

See `docs/source_licenses.md` for the source-specific release boundary. The
end-to-end cluster handoff is in `docs/marlowe_recovery.md`.
