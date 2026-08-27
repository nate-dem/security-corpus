# V3 Filtering Scripts

Install the optional dependencies before running these scripts:

```bash
pip install -e ".[classify]"
```

The 4-class `qa_quality_label` values remain useful for analysis and audit, but
the production QA prefilter is binary: `0` means not high enough quality for
Qwen, and `1` means high enough quality to spend Qwen inference on.

Derive binary labels from the manually labeled 4-class sample:

```bash
python scripts/classify/derive_qa_quality_binary_labels.py \
  --input data/classifier-labels/qa_quality_labeling_sample.parquet \
  --output data/classifier-labels/qa_quality_binary_labeling_sample.parquet
```

Train the binary QA quality task. Labels must use numeric
`qa_quality_binary_label` values in `0..1`. Training writes `metadata.json`,
`model.joblib`, and `metrics.json`:

```bash
python scripts/classify/train_tfidf_logreg.py \
  --labels data/classifier-labels/qa_quality_binary_labeling_sample.parquet \
  --model-dir models/classify/tfidf-logreg/qa_quality_binary \
  --task qa_quality_binary \
  --min-df 1 \
  --max-iter 5000
```

Score normalized Parquet into QA-source-only sidecar scores. For
`qa_quality_binary`, the implementation defaults to `stackoverflow`,
`stackexchange-*`, and `reddit-*`; `--qa-sources` makes that boundary explicit:

```bash
python scripts/classify/score_tfidf_logreg.py \
  --input data/training-clean-v2/normalized \
  --output data/filtering/v3/qa_quality_binary.parquet \
  --model-dir models/classify/tfidf-logreg/qa_quality_binary \
  --task qa_quality_binary \
  --qa-sources
```

Select QA/social candidates for Qwen. No thresholds are applied unless they are
passed explicitly by the researcher:

```bash
python scripts/classify/select_qa_qwen_candidates.py \
  --corpus data/training-clean-v2/normalized \
  --quality-sidecar data/filtering/v3/qa_quality_binary.parquet \
  --output data/filtering/v3/qa_qwen_candidates.parquet \
  --min-quality-score <RESEARCHER_DECIDES>
```

Recommended downstream behavior: send predicted binary label 1 records to Qwen,
send uncertain records to Qwen, and skip Qwen only for high-confidence binary 0
predictions such as `qa_quality_binary_prob_0 >= 0.85` or `0.90` after
researcher tuning.

Render Qwen prompts without loading Qwen/vLLM. The scorer fails fast if
task-required prompt columns are missing or blank:

- QA: `source_id`, `record_id`, `content_hash`, `content`
- arXiv abstract: `source_id`, `record_id`, `content_hash`, `title`, `abstract`
- arXiv full/chunk: `source_id`, `record_id`, `content_hash`, `content`

The arXiv abstract task does not load full `content`. Use source filters
whenever the input path is a combined normalized dataset: `--qa-sources` for
QA/social runs, `--source-id arxiv` for arXiv runs, or repeatable `--source-id`
/ `--source-like` for custom bounded runs. Filtering happens before
required-field validation. For Hive-partitioned normalized directories, matching
`source_id=...` partition files are selected before the task schema is
validated. If no source partitions are discoverable, the script falls back to
row-level filtering after opening the input, so mixed unpartitioned inputs still
depend on the inferred Arrow schema.

```bash
python scripts/classify/score_qwen_vllm.py \
  --input data/training-clean-v2/normalized \
  --task qa \
  --qa-sources \
  --candidate-sidecar data/filtering/v3/qa_qwen_candidates.parquet \
  --candidate-keep-column qa_candidate_for_qwen \
  --dry-run \
  --dry-run-prompts reports/filtering/v3/qwen_qa_prompt_sample.jsonl \
  --max-records 25
```

Real Qwen inference is local vLLM only and must be run manually by the
researcher on the GPU cluster. Codex should not run it:

```bash
python scripts/classify/score_qwen_vllm.py \
  --input data/training-clean-v2/normalized \
  --task qa \
  --qa-sources \
  --candidate-sidecar data/filtering/v3/qa_qwen_candidates.parquet \
  --candidate-keep-column qa_candidate_for_qwen \
  --output-dir data/filtering/v3/qwen_qa_shards/shard-0 \
  --model Qwen/Qwen3-4B \
  --shard-id 0 \
  --num-shards 100
```

Slurm templates are in `scripts/classify/slurm/`. Before manual `sbatch`
submission, create the corresponding log directory, for example
`mkdir -p logs/qwen_qa`.

Merge Qwen shard parts:

```bash
python scripts/classify/merge_qwen_sidecars.py \
  --input-dir data/filtering/v3/qwen_qa_shards \
  --output data/filtering/v3/qwen_qa.parquet
```

Write structural artifact-quality reports:

```bash
python scripts/classify/report_artifact_quality.py \
  --input data/sigma/normalized \
  --source sigma \
  --output data/filtering/v3/sigma_artifact_quality.parquet

python scripts/classify/report_artifact_quality.py \
  --input data/cloudtrail/normalized \
  --source cloudtrail-flaws \
  --output data/filtering/v3/cloudtrail_artifact_quality.parquet
```
