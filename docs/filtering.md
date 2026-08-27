We are building V3 filtering for a security-domain mid-training corpus. The corpus is already normalized to Parquet, with records containing source_id, record_id, content, content_hash, content_length, license, and source-specific metadata.

Goal:
Produce high-quality sidecar filtering scores/labels for downstream corpus assembly. Do not destructively modify the normalized corpus. All classifier/Qwen outputs should be written as sidecar Parquet keyed by:

source_id
record_id
content_hash

Final keep/drop decisions will be made downstream with DuckDB queries.

Execution Boundary

Codex should implement the filtering infrastructure, scripts, schemas, prompts, sidecar writers, parsers, tests, and documentation.

Codex must NOT run full Qwen inference, Slurm jobs, or large-scale filtering jobs.

Specifically, Codex must not run:
- sbatch
- srun
- vLLM inference over real corpus shards
- full QA Qwen filtering
- full arXiv Qwen filtering
- full 3-4B token inference passes
- bulk ingest or corpus rebuilds unless explicitly requested

Codex may run small local tests and smoke tests when useful, including:
- unit tests
- parser tests
- prompt construction tests
- sidecar schema tests
- tiny synthetic classifier tests
- small dry-run tests that do not load Qwen or call vLLM
- TF-IDF/logistic regression tests on tiny fixtures

For Qwen/vLLM scripts, Codex should provide runnable code and dry-run modes, but should not execute real model inference. Any command that loads Qwen weights, starts vLLM, uses GPUs, or submits Slurm jobs must be left for the human researcher to run manually.

Core Principle:
Use cheap classifiers only where they save expensive Qwen inference without risking obvious false negatives. Use Qwen where semantic judgment matters most.

V3 Filtering Architecture

1. QA Sources

Sources:
- stackoverflow
- stackexchange-infosec
- stackexchange-reverseengineering
- stackexchange-crypto
- stackexchange-tor
- reddit-* security subreddits

Plan:
Train a binary QA quality classifier first. This classifier predicts whether a QA/social record is high enough quality to spend Qwen inference on, based on coherence, substance, usefulness, answer quality, and low-noise structure.

The binary QA quality classifier should be conservative. Its purpose is not to create the final corpus. Its purpose is to remove obvious low-quality junk before Qwen.

Do not keep only the top few percent. Instead, drop only the clearly bad bottom slice, for example:
- empty/near-empty
- incoherent
- pure social chatter
- unresolved low-signal helpdesk threads
- spam
- badly formatted garbage
- extremely low-value Q&A

Send to Qwen if:
- quality score is medium/high, OR
- classifier uncertainty is high, OR
- source metadata suggests possible security value.

Then use Qwen to judge security relevance for the remaining QA candidates.

Qwen QA prompt should evaluate:
- Is the thread actually security-relevant?
- Is it useful for continued pretraining?
- Does the accepted/high-score answer contain substantive technical content?
- Is it too generic, off-topic, homework-like, opinion-only, or low-signal?
- Should this be kept for a security-domain mid-training corpus?

Qwen QA output should be compact JSON:
{
  "security_relevance": 0-3,
  "quality": 0-3,
  "should_keep": true/false,
  "reason": "short explanation"
}

Suggested label meanings:

security_relevance:
0 = not security-relevant
1 = weak/general technical relevance
2 = security-adjacent or useful security context
3 = directly security-relevant

quality:
0 = garbage/broken/spam
1 = low value or thin
2 = usable
3 = high quality/substantive

Write sidecar columns:
qwen_security_relevance
qwen_quality
qwen_should_keep
qwen_reason
qwen_model
qwen_prompt_version
qwen_scored_at

2. arXiv

Sources:
- arxiv papers and chunks

Plan:
Use Qwen as the main judge for arXiv. The primary objective is security relevance. Quality matters too, but arXiv papers are usually high prose quality, so the main failure mode is high-quality but irrelevant papers.

Use a two-stage approach if possible:

Stage 1:
Classify title + abstract + categories + metadata.

Stage 2:
Only run full-paper or chunk-level Qwen classification for papers that are:
- likely security-relevant, OR
- uncertain/borderline, OR
- belong to a broad adjacent area such as crypto, privacy, systems, formal methods, adversarial ML, networking, software engineering, or distributed systems.

This avoids spending full-text inference on clearly irrelevant papers.

Qwen arXiv prompt should evaluate:
- Is this paper actually cybersecurity/security/privacy/cryptography/safety relevant?
- Is it merely general CS/math/ML/systems with no security angle?
- Is it useful for mid-training a security-domain model?
- If chunk-level, does this chunk contain useful security-relevant content?

Qwen arXiv output:
{
  "security_relevance": 0-3,
  "quality": 0-3,
  "should_keep": true/false,
  "reason": "short explanation"
}

For full papers/chunks, preserve enough metadata to aggregate back to paper-level decisions.

3. Qwen Inference Environment / Prompting Contract

Qwen inference will eventually run locally on the Stanford Slurm GPU cluster using vLLM, but Codex should only implement this path. Codex should not execute Qwen inference or submit Slurm jobs.

Do not use hosted APIs for Qwen filtering. There should be:
- no OpenAI API calls
- no OpenRouter calls
- no hosted Qwen API calls
- no per-token API billing

Use vLLM's local Python inference API:

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B or Qwen/Qwen3-8B",
    dtype="bfloat16",
    max_model_len=<appropriate context length>,
    enable_prefix_caching=True,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64-128,
)

outputs = llm.generate(prompts, sampling_params)

Use Hugging Face only for:
- downloading Qwen model weights
- downloading any needed datasets or shards
- loading tokenizers and chat templates

Use Slurm array jobs for scalable inference. Each array task should:
- process one shard or a bounded chunk of records
- write one local shard output/cache file
- skip records already present in the shard cache
- write outputs incrementally after each batch
- be safe to resume after interruption
- include shard id, model name, prompt version, and scored_at timestamp
- avoid loading unnecessary content columns when metadata-only filtering is enough

After all Slurm tasks finish:
- merge shard outputs into one sidecar dataset
- write a manifest with total records scored, kept, dropped, parse failures, model name, prompt version, and timestamp
- run manual audit samples before final downstream filtering

Prompt construction requirements:
- Use separate prompts for QA and arXiv.
- Use compact JSON output only.
- Do not ask for chain-of-thought.
- Disable Qwen thinking where supported, e.g. enable_thinking=False in tokenizer.apply_chat_template.
- Use temperature=0.
- Keep max_tokens small, ideally 64-128.
- Require exactly one JSON object and no markdown/code fences.
- Parse failures must be recorded explicitly.
- Fallback behavior must be conservative and auditable.

Suggested generic Qwen output schema:
{
  "security_relevance": 0,
  "quality": 0,
  "should_keep": false,
  "reason": "short explanation"
}

Parsing contract:
- Parse strict JSON first.
- If strict JSON fails, optionally extract the first JSON object from the response.
- If parsing still fails, write a parse_failure flag.
- Do not silently drop parse failures.
- For parse failures, choose a conservative fallback per source family and record it.

Suggested sidecar fields:
source_id
record_id
content_hash
qwen_security_relevance
qwen_quality
qwen_should_keep
qwen_reason
qwen_parse_status
qwen_model
qwen_prompt_version
qwen_scored_at
qwen_shard_id

4. Sigma

Source:
- sigma

Do not train a generic security-relevance classifier initially. Sigma is already source-selected as detection-rule content, so relevance is mostly established.

Do not run Sigma through a generic prose quality model. Sigma rules are structured artifacts and will look strange to prose classifiers.

Use structural/artifact-quality checks first:
- deprecated/unsupported status
- missing id/title/logsource/detection
- empty or trivial detection logic
- extreme content length outliers
- exact duplicate rules
- malformed YAML
- incomplete rule source

If a classifier is added later, it should be a sigma_artifact_quality classifier, not a generic quality classifier.

5. CloudTrail

Source:
- cloudtrail-flaws

Do not train a generic security-relevance classifier initially. The dataset is source-selected as security-relevant CloudTrail sessions.

Do not run CloudTrail through a generic prose quality model. It is event-log/session data, not prose.

Use structural/session-quality checks first:
- event count
- session duration
- service/action diversity
- principal diversity
- excessive repetition
- obvious bot/noise patterns
- extremely long sessions needing chunking
- sessions with insufficient context

If a classifier is added later, it should be a cloudtrail_session_quality classifier focused on whether a session is structurally useful for training.

6. Strongly Source-Selected Structured Sources

Sources:
- nvd
- cisa-kev
- github-advisory
- mitre-attack
- mitre-cwe
- capec
- bron

Do not train security relevance classifiers initially. These sources are already selected for relevance.

Use structural validation, deduplication, length checks, and source-specific quality features instead.

7. Compute Strategy

Use Qwen where it buys semantic judgment:
- QA security relevance after cheap quality prefiltering
- arXiv relevance and quality, preferably abstract-first

Avoid Qwen where source selection already establishes relevance:
- NVD
- MITRE
- Sigma
- CloudTrail
- advisories

Avoid generic classifiers for artifact sources:
- Sigma
- CloudTrail

Expected Qwen load:
- Naive QA + arXiv full-text pass: roughly 3-4B input tokens.
- With QA quality prefiltering and arXiv abstract-first filtering: likely closer to 1.5-2.5B input tokens, depending on thresholds and retention.

Run a pilot before full-scale inference:
- sample 1% of QA candidate tokens
- sample 1% of arXiv tokens, or an equivalent paper/chunk sample
- measure tokens/sec, records/sec, GPU-hours, parse failure rate, retention rate, and output quality
- manually audit kept and dropped samples

Minimum manual audit:
- 100 kept QA
- 100 dropped QA
- 100 kept arXiv
- 100 dropped arXiv

8. Output Contract

All filtering outputs must be sidecar Parquet files keyed by:
source_id
record_id
content_hash

Do not overwrite normalized corpus data.

Include:
classifier/model name
prompt version
scored_at timestamp
raw label/score fields
should_keep field
short reason
parse status/failure fallback if applicable

Final keep/drop logic is downstream and queryable, not hardcoded into ingestion.

Recommended order of implementation:
1. Build binary QA quality classifier and sidecar scoring.
2. Use binary QA quality probabilities to select candidate QA records for Qwen.
3. Build Qwen QA relevance scorer.
4. Build Qwen arXiv abstract/metadata scorer.
5. Build optional Qwen arXiv full/chunk scorer for likely/uncertain papers.
6. Add structural artifact-quality reports for Sigma and CloudTrail.
7. Audit outputs and tune thresholds in DuckDB.

## Implemented V3 tooling

The filtering code lives under `src/classify/` and `scripts/classify/`. All
outputs are sidecar Parquet keyed by `source_id`, `record_id`, and
`content_hash`.

### QA quality classifier

The original 4-class labels remain useful for analysis and audit:

- `0` = junk/broken/spam/incoherent/empty
- `1` = low value/thin/chatter/helpdesk/unresolved
- `2` = usable technical QA
- `3` = high-quality substantive technical QA

The production V3 QA prefilter is binary. It answers: "Is this QA record high
enough quality to spend Qwen inference on?"

- `0` = not high enough quality for Qwen
- `1` = high enough quality to send to Qwen

Derive the binary labels from the 4-class labels:

```bash
python scripts/classify/derive_qa_quality_binary_labels.py \
  --input data/classifier-labels/qa_quality_labeling_sample.parquet \
  --output data/classifier-labels/qa_quality_binary_labeling_sample.parquet
```

Mapping: `qa_quality_label` 0/1 becomes `qa_quality_binary_label` 0; 2/3
becomes 1. The derivation preserves the original 4-class label and label notes.

Train the conservative binary QA/social quality classifier. The label file must
include numeric `qa_quality_binary_label` values in `0..1`; string labels and
out-of-range labels fail fast. Training writes both `metadata.json` and
`metrics.json` next to the model:

```bash
python scripts/classify/train_tfidf_logreg.py \
  --labels data/classifier-labels/qa_quality_binary_labeling_sample.parquet \
  --model-dir models/classify/tfidf-logreg/qa_quality_binary \
  --task qa_quality_binary \
  --min-df 1 \
  --max-iter 5000
```

Score normalized QA/social records into a sidecar. `qa_quality_binary` scoring
is restricted to `stackoverflow`, `stackexchange-*`, and `reddit-*` by default;
`--qa-sources` is included for readability and explicitness:

```bash
python scripts/classify/score_tfidf_logreg.py \
  --input data/training-clean-v2/normalized \
  --output data/filtering/v3/qa_quality_binary.parquet \
  --model-dir models/classify/tfidf-logreg/qa_quality_binary \
  --task qa_quality_binary \
  --qa-sources
```

Select Qwen candidates. Thresholds are intentionally CLI parameters with no
research-policy defaults; omit them to keep all QA records as candidates:

```bash
python scripts/classify/select_qa_qwen_candidates.py \
  --corpus data/training-clean-v2/normalized \
  --quality-sidecar data/filtering/v3/qa_quality_binary.parquet \
  --output data/filtering/v3/qa_qwen_candidates.parquet \
  --min-quality-score <RESEARCHER_DECIDES>
```

Recommended downstream behavior: send predicted label 1 records to Qwen; also
send uncertain records to Qwen; skip Qwen only for high-confidence label 0
predictions. The probability threshold is researcher-tuned, for example
`qa_quality_binary_prob_0 >= 0.85` or `0.90` for high-confidence drops.

### Qwen dry-run and inference scripts

Dry-run prompt rendering is safe for Codex and local tests because it does not
import vLLM or load model weights. The scorer fails fast when task-required
prompt fields are missing or blank:

- QA requires `source_id`, `record_id`, `content_hash`, `content`
- arXiv abstract requires `source_id`, `record_id`, `content_hash`, `title`, `abstract`
- arXiv full/chunk requires `source_id`, `record_id`, `content_hash`, `content`

Optional metadata such as tags, categories, authors, scores, and arXiv IDs is
loaded only when present. The arXiv abstract task does not load full `content`.

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

Real Qwen runs use local vLLM and must be launched manually by the researcher:

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

The same scorer supports:

- `--task qa`
- `--task arxiv-abstract`
- `--task arxiv-full`

Use source filters on every Qwen run against combined normalized data:

- `--qa-sources` for QA/social sources (`stackoverflow`, `stackexchange-*`, `reddit-*`)
- `--source-id arxiv` for arXiv abstract/full jobs
- repeatable `--source-id SOURCE` and `--source-like PATTERN` for other bounded runs

Source filtering happens before prompt-field validation, so unrelated rows in a
combined dataset do not cause missing-field failures. For Hive-partitioned
normalized directories, the scorer discovers matching `source_id=...` partition
files before building the task dataset schema. Exact `--source-id` filters map
to `source_id=<value>` partitions; `--source-like` matches partition source IDs
with glob syntax. If no `source_id=...` partitions are discoverable, the scorer
falls back to row-level source filtering after opening the input, so mixed
unpartitioned inputs can still be limited by their inferred Arrow schema.

Each real run writes batch-sized part files under the shard output directory so
interrupted jobs can resume by skipping keys already present in shard output.
Merge completed shards with:

```bash
python scripts/classify/merge_qwen_sidecars.py \
  --input-dir data/filtering/v3/qwen_qa_shards \
  --output data/filtering/v3/qwen_qa.parquet
```

Slurm templates are provided under `scripts/classify/slurm/`. They are templates
for manual researcher submission only; Codex must not run `sbatch` or `srun`.
Create the template-specific log directory before submission, for example
`mkdir -p logs/qwen_qa`.

### Structural artifact sidecars

Sigma and CloudTrail use structural checks rather than generic prose models:

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

Numeric outlier thresholds for these reports are optional CLI parameters marked
for researcher tuning.
