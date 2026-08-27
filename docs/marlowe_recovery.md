# Marlowe recovery runbook

This runbook restarts only the corpus work. It does not submit SecurityClip,
benchmark, YouTube, RegMix, FineWeb, BRON, or GitHub Advisory jobs.

## 1. Authenticate and transfer the bounded restart set

First establish an interactive Marlowe session so Stanford authentication is
active. Do not place passwords or tokens in scripts.

```bash
ssh marlowe
exit
bash scripts/marlowe/sync_inputs.sh
```

The default transfer includes the 1.6 GB QA scoring queue, citation abstract
universe, structured checkpoint, arXiv metadata, and recovery reports. It does
not copy the duplicate QA universe or the 3.4 GB legacy normalized paper tree.
The legacy tree remains on the laptop as a checkpoint. Copy it only for
comparison or emergency recovery:

```bash
bash scripts/marlowe/sync_inputs.sh --include-legacy-papers
```

Both scripts honor `REMOTE_HOST` and `REMOTE_ROOT`. The default destination is
`/scratch/m000091-pm05/natedem/security-corpus`. No sync command uses
`--delete`.

## 2. Create the GPU environment

On Marlowe, load the Stanford-supported Python/CUDA environment appropriate
for vLLM, then create an isolated project environment. Module names are
intentionally not hardcoded because they are cluster-managed.

```bash
cd /scratch/m000091-pm05/natedem/security-corpus
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,qwen]"
pytest
```

Confirm that vLLM sees the allocated GPU before starting the full arrays.

## 3. Review prompts and score QA

Generate a bounded prompt sample first:

```bash
python scripts/classify/score_qwen_vllm.py \
  --input data/filtering/v4/qa_to_score \
  --task qa \
  --qa-sources \
  --dry-run \
  --dry-run-prompts reports/release/qa_prompt_sample.jsonl \
  --max-records 100
```

The Slurm output directory must exist before submission because Slurm opens
the log file before the job script starts.

```bash
mkdir -p logs/qwen_qa
sbatch scripts/classify/slurm/qwen_qa_array.sbatch
```

After every shard finishes, merge and prove one-to-one coverage:

```bash
python scripts/classify/merge_qwen_sidecars.py \
  --input-dir data/filtering/v4/qwen_qa_shards \
  --output data/filtering/v4/qwen_qa_decisions.parquet

python scripts/release/audit_qwen_coverage.py \
  --corpus data/filtering/v4/qa_to_score \
  --decisions data/filtering/v4/qwen_qa_decisions.parquet \
  --expected-model Qwen/Qwen3-8B \
  --expected-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --output reports/release/qwen_qa_coverage.json
```

Choose the manual review sample size as a research decision, then create a
deterministic sample covering every source and both decisions:

```bash
python scripts/release/sample_qwen_audit.py \
  --corpus data/filtering/v4/qa_to_score \
  --decisions data/filtering/v4/qwen_qa_decisions.parquet \
  --per-stratum YOUR_REVIEWED_SAMPLE_SIZE \
  --seed 0 \
  --output reports/release/qwen_qa_human_audit.csv
```

Fill the `manual_should_keep`, `manual_notes`, `reviewer`, and `reviewed_at`
columns. Any systematic disagreement requires a documented prompt change and
a new complete scoring run, not a hidden post-hoc rule.

After review, validate completeness and record agreement without imposing an
undocumented acceptance threshold:

```bash
python scripts/release/audit_qwen_human_labels.py \
  --input reports/release/qwen_qa_human_audit.csv \
  --output reports/release/qwen_qa_human_audit.json
```

## 4. Score citation abstracts and select papers

```bash
mkdir -p logs/qwen_citations
sbatch scripts/classify/slurm/qwen_citation_abstract_array.sbatch

python scripts/classify/merge_qwen_sidecars.py \
  --input-dir data/filtering/v4/qwen_citation_abstract_shards \
  --output data/filtering/v4/qwen_citation_abstract_decisions.parquet

python scripts/release/audit_qwen_coverage.py \
  --corpus data/filtering/v4/citation_abstract_universe.parquet \
  --decisions data/filtering/v4/qwen_citation_abstract_decisions.parquet \
  --expected-model Qwen/Qwen3-8B \
  --expected-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --output reports/release/qwen_citation_coverage.json

python scripts/release/sample_qwen_audit.py \
  --corpus data/filtering/v4/citation_abstract_universe.parquet \
  --decisions data/filtering/v4/qwen_citation_abstract_decisions.parquet \
  --per-stratum YOUR_REVIEWED_SAMPLE_SIZE \
  --seed 0 \
  --output reports/release/qwen_citation_human_audit.csv

python scripts/release/audit_qwen_human_labels.py \
  --input reports/release/qwen_citation_human_audit.csv \
  --output reports/release/qwen_citation_human_audit.json

python scripts/arxiv/select_accepted_citations.py \
  --decisions data/filtering/v4/qwen_citation_abstract_decisions.parquet
```

The accepted-ID command refuses incomplete, duplicate, parse-failed, or
wrong-model decisions.

## 5. Download and re-extract every selected paper

Harvest authoritative arXiv metadata for newly accepted citation papers, then
download both the cs.CR seed and accepted citation source sets:

```bash
python scripts/arxiv/harvest_citation_metadata.py \
  --id-file data/filtering/v4/citation_accepted_ids.txt \
  --output-dir data/arxiv/raw/metadata/citations

python scripts/arxiv/download_sources.py \
  --id-file reports/recovery/arxiv/seed_reextract_ids.txt \
  --output-dir data/arxiv/raw/source/downloads

python scripts/arxiv/download_sources.py \
  --id-file data/filtering/v4/citation_accepted_ids.txt \
  --output-dir data/arxiv/raw/source/downloads

python scripts/arxiv/normalize_sources.py \
  --downloads-dir data/arxiv/raw/source/downloads \
  --output-dir data/arxiv/raw/source/normalized

python scripts/ingest.py arxiv \
  --input data/arxiv/raw \
  --output-dir data/rebuilt/arxiv-normalized
```

The connector rejects unversioned legacy extraction results. Only
`latex-v2` and `pdf-text-v1` statuses enter the rebuilt Parquet.

## 6. Checkpoint results locally

From the laptop, copy decisions, shard provenance, accepted IDs, extracted
papers, normalized Parquet, and release reports back without deleting local
files:

```bash
bash scripts/marlowe/sync_outputs.sh
```

Raw downloaded sources are likely much larger. Preserve them locally when
space permits because they make extraction repeatable without another network
download:

```bash
bash scripts/marlowe/sync_outputs.sh --include-arxiv-downloads
```

Do not delete the Marlowe scratch directory until local file counts and hashes
have been checked.

## 7. Final gates before assembly

Before producing a Hugging Face candidate:

1. Complete and sign off both manual Qwen audits.
2. Resolve NVD Rejected records, the Sigma hash-list outlier, CloudTrail
   context/chunking, and exact-duplicate precedence.
3. Repair Stack Exchange contribution-level attribution by re-ingesting from
   source dumps.
4. Obtain or document redistribution authority for Reddit and flaws.cloud.
5. Apply the per-record arXiv license boundary in `docs/source_licenses.md`.
6. Choose a code license for the public repository.
7. Run the license audit on the exact release files and generate immutable
   record/token counts from those files.

No current count is a final release count until these gates pass.
