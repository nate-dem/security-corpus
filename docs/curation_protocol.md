# Quality-first expansion work — 2026-09-14

The researcher clarified that 3B tokens is an aspiration, not a release gate.
Retain useful, well-supported research data and accept a smaller public corpus
when quality or publication terms require it. Historical model and inventory
outputs remain immutable, including their earlier target wording.

**Current action:** v3 reports are reviewed and CPU input preparation is still
running. Use [the independent GPU follow-up](../scripts/curation/GPU_REVIEW.md)
to test a source-only quality reviewer and additional examples now. Neither model
is approved for bulk selection; see [the findings](curation_v3_review.md).
Earlier comparison/manual-review instructions below are historical.
No user labeling task is pending; assistant origin is explicitly recorded.

**Inventory status:** jobs 485980 and 485981 completed and their reports have been
reviewed. English candidates contain 7,312,515,096 exact-unique cl100k_base
tokens; the existing baseline recount is 1,467,709,789. See
[the findings and immediate review step](expansion_preflight_review.md).
The submission commands below document the completed preparation; do not
resubmit them for the current run.

## Work now implemented

1. `scripts/youtube/prepare_english.py` prepares the complete approved English
   frame, including translations. It preserves original text, source metadata,
   licenses, and shard/row lineage; recomputes cl100k_base tokens; and produces
   an exact-content index. It does not make quality or final selection decisions.
   All variants remain available. Blank/missing English rows remain in accounting.
2. `scripts/curation/review_packet.py` builds a local review page with no model
   predictions displayed. The initial packet contains 40 saved web samples per
   source and the six unresolved YouTube spans. Forty is a configurable review
   workload, not an acceptance threshold or a statistical power claim.
3. `scripts/curation/rubric.py` and `score.py` provide a development classifier
   with distinct web/transcript instructions and separate evidence arrays per
   judgment. It reuses tested checkpoint/inference mechanics. It preserves raw
   responses, pins model/tokenizer/configuration, and leaves selection null.
   The earlier YouTube scorer and repair implementations are unchanged.
4. `scripts/release/baseline_preflight.py` recomputes baseline tokens/hashes,
   inventories current source-policy states, and produces a draft file manifest.
   Findings remain explicit even when the scan completes normally. This does
   not rescore or modify the baseline, infer permission, or approve publication.
5. `docs/dataset_card.draft.md` starts the release documentation. Final counts,
   quality evidence, source treatment, and uploaded-file checksums remain pending.

## Review and evaluation

Open `reports/curation/development-v1/review.html` in a local browser. Judge each
dimension independently and include a short supporting passage/reason. Save
annotations with the export button; browser storage is not a backup. The page
operates locally and sends no data to a server. The exported file binds labels
to the exact packet and text. `validate_labels` verifies that binding and rejects
missing reviewer information and incomplete or invalid labels.

All current examples are **development data**. The six transcript cases were
chosen because the classifier disagreed with itself. Web samples were originally
sampled within shards and have unequal source-row inclusion probabilities.
Neither group supports an unweighted whole-corpus quality/yield estimate. Do not
call an assistant's diagnostic reading a human label or independent ground truth.

Next, finalize the rubric using reviewed development cases, freeze it, and draw
an independent evaluation sample from the complete candidate inventory. Keep
duplicates and video variants together; inspect channel/domain leakage as well.
Evaluate predicted keeps and drops, including usable introductory computing,
off-topic technical material, promotion without explanation, mistranslation,
and missing visual context. These are diagnostic failure modes, not production
keyword lists or format-based exclusions. Report uncertainty, disagreement, and
token-weighted as well as document-weighted outcomes with the sampling design.

Proposed selection discussion: central/supporting relevance with actual substance
and usable text is the starting candidate combination. Mixed or partly usable
documents require context-preserving section review; the existence of one good
passage does not admit an entire unrelated document. This is a proposal for
researcher review, not an implemented keep rule. No numeric quality thresholds
have been introduced.

Do not spend another GPU allocation trying to force old evidence failures to
zero. The new runner is a different development experiment. Its 86-case dry run
measured 149,570 input tokens and a maximum prompt of 5,422 tokens, leaving room
for the 1,024-token response in an 8,192-token context. No new GPU inference has
run. The next classifier allocation should follow review of the packet/rubric.

## Assembly and deduplication work

The new English index separates scoring identity (exact text) from attribution
lineage (every source row). Its deterministic read pointer is not an approved
release-source precedence rule. Existing web sidecars likewise supply exact
hashes and candidate counts. Preserve provenance through downstream assembly.

Near-deduplication still needs an implemented candidate-pair workflow and
researcher-reviewed thresholds/precedence. Evaluate candidate pairs on real
examples before automatic removal, especially copied code, short descriptions,
and partial overlap. An upstream dataset's own deduplication does not establish
that the merged corpus is deduplicated. Any segmentation must record source
offsets and account separately for overlap and unique content tokens.

Final assembly still needs the reviewed selection policy and source release
treatment. Keep the canonical corpus schema unchanged until a web/transcript
representation is reviewed; current preparation files are intermediate artifacts.
The draft manifest enumerates existing files and their hashes without pretending
that a new assembled candidate already exists.

## Publication evidence for the additions

Checked against upstream cards on 2026-09-14:

- [YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons)
  describes CC-BY video provenance and requires contributor credit. Preserve
  video/channel identifiers, source links, supplied license, and translation
  metadata when selecting transcripts.
- [RedSage-CFW](https://huggingface.co/datasets/RISys-Lab/RedSage-CFW)
  lists ODC-By and describes a mixture of security and educational web material.
  Source URLs and original metadata must survive selection; its relevance flag
  is not a substitute for the corpus's quality assessment.
- [Primus-FineWeb](https://huggingface.co/datasets/trendmicro-ailab/Primus-FineWeb)
  lists ODC-By while explicitly retaining source-site terms. The absence of
  page-level license fields in our profile remains an unresolved evidence gap;
  dataset access does not itself supply those missing terms.

These observations begin release preparation; they do not silently change the
existing machine-readable source policy. Review full-text versus reference-only
release treatment for unresolved sources before packaging the final subset.

## Next Marlowe commands

Copy code and configuration from the **Mac** (no bulk data transfer):

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/marlowe/sync_code.sh
```

Then on **Marlowe**:

```bash
cd /scratch/m000091/natedem/security-corpus
export YOUTUBE_DATA_DIR=/scratch/m000091/natedem/youtube-transcripts
bash scripts/youtube/run_prepare_english.sh --check-env && \
mkdir -p logs/youtube logs/release && \
sbatch scripts/youtube/prepare_english.sbatch
```

The main next job requests **4 CPUs, 16 GB RAM, no GPUs, up to four hours**.
It scans the verified raw shards and writes under
`/scratch/m000091/natedem/youtube-transcripts/english-candidates-v1/`.
Expect additional storage for English text and its index on Marlowe. The full
data stay there. After timeout/preemption, resubmit the same command with the
same code; finished shards are verified and reused. An unfinished shard restarts.
Do not change its implementation/dependencies during the run or before resuming.

An independent **release-preparation** job can also run now:

```bash
sbatch scripts/release/baseline_preflight.sbatch
```

It requests the same CPU/RAM/time allocation and uses the existing web CPU
environment. It writes to `/scratch/m000091/natedem/release-review/baseline-JOBID/`.
It scans the existing selection once and currently restarts the scan if preempted;
it has no model inference. A completed scan may report integrity or license
findings. Read `summary.json`; job completion does not mean publication approval.

Both `sbatch` jobs survive SSH/tmux disconnection. Monitor with `squeue -u natedem`
and the logs under `logs/youtube/` and `logs/release/`.

After English preparation completes, copy **only its small reports** to the Mac:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/youtube/english-candidates-v1
rsync -av \
  --include='/summary.json' --include='/run-config.json' --exclude='*' \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/youtube-transcripts/english-candidates-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/youtube/english-candidates-v1/
```

The English summary contains per-shard checksums and all aggregate counts.
The much larger `rows/` files and `unique_index.parquet` remain on Marlowe.
