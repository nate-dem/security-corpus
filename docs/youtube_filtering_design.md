# YouTube ingestion and filtering design

Status: content-classification pilot implemented after the 2026-09-11 profile;
production selection policy and the canonical transcript schema remain open.
For execution status as of 2026-09-14, see the [completion plan](finish_plan.md)
and [latest repair review](youtube_repair_review.md). GPU pilot inference has
run; its observed classification errors still need resolution and evaluation.
The researcher rejected the
teammate's channel/metadata classifier strategy and requested ingestion of
all shards followed by LLM-based classification.

The researcher approved English transcripts, including translations, for the
first pilot and subsequently approved proceeding with the reviewed plan. The
completed profile, sample review, and provisional token-yield scenarios are recorded in
`docs/youtube_profile_review.md`; the proposed prompt is in
`docs/youtube_classifier_prompt.md`. Production selection policy remains open.

Execution progress: the researcher reported all 439 shards downloaded and
verified on Marlowe at revision `9addbabbfcd7409acbcd11a3b59ec2aef6da7eb0`.
The completed `scripts/youtube/profile.py` run inventoried 22,684,737 rows and
exported 439 full-text diagnostic samples with provenance. It does not yet
normalize all transcripts, compute exact content duplicates or token totals,
or apply any filtering decision. Download/profile instructions are in
`scripts/youtube/README.md`.

The new [`scripts/youtube_filter/README.md`](../scripts/youtube_filter/README.md)
provides CPU preparation and a pinned Qwen3-8B GPU pilot. Preparation draws
500 random English raw rows plus separate diagnostic cases, preserves duplicate
lineage, and retrieves complete text. The scorer labels every contiguous span
with evidence and explicit failure accounting; it does not select training data.
Local CPU tests and real-tokenizer checks pass. The 8B and 32B GPU pilots have
run on Marlowe; evidence checks alone do not establish classification accuracy
or final keep rules.

The existing non-YouTube baseline remains 659,147 records and 1,467,709,789
stored cl100k_base reference tokens. This work does not rescore that baseline.

## Intended contribution

Proposed objective: add useful spoken explanations of security concepts,
mechanisms, practitioner workflows, investigations, and technical reasoning.
The current retained corpus has about 909.5M academic-paper tokens, 284.6M
Q&A/community tokens, and 246.1M CloudTrail tokens. Complementing those sources
is a reason to measure the kinds of content YouTube contributes, not a reason
to impose a new mixture quota or discard overlapping concepts.

Assess substantive introductory explanations as well as advanced content.
Difficulty, polish, channel fame, and audience size are not quality criteria.
The researcher approved security plus directly supporting technical material
on 2026-09-11, including substantive operating systems, networking, cloud
infrastructure, and software engineering explanations. Keep the connection to
security and actual instructional substance explicit in the pilot. Career,
news, product, and certification formats still require content-level assessment;
no blanket inclusion or exclusion has been chosen for these formats.

## Ingest before selecting

1. Pin the YouTube-Commons dataset revision and enumerate the actual Parquet
   inventory. Do not assume the historical shard count remains correct.
2. Download every shard with resumable transfers; record expected and observed
   size/checksum, completion, row count, and schema. A failed shard is reported
   and retried, not silently skipped.
3. Preserve original transcript text, all source metadata, source shard and row
   provenance. Normalized data and later decision tables are separate artifacts.
   There is no channel allowlist, keyword gate, or inherited 50-word cutoff.
   Empty/malformed records are accounted for separately without losing raw data.
4. Inventory transcript languages, original languages, variants per video,
   missing metadata, lengths, and exact duplicates. Language eligibility is
   decided downstream; downloading all shards does not commit the training
   mixture to every language or translation.
5. Build an exact-content duplicate map to avoid repeated content-only scoring,
   preserving every variant's video/channel/license provenance. If a future
   prompt uses metadata to determine its label, its cache key must bind that
   metadata too; a text hash alone is then insufficient.

YouTube-Commons contains both original and automatically translated transcripts.
Do not collapse these to video ID alone. The published schema includes language
metadata but does not advertise word-level timestamps or reference audio;
sentence/character offsets can be retained, but reliable timing or measured
speech-recognition error rates cannot be assumed from these Parquets.

## Proposed classification rubric

These are proposed downstream sidecar fields, not approved schema additions or
numeric thresholds. The production prompt and label definitions require a
small calibration exercise on actual ingested transcripts.

| Dimension | What the classifier should assess | Example label anchors |
|---|---|---|
| Security relevance | Does the text teach a security mechanism, problem, practice, or meaningful supporting concept? | Central; supporting/adjacent; incidental; absent |
| Training value | Does it contain concrete explanation, procedures, examples, evidence, or causal reasoning? | Substantive; useful but limited; mostly unsupported claims/promotion; no usable content |
| Transcript usability | Can a reader learn from the text alone? | Self-contained; locally damaged but recoverable; critically dependent on visuals; incoherent/repetitive |
| Content form | What kind of material is it? | Explanation, demonstration, incident analysis, interview, news, product presentation, study material, etc.; proposed categories need review |

Keep relevance, usefulness, and transcript usability separate. A model's
single `should_keep` judgment should not be the only durable output. Derive
the selected subset later from the researcher's reviewed policy.

Require short evidence spans copied from the transcript with verifiable
character offsets, plus concise reasons. Treat transcript text as data, never
as classifier instructions. Validate structured responses; parsing failures
and insufficient evidence remain undecided and eligible for retry.

Do not present the classifier as a factual verifier. It can flag internal
inconsistency, unsupported assertions, and unreadable transcription, but
plausible fluent text does not prove technical correctness. Evaluate identifiable
errors in the calibration set rather than treating self-reported confidence as
a calibrated probability.

Examples for rubric calibration, not automatic source exclusions:

- A narrated authentication failure and its cause can be valuable even when
  spoken informally or intended for beginners.
- A demonstration consisting mostly of "click here" and "look at this output"
  may be unusable without its missing screen recording.
- An incident report that explains mechanisms differs from a headline recap.
- A product presentation may contain a substantive technical section; its
  channel or format alone should not determine the result.

## Long transcripts and mixed content

Classify full short transcripts. For long transcripts, assess contiguous,
context-preserving spans covering the entire text, with source offsets and
neighboring context. A 2,000–4,000-token span is an initial benchmarking
proposal, not a selected cutoff or maximum allowed document length.

Do not classify a full video from title/description alone, its first few
minutes, or only head/tail excerpts. Important content can occur in the middle.
Measure the fractions of original, non-overlapping tokens in each category;
do not let overlap inflate counts or a single good segment admit a mostly
irrelevant long document.

Retain parent-document identity and all span decisions. Whole-document
selection is the simpler initial output. Segment salvage is an optional
follow-on if sampling shows substantial recoverable mixed content: preserve
contiguous context and dependencies rather than stitching isolated high-scoring
sentences together. Do not generate synthetic rewrites of damaged transcripts.

## Calibration before the full GPU run

Propose an initial few hundred examples spanning random records, likely
security material, adjacent/hard-negative topics, length bands, languages,
translation variants, and noisy or visually dependent transcripts. Include
unfamiliar channels. Sampling signals may identify review strata but must not
become production exclusion gates. Exact sample size is a researcher decision.

Separate rubric-development examples from held-out human-labeled checks; split
by video and preferably channel so translations, duplicates, or repeated
channel templates do not leak across these sets. Review both predicted keeps
and drops. Random sampling provides prevalence estimates; enriched review
strata alone cannot estimate whole-corpus precision or retention.

Compare the existing smaller Qwen inference stack with a stronger reference
judge on the same held-out examples. Select the production model and prompt
using observed recall of useful content, retained-token quality, error types,
coverage of spoken formats, and measured GPU throughput. Larger-model agreement
is supporting evidence, not ground truth. Do not hardcode the teammate's 4B
model choice, model confidence cutoff, or prior prompt.

Use a stronger model for unresolved cases only if the pilot shows a useful
cost/quality tradeoff. Training a separate distilled classifier is optional
and not part of the critical path for this deadline. Model choice, span size,
acceptance thresholds, and escalation rules follow the pilot; do not estimate
total GPU hours from a historical metadata-only job.

This validates the filtering instrument. It does not establish a downstream
model-training gain; that would require a controlled training/evaluation run.

## Outputs and Marlowe execution

- CPU ingestion: immutable raw shards, inventory, normalized candidate records,
  source lineage, duplicate maps, and corpus statistics. No GPUs required.
- GPU pilot, then production classification: pinned model and prompt revisions,
  resumable decisions, evidence spans, and explicit retry/coverage reporting.
- CPU selection: researcher-approved rule applied to sidecars, retained Parquet,
  kept/dropped/undecided token counts, and source/format/translation breakdowns.
- Back up normalized candidates and decisions off scratch before cleanup so a
  deleted Marlowe working directory does not erase the work again.

Already approved: English including translations, and cybersecurity plus
directly supporting computing topics. Still to decide: ambiguous content forms,
final acceptance policy, and whether mixed videos need segment-level salvage.
All raw shards have already been downloaded.

## References

- [YouTube-Commons dataset card](https://huggingface.co/datasets/PleIAs/YouTube-Commons)
  documents original/translated transcripts and source metadata.
- [YouTube-Commons published schema](https://huggingface.co/datasets/PleIAs/YouTube-Commons/raw/main/README.md)
  documents the available fields; actual per-shard schemas must still be inspected.
- [FineWeb paper](https://arxiv.org/abs/2406.17557) supports evaluating filtering
  and deduplication choices empirically; its results do not establish the best
  policy for this security transcript corpus.
- [FineWeb-Edu classifier limitations](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier#limitations)
  warn about specialized domains and preference for academic-looking prose.
  This motivates a transcript-specific rubric rather than reusing its threshold.
