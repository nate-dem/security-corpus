# YouTube profile review and proposed pilot

Reviewed against the transferred profile completed 2026-09-11. The researcher
approved **English transcripts, including translations, for the first pilot**,
then approved proceeding with this preparation and filtering plan. Pilot
calibration still determines the final acceptance rules; no production filters
have been applied. The existing non-YouTube baseline remains 1,467,709,789 stored
cl100k_base tokens; no YouTube tokens have been added to that retained total.

## Verified inventory

The local inspection sample matches the report's SHA-256. All 439 sampled
locations have valid shard/row provenance and the pinned dataset revision
`9addbabbfcd7409acbcd11a3b59ec2aef6da7eb0`.

| Measurement | Result |
|---|---:|
| Raw transcript rows | 22,684,737 |
| Distinct source video IDs | 3,156,666 |
| Exact `en` language-label rows | 3,262,750 |
| Distinct IDs with an `en` row | 3,127,432 (99.07% of all IDs) |
| Source-reported words in `en` rows | 6,664,116,242 |
| English-label word length: median / p90 / p99 | 598 / 5,698 / 18,119 |
| Longest English-label row | 194,423 reported words |
| Extra rows in repeated video/language groups, all languages | 947,226 across 28 groups |
| Missing license values, all languages | 965,728 (4.26%) |
| Missing channel IDs, all languages | 965,721 |

These are metadata counts, not exact-unique text counts or token totals. The
publisher's current card gives slightly different totals; the pinned download
manifest and measured Parquet inventory define this run. There is no evidence
here of an incomplete download.

## Expected token yield: planning scenarios

Use approximately **100 million additional tokens** as a provisional planning
number, not a measured yield. A working range of roughly **40–200 million**
would put the combined retained corpus around **1.51–1.67 billion tokens**.
This range is judgment for planning, not a statistical confidence interval;
the eventual yield may fall outside it. The inspection sample is too small
and has too few reviewed security-positive examples to establish retention.

The 58 exact-English samples have 141,766 reference tokens and 130,298 reported
words, or 1.088 tokens per word in aggregate. Applying that rough ratio to the
6,664,116,242 reported English words suggests about **7.25 billion raw English
reference tokens before cleaning or deduplication**. This is a sample-based
conversion, not a full-corpus token count or a formal survey estimate; shard
weights, length mix, and translation mix may change the ratio.

| Assumed final share of raw English tokens retained | Added tokens | Combined corpus |
|---|---:|---:|
| 0.5% | 36M | 1.504B |
| 1% | 73M | 1.540B |
| 2% | 145M | 1.613B |
| 3% | 218M | 1.685B |

These percentages describe hypothetical **final token retention after all
cleaning, deduplication, relevance, and usability decisions**. They are not
selected thresholds, measured retention, or percentages of videos. Useful
conference talks can be much longer than irrelevant clips, so document
retention and token retention must be estimated separately. Replace this
planning number with exact candidate token counts and a weighted, reviewed
retention estimate after the pilot; report any cross-source dedup losses too.

## Language and metadata interpretation

The seven dominant language labels account for all but 6,421 rows. English
rows include 2,183,071 labeled original-English rows, 941,703 with another
non-null original-language label, and 137,976 with unknown original language.
These labels are supplied metadata, not independently verified language.

There are 382 distinct transcription-language labels, including region codes
and suspicious values such as `en-GHEw9DUond8`. Do not interpret this as 382
actual languages or silently normalize every `en-*` value to English. Preserve
unrecognized and compound labels for inspection; final language eligibility
must account for actual text. The existing pilot export uses exact `en` labels
because every English-labeled row in this diagnostic sample has that value.
This does not define the final handling of dialect or malformed labels.

`source_language` contains `metadata`, `detection`, and `unknown`; it is an
identification-method field in practice, not a language code. Some shards use
`language_id_method` instead. The three observed physical schemas must be
handled without confusing these fields with `original_language`.

## What the sample demonstrates

The 439 examples contain 8 blank transcripts and 1,417,863 cl100k_base tokens.
Their supplied character counts all match actual text lengths. There are no
nonempty exact-text duplicates within this sample; that says nothing definitive
about full-corpus text duplication. One draw per shard has unequal inclusion
probabilities because shard sizes range from 19,405 to 112,474 rows.

All 58 exact-English sample rows were exported locally as unlabeled pilot
inputs: 41 have original-language label `en`, and 17 have other labels. They
contain **141,766 reference tokens**. This is neither a corpus token estimate
nor a retained-token count. The longest English sample is 15,306 reference
tokens; Qwen token lengths must be measured separately for inference.

The following are diagnostic observations, not human ground-truth labels.
Short entries were read completely; long-entry observations below come from
inspected passages and must be checked across the full English counterpart.
Sample indices are zero-based; the JSONL line number is index + 1.

| Sample / video ID | Observation | Consequence for the pilot |
|---|---|---|
| 236 / `5thPD-K9fRk` | *Automated Triage Collection at Scale in the AWS Cloud*: Italian sample includes concrete incident-response collection workflows, EC2/SSM/S3 design choices, and tradeoffs. | Retrieve its English counterpart as a useful-content candidate. Check technical terms damaged by transcription or translation. |
| 360 / `ec56KiYcBws` | Kali/Crunch wordlist tutorial title, but French text is only four words including a music marker. | A relevant title cannot supply missing instructional content. Check English counterpart independently. |
| 61, 89, 313, 369, 374 | Technical Q&A-style titles; full sampled text is largely introductions, subscription requests, music markers, or incoherent speech. | Title relevance and apparent tutorial format are insufficient. Judge whether the transcript actually explains the solution. |
| 226 / `3y_ip0gRKZQ` | English-labeled Autodesk installation text contains unrelated phrases and broken technical narration. | English labels and technical vocabulary do not establish usability. |
| 29 / `FMxa2cXphEo` | *Cyber Strike Deck Profiles And Combos* is about a card game. | Include misleading security-related wording among negative examples. |
| 24 / `Wmq0pHj20mE` | Long live-coding session includes extensive setup chatter and screen-dependent discussion. | Measure whole-text coverage and useful content proportion; do not decide from the opening or a single favorable passage. |
| 426 / `Ei3PUbnjwQg` | Data-governance talk discusses privacy/security alongside general metadata architecture. | Use as a boundary case for supporting technical material; policy remains to be reviewed. |

Several sampled metadata-poor records repeat the same video IDs across
languages. The 28 repeated video/language groups are anomalous enough to inspect
before production. **Do not delete 947,226 rows on ID equality alone.** Group
by raw-text SHA-256, report conflicting texts/metadata within each ID/language,
and retain all source locations in the duplicate map. The precise cause and
amount of compute saved remain unverified until that pass.

The source carries a Creative Commons attribution string on 21,719,009 rows.
Keep that observed value and preserve nulls. The dataset-level license claim
does not reconstruct missing per-video attribution metadata. Resolve release
handling separately from semantic scoring; do not silently fill missing values
or infer a specific CC-BY version from the source string.

## Proposed filtering implementation

1. **Prepare English candidates on CPUs.** Preserve raw text and provenance;
   account separately for null/empty text. Compute exact text hashes and
   cl100k_base counts. Resolve repeated-ID groups by comparing text and retain
   variant lineage. No channel, keyword, minimum-word, or popularity gate.
2. **Score actual transcript content with an LLM.** Evaluate security relevance,
   technical substance, text usability, and whether a record mixes useful and
   irrelevant material. Save each dimension and supporting evidence separately.
   Keep the transcript out of the instruction role. Titles/channels are omitted
   from the proposed scoring prompt to avoid supplying nonexistent content.
3. **Cover long records completely.** Fit short transcripts in one request.
   Split longer ones into contiguous spans with source character offsets and
   contextual overlap. Determine span size using the model tokenizer and prompt
   budget. Benchmark 4K and 8K input-token budgets as execution settings, not
   document eligibility thresholds. Account for every non-overlap span; missing
   or failed spans keep the document undecided. Do not stitch unrelated snippets.
4. **Apply the reviewed selection policy downstream.** A parse failure, context
   overflow, uncertain judgment, or unprocessed span is never an automatic drop.
   Save prompt/model/tokenizer revisions and raw responses. Audit coverage of
   exact-unique candidates and propagate content-only decisions through the
   duplicate map while preserving each source's attribution.

An 8B model evaluating these dimensions in one pass is the baseline to test.
A cheap relevance pass followed by a stronger quality judge is an alternative
only if the pilot demonstrates adequate recall and a useful compute saving.
False negatives in an early gate cannot be recovered by the second model, so
review random rejects as well as keeps and uncertain cases.

## Pilot plan

The available 58 English examples are enough to exercise prompt formatting,
long inputs, JSON parsing, and a rough throughput measurement. They do not
provide enough known security-positive examples to validate a classifier.
Do not treat unreviewed sample rows or another model's labels as ground truth.

For rubric development, retrieve the English counterparts of the observed
useful-content, misleading-title, translation-damage, and boundary cases above.
Add reviewed security-positive examples and an independent random English
sample before evaluating recall or retention. Keep source/video duplicates and
translations together when separating development and held-out data; prefer
channel separation where feasible. Enriched examples diagnose errors but do
not alone estimate corpus prevalence or precision. With potentially rare useful
content, a small random sample cannot tightly bound the false-positive rate.

Compare the existing **Qwen3-8B** stack with **Qwen3-32B** as a larger comparison
model on the same reviewed examples. These are proposed engineering baselines,
not a claim that they are the best current classifiers. Pin both model and
tokenizer revisions before inference. Start with thinking disabled and bounded
structured output; investigate systematic failures before changing model size.
Both model cards document vLLM support and controllable thinking modes:
[8B](https://huggingface.co/Qwen/Qwen3-8B),
[32B](https://huggingface.co/Qwen/Qwen3-32B).

Begin the 8B benchmark with one H100, then test the larger model with memory
and input-length settings established by a smoke run. Do not request a GPU
array yet. Report input/output tokens, end-to-end throughput, span counts,
parse failures, disagreement cases, useful-content misses, and false keeps.
Use those measurements and deduplicated candidate lengths to size production.
No credible GPU-hour or retained-token total is available yet.

The existing QA scorer is not a YouTube scorer: its task/prompt and long-input
handling must not be reused unchanged. No GPU pilot was submitted during this
review, and the retired `pm05` templates are not instructions for this run.

## Reproduce the local inspection

From the repository's Mac environment:

```bash
venv/bin/python scripts/youtube/inspect_profile.py \
  --profile-dir reports/youtube/profile-v1 \
  --output-dir reports/youtube/review-v1 \
  --pilot-language en
```

Outputs: `sample_analysis.json`, `sample_features.jsonl`, and
`pilot_available.jsonl` under `reports/youtube/review-v1/`. These are analysis
artifacts; the script creates no production keep/drop labels. The proposed
prompt is in `docs/youtube_classifier_prompt.md`.
