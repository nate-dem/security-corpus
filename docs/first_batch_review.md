# First completed model comparison — 2026-09-15

Job 486937 completed both model tasks. The transferred archives match their
declared sizes and SHA-256 hashes. Both copied summaries match the archived
summaries. All ten evaluations were recomputed from bound packets, requests and
raw responses; they matched the transferred evaluations before adding the new
critic support. The GPU sampler/compiler setup now works on Marlowe.

**Prefer Qwen3.8-27B-FP8 for the next assessment. Do not use either model's
single-pass eligible label as final corpus inclusion.** The faster MoE model
misses much of the approved supporting-computing material. The 27B model recovers
more useful examples, but still accepts damaged or questionable text.

## Comparison evidence

| Observed diagnostic | Qwen3.6-35B-A3B-FP8 | Qwen3.8-27B-FP8 |
|---|---:|---:|
| Route agreement on 137 assistant development controls | 70 | 91 |
| Recovered controls previously marked eligible (38 total) | 15 | 28 |
| Eligible controls previously marked exclude or review | 9 | 21 |
| Generation time for the same 4,187 candidate documents | 19.3 min | 28.1 min |
| Unresolved responses across 4,767 total spans | 39 | 2 |

Controls influenced prompt development and references are assistant-authored;
these are disagreements with a development reference, not population precision
or recall estimates. Both models used the same v4 prompt, source documents,
non-thinking temperature-zero generation, 8,192-token context and one H100.
The MoE's higher rejection rate is not evidence of better quality filtering.

The 27B model's candidate routes were:

| Source partition | Documents | Eligible candidates | Review items | Eligible candidate tokens |
|---|---:|---:|---:|---:|
| Primus 1785 | 1,094 | 357 | 117 | 197,523 |
| RedSage 2686 | 2,048 | 211 | 91 | 184,701 |
| YouTube 9211 | 1,045 | 23 | 43 | 47,657 |

These are one partition per source, not representative corpus yields. The
429,881 provisionally eligible tokens are **not new retained corpus tokens**.
The retained working baseline remains 1,467,709,789 tokens.

The two 27B unresolved responses stopped at the output limit, one in RedSage and
one in YouTube. They stay review items. The older experiment's `complete: true`
means the comparison workflow finished; its separate
`scoring_coverage_complete: false` correctly records the unresolved spans.
The next job exits 2 and still produces a review bundle if coverage is incomplete.

## Source inspection

Disagreement review confirmed examples where the 27B model's rationale names an
actual explanation but its substance label is `limited`, including a short
truth-table definition and a concrete SNMP UI procedure. Conversely, the model
accepted a passage that joins OAuth authorization-code injection to a generic
arbitrary-code-execution description without recognizing the mechanism mismatch.
Historical, attributed observations must not be rejected merely because they
are not absolute proofs; some reference/model differences require judgment.

A deterministic source/route-stratified inspection list contains 24 candidates.
Twelve were read in full and recorded separately; the longer twelve have not
received full source review. This was post-prediction inspection, not a blinded
or representative accuracy audit. Concrete findings include:

- `primus-fineweb:data/01001.jsonl.gz:472`: accepted `fnmatch` reference has a
  truncated declaration, missing flag names and incomplete sentences. Intact
  return-value prose does not fix the whole document.
- `youtube-commons:cctube_173.parquet:48322`: accepted tutorial refers to the
  essential auto-editor command in the video description; that command is absent
  from the transcript. This cannot be silently reconstructed.
- `redsage-cfw:chunk_1/train-00006-of-00026.parquet:5384`: a short abstract states
  a concrete isolation/cost tradeoff and an attributed packet-processing latency
  result. Abstract form and short length alone are not disqualifiers.
- `youtube-commons:cctube_173.parquet:48940`: rejected garbled, unrelated
  commentary does not supply a computing/security explanation.

Full findings, case IDs and text hashes are in
`reports/curation/first-batch-v4/local-review/inspection-notes.json`. Source text
and raw responses remain outside Git. No external factual verification of every
source claim is implied.

## Next run and compute implications

The [second-pass runbook](../scripts/curation/QUALITY_PASS.md) uses the cached 27B
model on 1,015 documents / 1,200,868 reference tokens: all 137 controls, all 591
eligible candidates, all 251 review items, and 12 deterministic rejections per
source. All text is preserved. The new source-only critic uses literal-source
checks and synthetic demonstrations to address observed damage and substance
errors. It sees no previous verdict. The output budget is 1,536 tokens instead
of 1,024; long source documents are still covered by contiguous spans. These are
development changes, so agreement between the two calls is not independent
validation. Disagreements and old unresolved items remain open.

Full first-pass generation at the measured source-specific token throughput
would take roughly **2,195 H100-hours** with 27B: Primus 395, RedSage 1,337 and
YouTube 462. This is an extrapolation from one partition per source, not a quote
or a scheduling guarantee. It excludes model startup, preparation, second-pass
review, retries, near deduplication and publication. Sixteen continuously
allocated GPUs would imply roughly 5.7 days for that generation alone. Measure
broader workload variation and confirm available capacity before a full launch.
Prioritizing Primus and YouTube can produce useful additions while avoiding an
immediate commitment to the much larger RedSage pass.

Before release, independently reviewed samples, scalable production sidecars,
selection/section handling, cross-source near deduplication, baseline provenance
repairs and redistribution/attribution checks remain required. This run neither
publishes data nor changes approved scope or quality policy.

## Artifact hashes

| Model bundle | SHA-256 |
|---|---|
| qwen36-moe-fp8 | `55ba2acd620db9b390966e26424c17a3b0330b306fad187677d7686b2e88b3f3` |
| qwen38-27b-fp8 | `551144fad28944febb6d86a0472ed9d86880abf121335cf7bb7bf176df9581dc` |

Recomputed evaluations and arithmetic are saved under
`reports/curation/first-batch-v4/local-review/`. Extracted originals are under
`reports/curation/first-batch-v4/verified/`; original archives remain intact.
