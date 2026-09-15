# Source quality recheck — 2026-09-15

Job **487038 completed the one-document retry**, and its transferred artifacts
have been verified. The quality recheck now has **1,172 of 1,172 spans parsed**
across 1,015 documents when the explicitly linked retry replaces the failed
assessment. The original job 487003 remains recorded as failed with exit code 2;
its files have not been modified. No further formatting retry is needed.

## Verified retry result

The retry archive is 14,955 bytes, SHA-256
`0eb9a5f2f287de8b40c51cc9342d5ebb2b329dd6f98b0423218d444406266875`.
All ten archived files match the transferred files byte for byte. The outer
configuration hash is
`a747c04b3ce5b98729c9592e086f0b2a5423ac07bc4282dede36efada391fb92`.
The original packet, source, request, decision and configuration bindings were
checked; the retry uses the identical source span and task hash. Its raw response
was re-parsed and exactly reproduces the saved evaluation.

The response ended normally after **277 output tokens**, taking 6.4 seconds of
generation within the reported 2m07s job. It identifies extensive corruption of
technical terms in the ES6/TypeScript transcript and labels it `partly_usable`,
with `damaged_or_missing_content`. The existing policy routes it to
**`review_required`**. Parsing is resolved; the transcript is not accepted into
the corpus. The model's summary is not a corrected replacement for the source.

The first-pass and quality-pass evaluations were re-parsed again before applying
the retry to a separate combined diagnostic report. The other 1,014 document
assessments are unchanged. Local artifacts:

- `reports/curation/quality-retry-v1/local-review/verification.json`
- `reports/curation/quality-retry-v1/local-review/combined-cases.jsonl`, SHA-256
  `1a0084e0a00702e096b006b82f345dd2e2b150a296ba21d4f1261a11202b39a3`

There are no remaining unparsed documents in this quality recheck. The earlier
finding that 303/591 previously eligible candidates need exclusion or further
review is unchanged. Complete parsing does not establish classifier accuracy.

## Verification and failure

The transferred archive is 8,026,413 bytes with SHA-256
`c34308085484c045889797de2aeb72a19efecf56c5a40b2a31a3793c0185a805`.
It was checked against `bundle-summary.json` and safely extracted under
`reports/curation/quality-pass-v1/verified/qwen38-27b-fp8/`. All five evaluations
were regenerated from the raw responses and matched their archived versions.
All five first/second-pass comparisons were also regenerated against the
verified first-batch-v4 outputs and matched. The outer configuration hash is
`4a3bed4421435c04437fc52e8e26ab10723395a0bd583da8022a1da2dd5a9e17`.
Local verification details are in
`reports/curation/quality-pass-v1/local-review/verified-evaluations.json`.

The failed case is `youtube-commons:cctube_173.parquet:47020`, a complete
4,759-character / 932-reference-token transcript comparing ES6 and TypeScript.
Its source hash is
`6feaaba1b5a2b5a446194abadb4072d69d9eff33cf4f296fec7bf2dcd3c9a488`.
The model began a rationale, closed its JSON string early, then repeated literal
carriage returns until it reached the 1,536-token output limit. These are legal
JSON whitespace between fields, so the schema constraint did not end the loop.
The saved raw response has `finish_reason=length`; no labels were accepted.

The retry pins vLLM's `xgrammar` backend and sets
`disable_any_whitespace=True`, which restricts inter-field JSON whitespace.
The option and backend restriction are documented in the
[pinned vLLM 0.29.0 configuration](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/structured_outputs.py).
This removes the observed loop path; it does not guarantee correct semantic
classification or eliminate every possible generation failure. Source text,
rubric, prompt, model revision, output budget and native sampler remain the same.

`retry_quality.py` re-parses the original results and prepares only documents
with unresolved spans. On the transferred artifacts this is **one document and
one span**. A separate output directory records original packet/configuration,
request and decision hashes, and requires identical span keys and task hashes.
It never overwrites the original run or promotes a failed response to eligible.
Job 487038 demonstrates successful GPU execution of the compact setting on the
previously failing case. Local tests cover configuration, selection, resume,
and provenance mechanics. The [runbook](../scripts/curation/QUALITY_PASS.md)
retains the completed commands as historical instructions.

## Quality findings

Model: `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, `critic-v2`, temperature zero.
The second pass reads the supplied source without seeing the first-pass labels.

| Candidate source | Previously eligible | Eligible on both passes | Second pass: exclude | Second pass: review |
|---|---:|---:|---:|---:|
| Primus | 357 | 186 | 44 | 127 |
| RedSage | 211 | 89 | 46 | 76 |
| YouTube | 23 | 13 | 6 | 4 |
| Total | 591 | 288 | 96 | 207 |

Thus **303 of 591 previously eligible candidates were flagged** by the second
pass. These are diagnostic routes, not final deletions or retained counts.
Disagreement between passes remains `review_required` in the combined report.
The retried transcript is also held for review because of its damaged content.

On the 137 development controls, second-pass agreement with existing assistant
references is 104/137, compared with 91/137 for the first pass. Both passes agree
on eligibility for 26 controls: 19 previously reference-eligible and seven
reference-review-required. These controls were used during development, and
the references are assistant judgments. Neither these figures nor agreement
between two calls to the same model establish independent precision or recall.

Inspection of the complete source and associated responses found:

- The critic catches a truncated `fnmatch` flag description in Primus
  `data/01001.jsonl.gz:472` and missing commands in the YouTube auto-editor
  tutorial `cctube_173.parquet:48322`.
- It still admits the AMTSO article `data/01328.jsonl.gz:359`, which introduces
  a capabilities list that is absent from the supplied text. It also admits
  `data/01570.jsonl.gz:812`, whose OAuth authorization-code discussion is paired
  with an arbitrary-code-execution/CWE-94 claim that the described mechanism
  does not substantiate.
- A critical route does not make every rationale correct. In the RedSage GPU
  buying article `chunk_1/train-00006-of-00026.parquet:4151`, the critic assumes
  a particular direction for a connector quality ordering. The supplied wording
  is ambiguous, so that claimed factual error is not established by the text.
- Some reference disagreements remain legitimate judgment calls. For example,
  the flattened Mercurial SSH patch discussion `data/01183.jsonl.gz:164` contains
  a useful wire-protocol explanation alongside damaged diff formatting.

The critic is useful as an additional quality check, but still misses source
damage and can overstate defects. It is not yet a validated automatic release
selector. Repairing the one malformed response addresses execution coverage,
not these remaining semantic limitations. The three candidate partitions are
diagnostic samples, not a representative estimate of source-wide retained yield.

## What this changes

The formatting retry is complete. The environment and cached model now have
verified successful execution with compact JSON. The completed 1,015-document
quality run should not be repeated. Its original generation time was about
372.5 seconds, with another 6.4 seconds for the retry.

Production scoring, reviewed selection, cross-source exact/near deduplication,
release assembly and publication checks remain outstanding. The working
baseline remains **1,467,709,789 cl100k_base tokens**; no new YouTube or web tokens
have been added to that retained count. Quality remains the criterion for
inclusion, with 3B aspirational.
