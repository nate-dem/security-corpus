# Source quality recheck — 2026-09-15

Job **487003 finished all five packets**, then returned exit code 2 because
one response remained unresolved. This was an output-generation failure, not
a CUDA, dependency, or allocation failure. **1,171 of 1,172 spans parsed** across
1,015 documents. The completed work is available for review.

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
The GPU behavior of the compact setting still needs the Marlowe retry;
local tests verify configuration, selection, resume, and provenance mechanics.
See [the current commands](../scripts/curation/QUALITY_PASS.md).

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
The unresolved transcript is also held for review.

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

The next GPU job is a one-document formatting retry, using the already working
environment and cached model. The completed 1,015-document quality run should
not be repeated. Its total generation time was about 372.5 seconds, within the
user-reported 9m36s job; a one-document retry will still incur model startup.

Production scoring, reviewed selection, cross-source exact/near deduplication,
release assembly and publication checks remain outstanding. The working
baseline remains **1,467,709,789 cl100k_base tokens**; no new YouTube or web tokens
have been added to that retained count. Quality remains the criterion for
inclusion, with 3B aspirational.
