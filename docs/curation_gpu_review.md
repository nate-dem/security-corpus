# Transferred preparation and GPU review — 2026-09-15

Both jobs completed: **486312** prepared inference inputs and **486326** ran the
GPU follow-up. The Mac revalidated manifest/configuration totals and reparsed raw
model responses with matching input, prompt and model bindings. Source Parquet
files remain on Marlowe, so their individual bytes were not rehashed on the Mac.
The next scorer rechecks each selected partition before use.

## Candidate inventory

| Inference read source | Exact text/kind work units | cl100k_base tokens | Parquet partitions |
|---|---:|---:|---:|
| Primus-FineWeb | 3,386,733 | 2,469,346,082 | 2,384 |
| RedSage-CFW | 12,979,432 | 11,238,923,013 | 6,398 |
| YouTube-Commons English, including translations | 2,944,120 | 7,312,515,096 | 2,205 |
| Total | 19,310,285 | 21,020,784,191 | 10,987 |

Web exact duplicates already share an inference read pointer; all aliases remain
in lineage. Across web/transcript prompt kinds, one additional duplicate of 697
tokens remains. Across-kind unique total is 19,310,284 texts / 21,020,783,494 tokens.
Baseline overlap, near duplicates, semantic filtering and release-source precedence
still affect final additions. These numbers are not retained tokens. Baseline:
**659,147 documents / 1,467,709,789 recomputed working tokens**.

## GPU quality findings

- 32B critic: all 90 control spans and all 347 additional spans parsed with valid
  evidence. 8B additional screen: 323/332 parsed; nine evidence failures remain
  review items. No silent discard or acceptance of those failures.
- On the original 89 controls, the critic recovered all 18 assistant-eligible
  cases but admitted 22 assistant-noneligible cases. Joint 8B/critic eligibility
  recovered 16/18 and admitted 11 noneligible cases. Model agreement is insufficient.
- On the 300 additional diagnostic texts, both stages provisionally accepted
  48 (34 Primus, 13 RedSage, one transcript), excluded 177 and left 75 in review.
- The assistant then read all 48 accepted texts: **20 eligible candidates,
  20 requiring review/repair, eight excluded for limited substance**. This was
  prediction-visible development review, not independent human labeling or a
  representative yield/precision estimate. The old model verdicts remain unchanged.

Examples include accepted course objectives without the promised instruction;
missing key-space quantities in a watermarking abstract; a reverse-engineering
write-up with fused GDB commands; a trojan description interrupted by generic
false-positive support instructions; and a phishing article endorsing an absolute
authentication guarantee. Useful short material was also retained in the assistant
assessment: balanced-tree explanations, TLS cipher-mismatch diagnosis, archive
semantics, audit-record fields and tracing API examples. No new length threshold,
topic keyword list or automatic source exclusion was introduced.

Quality concerns are source-text findings and requests for claim/extraction review,
not exhaustive external fact-checks. Missing images alone are not a defect when
the words supply the explanation. Historical content and offensive techniques
are not inherently low quality.

## Next execution

**Current:** job 486937 completed both model tasks. The bundles and raw-response
evaluations were verified. See [the new comparison review](first_batch_review.md)
and [the single-GPU quality recheck](../scripts/curation/QUALITY_PASS.md).
The earlier failures below are historical; do not resubmit their repairs.

**Historical, job 486877:** both tasks passed compiler and Ninja checks. The early
FlashInfer sampling test then failed with `fatal error: curand.h: No such file
or directory`, before model loading (41/43 seconds). Logs are in
`reports/curation/first-batch-v3-logs/`. The system toolkit cannot find the cuRAND
development header required by this optional random sampler. The next launcher
sets `VLLM_USE_FLASHINFER_SAMPLER=0`, a supported vLLM option. The startup check
verifies actual native dispatch and known outputs for native and greedy sampling
at batch sizes 2 and 32. Classifier requests stay at temperature zero; model,
prompt and selection policy are unchanged. Results go to `first-batch-v4`.
Existing dependencies and model weights are reused. Local regression tests do
not establish that full GPU inference works; that still requires the Marlowe run.

**Previous, job 486864:** both tasks passed the CUDA header test using system NVCC
12.9, loaded the models, captured CUDA graphs, and allocated GPU caches. They
then failed during FlashInfer sampler warm-up with
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`.
The package was installed, but the launcher did not put `.venv-next/bin` on PATH.
The correction exposes that directory, checks the selected executable before
submission, and runs a two-vector GPU sampler check before loading either model.
That retry wrote `first-batch-v3`; v1/v2 remain intact. No corpus classification
results have been produced by these attempts. Detailed logs are in
`reports/curation/first-batch-v2-logs/`. The Triton artifact warnings were not the
fatal error: execution continued through graph capture to sampler warm-up.

Historical failure and first repair:

Setup job 486840 completed. Both tasks in array 486841 failed in DeepGEMM NVCC
compilation with `fatal error: nv/target: No such file or directory`; neither
reached classification. Weights loaded successfully using 33.42 GiB for the MoE
and 27.67 GiB for the dense model. The compiler could not see the installed CCCL
headers. The retry exposes the wheel's include tree, tests a small CUDA compilation
before model loading, and records compiler/header bindings. Reuse the installed
environment and weights, keeping the failed v1 outputs; new results go to
`first-batch-v2`. This fixes the observed search-path omission; full GPU inference
remains to be validated. Transferred logs are in
`reports/curation/first-batch-v1-logs/`.

The user's model-choice question prompted a current model review. Test the official
FP8 Qwen3.6-35B-A3B and Qwen3.8-27B checkpoints on the **same** compact v4 prompt
and inputs, one H100 per model. The new prompt removes repetitive demonstrations
and emphasizes actual supplied substance, literal extraction, and concrete claim
defects. Its schema requires a single evidence line and conditionally valid
quality fields; the strict parser independently checks them. Prior prompt versions
are unchanged. Scope, policy vocabulary and final-selection status are unchanged.

Each task runs 137 assistant-reviewed controls plus 4,187 complete candidate
documents from three prepared partitions (4,841,659 reference tokens). It preserves
raw responses, immutable model revisions, tokenizer fingerprints, code/configuration
bindings and source offsets. Final selection remains null. Different prompt/runtime
settings mean comparison with historical Qwen3 numbers is not a model-only ablation.
No broad production launch is justified until actual output quality and throughput
are checked. See [FIRST_BATCH.md](../scripts/curation/FIRST_BATCH.md) for commands.

Local evidence is under `reports/curation/gpu-review-v1-local-review/` and
`reports/curation/accepted-audit-v1/`. Review packets/annotations contain source
text and remain outside Git; the sync helper transfers only those small control
files to Marlowe. Original transferred artifacts remain intact.

Remaining release work: validated expansion selection, repair/section processing
where needed, cross-source exact/near deduplication, baseline ID/provenance repairs,
source attribution and redistribution treatment, final Parquet/token recount,
dataset card/manifest, and Hugging Face publication. No new retained additions
have been claimed.
