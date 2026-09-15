# V3 model results and independent GPU follow-up

The transferred v3 raw responses were reparsed with matching source, request,
rubric, configuration and assistant-reference bindings. CPU input preparation is
still running, per the researcher; its report has not yet transferred.

| Diagnostic on 89 development cases | 8B v3 | 32B v3 |
|---|---:|---:|
| Parsed, evidence-valid responses | 89 | 88 |
| Eligible / excluded / review | 34 / 38 / 17 | 36 / 34 / 19 |
| Assistant-eligible recovered | 16 / 18 | 16 / 18 |
| Model-eligible, assistant not eligible | 18 | 20 |
| All-route agreement with assistant | 60 / 89 | 57 / 89 |
| Generation seconds, excluding model load | 77.50 | 89.54 |

The source-line presentation and explicit quality status fixed most mechanical
failures and the blanket rejection of readable material. However, the models
still sometimes confuse topic mentions with instruction, overlook broken commands
or missing contrasts, accept unsupported security benefits, or apply the computing
scope too broadly. The 32B model also had one invalid focal-line citation. Neither
run establishes production selection accuracy. The assistant reference is a
development judgment, not independent human ground truth.

8B is the provisional first-pass choice because it produced complete coverage,
recovered the same number of reference-eligible cases and uses one GPU. This is
not approval to use its exclusions or accepts as final corpus selection. 32B is
being tested in a different role with a source-only review prompt rather than
repeating the same prompt and treating agreement as independent evidence.

The review prompt writes a concrete rationale and quality checks before the
verdict fields, distinguishes promised topics from actual explanations, and checks
specific extraction/claim problems. Short authored contrasts demonstrate readable
computing instruction, topic-only promotion, physical estimation without software
instruction, missing commands, and logging versus enforcement. These are prompt
examples, not evaluation records. It uses the existing label/policy vocabulary.

The next GPU job runs the reviewer on the original 89 cases, reuses the saved 8B
screen, and runs both stages on 300 additional texts. The additional packet excludes
the reviewed document hashes, including known parent transcript hashes. Sampling
is deterministic within existing diagnostic pools, not representative sampling
from all 19M+ candidate texts, and earlier pipeline exposure remains documented.
It supplies no reference labels or population yield estimate. Source-only review
also means the critic does not see the 8B answer; it is still another model call,
not an independent human assessment or external fact verification.

The new partition driver is implemented and tested using small Parquet fixtures,
including resumed scoring and a single model instance across partitions. Its
production throughput has not been measured. CPU preparation files are unchanged,
and the GPU follow-up has no dependency on their unfinished outputs.

Use [GPU_REVIEW.md](../scripts/curation/GPU_REVIEW.md) now. Quality review, actual
retained assembly, deduplication, source provenance/attribution repairs and final
publication remain unfinished. No new tokens have been added to the retained
1,467,709,789-token working baseline.

## Input bindings

- Reviewed packet: `9d03cdea2cf80db4d163aa19839104212f7d17100cca9f52f04684c173c70726`
- References: `cb45c93d18f7bb0851278516693bee35adcda89d3af93bef00c340e48a1778e2`
- 8B v3 config: `f79558e849e62de0c26aa4bbf32fa86572aabe74bb0f2410d75271166ec15cca`
- 32B v3 config: `6bce495f3d636b60c6ac73b0d05b7ce0692a6a898654437c69f80ef4180a4d0f`
- Additional packet: `3a116c6b9da789cd7b49bd3946ef45c740a17eef843cebf41ad0d04c7565704a`
- Additional packet size: 300 full texts; 353,943 cl100k_base tokens.
