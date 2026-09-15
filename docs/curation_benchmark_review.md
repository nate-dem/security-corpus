# Curation comparison review — 2026-09-14

## Result and next action

Job **486147** produced both model comparisons. Transferred requests,
configuration hashes, raw responses and source/reference bindings were checked
locally by reparsing all decisions. **Neither model is approved for bulk semantic
selection from this run.** Do not rerun `benchmark.sbatch` unchanged.

| Diagnostic on the 89 development cases | Qwen3-8B | Qwen3-32B |
|---|---:|---:|
| Parsed, evidence-valid responses | 87 | 75 |
| Invalid repeated quality concerns | 2 | 14 |
| Eligible / exclude / review | 1 / 40 / 48 | 0 / 45 / 44 |
| Action agreements with assistant assessment | 54 | 54 |
| Assistant-eligible cases also model-eligible | 0 / 18 | 0 / 18 |
| Generation seconds, excluding model load | 147.86 | 134.35 |
| Input / output model tokens | 174,017 / 32,654 | 174,017 / 29,343 |

The 32B run used two GPUs; the 8B run used one. Generation seconds are from this
small, short-document workload, not production throughput estimates. Neither
agreement nor apparent yield estimates accuracy: these are assistant-authored
development references, and the prompt has been adjusted after inspecting them.
The reference routes remain 18 eligible, 43 exclude, 28 review; no references
were changed to make either model look better.

## Observed failures

1. **Presentation boundaries were mistaken for source damage.** The v2 prompt
   split text at approximately 400 characters. Responses repeatedly claimed a
   sentence was missing when its continuation was in the following segment.
   This affected otherwise readable vulnerability explanations and articles.
2. **The model did not consistently follow the approved computing scope.**
   Several programming explanations were marked absent because they were not
   directly about cybersecurity. Creative coding was a conspicuous example.
3. **Quality flags were sometimes invented or explicitly inapplicable.**
   Responses cited ordinary speech, lack of legal disclaimers, or a claim's
   failure to address every possible limitation. Some flags explicitly said the
   concern did not apply. The clean forensic-triage control was held partly for
   informal speech and an imagined security guarantee about parallel collection.
4. **An array allowed repeated concern categories.** The parser correctly held
   these responses unresolved; a fixed-key output can prevent that shape error.
5. **The sole 8B eligible case was not an acceptable success.** It was a short
   permutation-theory fragment with no actual computing/security application
   explained, plus an apparently damaged group-size formula. The model inferred
   relevance from a possible cryptographic connection.

## Changes prepared

`rubric_v3.py` keeps source lines intact, explicitly tells the model to read the
whole passage, uses `domain_relevance` wire labels for security/computing, and
maps them to the existing policy labels. A fixed object contains exactly three
quality checks, each with `not_identified`, `identified`, or `uncertain` status.
No identified issue requires empty evidence and reason. Identified and uncertain
issues require cited focal lines and block eligibility. Short authored
examples distinguish readable informal computing instruction, unrelated prose,
an unsupported security guarantee, and truly missing commands. They are prompt
examples, not independent evaluation data. The quality selection policy is unchanged.

A source-line citation can span more than 400 characters, especially in unpunctuated
transcripts. Its exact offsets remain validated, but resolving an ID proves text
location, not the semantic validity of its claim. The next comparison is needed
to check whether these changes fix the observed behavior without missing bad data.

`prepare.py` is an independent CPU step over **all three downloaded sources**.
It verifies completed upstream artifacts, creates an exact-text inference index,
preserves all valid source aliases, and writes bounded Parquet work units. It
reuses the English-prepared YouTube rows, RedSage profile-v1 and Primus profile-v2.
It does not apply semantic quality filters, pick final publication attribution,
or remove baseline overlap. Web sources share an inference deduplication group;
YouTube remains a separate prompt kind. Cross-kind exact overlap is measured,
not subtracted from inference workload counts. Actual retained tokens are still
unknown. Raw sources are preserved and interrupted source-shard work can resume.

Use the [next-run commands](../scripts/curation/NEXT_RUN.md) to submit both jobs
in parallel. Production GPU scoring, retained assembly, near deduplication,
release provenance/attribution repairs and publication remain unfinished.

## Bindings

- Reviewed packet: `9d03cdea2cf80db4d163aa19839104212f7d17100cca9f52f04684c173c70726`
- Assistant references: `cb45c93d18f7bb0851278516693bee35adcda89d3af93bef00c340e48a1778e2`
- 8B model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- 32B model revision: `9216db5781bf21249d130ec9da846c4624c16137`
- 8B run config: `b603ba49b679dc42072e838d453fe875a7d9c929cd5dbbcdaf514b9d14f7b764`
- 32B run config: `ec66a179834ab418bb67466e36043b4a3d826e0f60ad1cd8303a9764cd393027`
- Original outputs: `reports/curation/marlowe-development-v2/`

Original v2 rubric and responses remain intact. New runs use a separate directory.
