# Expansion results and next Marlowe job

**Target correction, 2026-09-14:** 3B is now aspirational. The researcher prefers
a smaller research-quality, publishable corpus to a larger weak one. Arithmetic
below remains valid, but statements about a minimum reflect the earlier plan.

**Follow-up:** the 32B v3 results and targeted repair have transferred and been
reviewed. See the [repair findings](youtube_repair_review.md) for current status;
no repeat GPU job is needed now. The token inventory and v2 history below remain
valid. Submission instructions below are historical.

Reviewed 2026-09-14 after transfer of jobs 482460 (Primus recovery) and 482462
(YouTube Qwen3-8B v2). Both completed successfully. The web inventory provides
enough candidate volume to pursue the 3B target. The YouTube output is now
structurally valid, but its classification quality is not ready for a full run.

## Token budget

All corpus counts below use cl100k_base. They are separate from Qwen inference
token counts. The baseline uses its stored lengths; the web profiler recomputed
text lengths and hashes on Marlowe.

| Measurement | Tokens |
|---|---:|
| Existing retained baseline, 659,147 records | 1,467,709,789 |
| Additional retained tokens required for 3B | 1,532,290,211 |
| RedSage raw candidates, exact-unique within source | 11,445,384,960 |
| Primus raw candidates, exact-unique within source | 2,469,346,082 |
| Duplicate tokens shared by both web sources | 206,461,947 |
| Combined web candidates after cross-source exact dedup | 13,708,269,095 |
| Web candidate tokens matching baseline text | 255,135 |
| Novel exact-unique web candidate tokens | **13,708,013,960** |

About **11.18%** of the novel web pool would need to survive cleaning, relevance
and quality filtering, and near-deduplication to close the gap without YouTube.
This is the required retention fraction, **not a forecast**. The retained corpus
count has not increased yet; neither completed job produced final additions.

The two web sources share 241,408 exact texts. Another 1,848 unique texts match
the baseline. Whole-text equality does not remove repeated page sections,
reformatted copies, translations, or other near duplicates.

## What was verified

Local verification checked both download reports against pinned manifests,
profile/config/sample checksums, coverage of all 1,724 data shards, per-shard raw
hash correspondence, sample probabilities, and comparison-to-profile bindings.
It recomputed every field, content hash, and token count in all **6,896 diagnostic
samples** (4,927,728 tokens), including Primus's actual `content` field. It also
checked comparison arithmetic. Full raw datasets and web metadata sidecars stay
on Marlowe; their complete aggregates were not independently recomputed locally.

For YouTube, all **487 complete texts / 637 spans / 1,140,848 candidate tokens**
were checked. The pinned tokenizer reproduced each request hash and rendered
prompt. Every saved response was reparsed, each evidence ID resolved to exact
source text, and all focal spans covered their document without gaps or overlap.
Regenerated document reports match the transferred reports. No spans are
unresolved; eight documents contain an uncertain label. Final selection remains
null for every document.

Reproducible local audit: `reports/expansion-review-v1/verify.py` and
`integrity.json`. The audit reads the transferred reports and the previously
cached pinned 8B tokenizer. It does not fetch bulk data or certify label accuracy.

## Content findings

The 8B model assigned central relevance to 2 spans and supporting relevance to
40. These are model outputs, not approved keeps. Several supporting judgments
invent a cybersecurity connection that the cited text does not establish:

| Candidate hash prefix and focal character range | Observed error |
|---|---|
| `076a09289214`, 22367–43478 | Hemp, emissions, and sustainable agriculture treated as supporting cybersecurity through resource management. |
| `5f4e79cd194d`, 0–16435 | Fictional SCP containment and anomalous phenomena treated as cybersecurity mechanisms. |
| `bd05cc5ee010`, 21576–42860 | Pharmacogenomics and statin response treated as cybersecurity because genetic data could require protection. |
| `c9a5acf566e7`, 0–19952 | Universal jurisdiction over atrocities treated as supporting cybersecurity without a concrete technical connection. |

Actual source evidence was inspected for these cases. The model also finds useful
material, including Linux performance analysis and packet processing. It still
confuses relevance with usability: for example, the music-theory transcript
`05134ee23860` is labeled unusable with a rationale about being off-topic.
These diagnostic observations are assistant review, not researcher-labeled
ground truth or a measured precision/recall estimate. The mixed random/enriched
pilot cannot support an unweighted corpus-yield extrapolation.

The web samples also require independent content assessment. RedSage's upstream
`relevant=true` stratum contains 5.736B tokens; its null stratum contains 5.710B.
The [RedSage dataset card](https://huggingface.co/datasets/RISys-Lab/RedSage-CFW)
describes both cybersecurity filtering and general educational replay. Neither
the flag nor its absence is a sufficient keep/drop rule. Inspected true-flag
examples include trauma counseling advertised as critical-incident response,
job listings, and cryptocurrency promotion. The null stratum also contains
substantive computing material such as balanced-tree instruction.

Primus samples include useful security explanations alongside unrelated pages,
homework prompts, tiny book previews, and long concatenated category pages.
For example, `data/01257.jsonl.gz` row 262 is a 29,903-token WebTitan category
page. These are failure examples, not estimates of source-wide prevalence.
All sampled and profiled page-license fields are absent in both sources; keep
that absence explicit and preserve source provenance for release review.

## Next job: same sample, larger model

`scripts/youtube_filter/pilot_32b_v2.sbatch` runs the previously planned
[Qwen3-32B comparison](https://huggingface.co/Qwen/Qwen3-32B), pinned to
`9216db5781bf21249d130ec9da846c4624c16137`. It uses the same texts, v2 prompt,
segment evidence, inference budgets, and installed Python environment as 8B.
Only the model and tensor-parallel configuration change. This isolates model
capacity as a variable; a larger model is not automatically a trusted judge.

The allocation requests **2 GPUs, 8 CPUs, 128 GB RAM, up to 2 hours**, on the
working `marlowe-m000091` / `preempt` account and partition. Marlowe documents
[80 GB H100 GPUs](https://marlowe-research.stanford.edu/documentation/specs/);
two GPUs provide headroom for BF16 weights and inference memory. The job first
caches roughly 66 GB of new model weights **on Marlowe**. Queue delay and runtime
are not guaranteed. No package reinstall or dataset download is required.

On the **Mac**, copy the small script directory:

```bash
rsync -av --exclude='.venv/' --exclude='__pycache__/' \
  /Users/natedemchak/Desktop/security-corpus/scripts/youtube_filter/ \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/security-corpus/scripts/youtube_filter/
```

Then on **Marlowe**:

```bash
cd /scratch/m000091/natedem/security-corpus
export YOUTUBE_DATA_DIR=/scratch/m000091/natedem/youtube-transcripts
bash scripts/youtube_filter/run.sh --check-gpu-env && \
mkdir -p logs/youtube-filter && \
sbatch scripts/youtube_filter/pilot_32b_v2.sbatch
```

No separate `srun` allocation is needed. After submission, the batch job is
independent of the terminal or tmux session. Logs are
`logs/youtube-filter/youtube-qwen32b-v2-JOBID.log`. Outputs go to the separate
`$YOUTUBE_DATA_DIR/pilot-qwen32b-v2/` directory. Resubmit this same script after
preemption to reuse valid decisions; preserve the code/config for resumption.

After completion, run on the **Mac**:

```bash
rsync -av \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/youtube-transcripts/pilot-qwen32b-v2/ \
  /Users/natedemchak/Desktop/security-corpus/reports/youtube/pilot-qwen32b-v2/
```

Review both models' outputs against actual text, including negative cases,
before selecting the production classifier. Web classification needs its own
source-appropriate prompt and representative sample; feeding web pages into
the transcript rubric would not validate that workflow. Final keep rules and
near-deduplication settings remain researcher decisions. No production thresholds
or source-specific exclusions were introduced by this review.

Local validation of the new job used the actual pinned 32B tokenizer without
weights: all 487 documents produced 637 spans within the context budget. Its
rendered requests are byte-identical to the completed 8B requests (1,833,777
prompt tokens; largest prompt 5,969). Bash syntax validation passed, and the
unit suite passed with 338 tests; 10 tests requiring bulk corpus data were
excluded. GPU inference for this model still needs the Marlowe run.
