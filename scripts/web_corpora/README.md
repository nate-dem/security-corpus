# RedSage-CFW and Primus-FineWeb on Marlowe

**Current Marlowe state (2026-09-14):** both downloads, RedSage `profile-v1`,
Primus `profile-v2`, and the baseline overlap comparison are complete and their
reports have been reviewed. Do not resubmit `job.sbatch` or
`resume_primus.sbatch` for the current snapshot. See the
[completion plan](../../docs/finish_plan.md) and
[measured inventory](../../docs/expansion_results_review.md). The setup and
submission commands below document the workflow already executed.

The researcher approved evaluating and incorporating qualifying records from
both sources on 2026-09-11. The scope is **cybersecurity plus directly supporting
technical material**, including substantive operating systems, networking,
cloud infrastructure, and software engineering explanations. The final corpus
has an **aspirational 3B retained cl100k_base token target**, clarified on
2026-09-14. Research quality takes priority and smaller releases are acceptable.
The existing baseline has 1,467,709,789 stored tokens; another 1,532,290,211 would
reach that target, but this is not a requirement to relax any quality decision.

This workflow downloads all source payloads, profiles actual text, and measures
exact overlap. It does not yet clean or semantically filter records, produce a
canonical web connector, choose numeric thresholds, or change the baseline.
Upstream classifier labels remain diagnostic fields. In particular, RedSage's
general educational material is not automatically eligible.

## Pinned inventory

Public Hub API inventories were checked on 2026-09-11. Every data file and the
upstream README are enumerated in `manifests/`, with exact byte sizes and
upstream checksums. This is the complete authored snapshot, not Hugging Face's
separate automatic Parquet conversion.

| Source | Immutable revision | Data files | Download bytes including README |
|---|---|---:|---:|
| [RedSage-CFW](https://huggingface.co/datasets/RISys-Lab/RedSage-CFW) | `52b393cb49c59789c5cae272acc0dbc58c0831ff` | 125 Parquet | 34,623,961,891 |
| [Primus-FineWeb](https://huggingface.co/datasets/trendmicro-ailab/Primus-FineWeb) | `41058dbfa82fd0b11a12605cad17139ca57558d4` | 1,599 JSONL.gz | 4,407,484,859 |

About **39 GB raw** in total. Plan roughly **100 GB of available project storage**
for raw data, diagnostic sidecars, and temporary aggregation files; this is a
storage allowance, not a measured final footprint. Later cleaned outputs and
near-deduplication may need more space. Project quota matters as well as global
filesystem free space. This pipeline stores datasets under scratch, separate
from the source checkout.

## 1. Accept access conditions

While signed into the Hugging Face account you will use on Marlowe, open both
dataset links in the table and accept their access conditions. Both currently
require sharing contact information. A token alone does not accept these terms.

Both cards list an ODC-By dataset license. Primus explicitly retains the source
websites' applicable terms. The script preserves the upstream README and
supplied page license fields; missing page licenses remain missing. Access to a
dataset does not by itself settle redistribution of every page in the final
Hugging Face corpus. Carry source attribution through subsequent assembly.

## 2. Copy from the Mac

Run in your **Mac terminal**:

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/marlowe/sync_all.sh
```

This resumes the existing whole-project transfer, including the retained
baseline needed for comparison. It copies new/changed files without deleting
remote files and excludes local Python environments, credentials, and caches.
Raw YouTube and web downloads outside the checkout are unaffected.

For subsequent **code-only** updates after the corpus is already present:

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/marlowe/sync_code.sh
```

Do not update implementation files while a profile job is running. Profiles bind
to code/dependency versions; intentional algorithm changes require a new profile
directory. Plain restarts use the same files and configuration.

## 3. Set up and preflight on Marlowe

On the **Marlowe login node**:

```bash
cd /scratch/m000091/natedem/security-corpus
python3 -m venv scripts/web_corpora/.venv
scripts/web_corpora/.venv/bin/python -m pip install -r scripts/web_corpora/requirements.txt
scripts/web_corpora/.venv/bin/hf auth login
```

Paste a Hugging Face token into the interactive login prompt. It must have read
access to both gated repositories (including the appropriate gated-repository
permission if using a fine-grained token). Answer **no** to adding it as a Git
credential; these are HTTP dataset downloads. Do not put the token into scripts,
commands, or this conversation. Keep the same `HF_HOME`/`HF_TOKEN_PATH` settings
for login and job execution; the workflow does not override credential paths.

Then run these inexpensive checks **before entering the queue**:

```bash
export WEB_DATA_DIR=/scratch/m000091/natedem/web-corpora
bash scripts/web_corpora/run.sh --check-env && \
bash scripts/web_corpora/run.sh download --plan-only && \
bash scripts/web_corpora/run.sh download --check-access
```

The environment check verifies the exact Python packages and packaged tokenizer
without network access or bulk data reads. It avoids the missing-dependency
failure seen on the YouTube profiling job. Python 3.10+ is sufficient for this
workflow; no full project installation, CUDA, or PyTorch is needed. Set
`WEB_PYTHON` to an absolute interpreter path if using a different environment.
The access check makes one small authenticated HEAD request per dataset. It
must print `Access OK` for both sources before submission.

## 4. Submit the CPU job

From the **project root** on Marlowe:

```bash
mkdir -p logs/web-corpora
sbatch scripts/web_corpora/job.sbatch
```

This requests **4 CPUs, 16 GB RAM, no GPUs, up to four hours** using the account
and partition that worked for YouTube: `marlowe-m000091` / `preempt`. The
`marlowe-m000091-pm05` association previously had `GrpSubmitJobs=0`; it is not
used. The four hours are a scheduling limit, not a runtime promise. Full token
counting can require more than one allocation.

The job runs these stages in order:

1. Download and verify all payloads and source READMEs, with two HTTP workers.
2. Scan every record with four CPU processes, computing exact `cl100k_base`
   token counts and SHA-256 text hashes. Preserve missing/blank/non-string text
   counts and upstream metadata; do not use upstream relevance as a filter.
3. Compare exact text across the two sources and with the explicit retained
   baseline in `compare_baseline.sh`. Baseline hashes are computed from its
   current text; baseline token totals are still stored counts. This is not a
   fresh full baseline release audit.

`sbatch` continues after SSH disconnects; tmux is optional. Monitor using:

```bash
squeue -u natedem
ls -t logs/web-corpora/*.log | head
# Replace JOBID with the number printed by sbatch:
tail -f logs/web-corpora/web-corpora-JOBID.log
```

If you prefer to run inside tmux instead of submitting a batch job:

```bash
srun --account=marlowe-m000091 --partition=preempt \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=4:00:00 \
  bash scripts/web_corpora/run.sh full
```

Use one submission method at a time. Keep tmux on a specific login host if using
it; sessions on `login-01` will not appear on `login-03`.

## 5. Resume or run stages separately

After a timeout or preemption, **resubmit the same batch command**. Completed
downloads and profiles are reused. Download retries rehash raw files against
upstream checksums and resume partial HTTP transfers. Profiling rechecks raw
hashes and cached output checksums; an interrupted shard is rescanned without
retokenizing other completed shards. No automatic resubmission loop is installed.

For a specific stage, pass its name:

```bash
sbatch scripts/web_corpora/job.sbatch download
sbatch scripts/web_corpora/job.sbatch profile
sbatch scripts/web_corpora/job.sbatch compare-baseline
```

These examples are alternatives; do not submit all three simultaneously.
To isolate a failed source:

```bash
sbatch scripts/web_corpora/job.sbatch profile --source primus-fineweb
```

If comparison reports missing baseline files, rerun `sync_all.sh` from the Mac,
then submit only `compare-baseline`; raw downloads and profiles remain usable.
To compare only the two web sources without a baseline, use the `compare` stage.
Its report explicitly has no baseline comparison.

Changing sample size, seed, profile code, or dependencies requires a different
profile directory, for example `profile --profile-name profile-v2`; pass that
same profile name to comparison. Worker count may change between restarts.

## 6. Bring back the reports

Once both `PROFILE COMPLETE` messages appear and the comparison report has
`complete: true`, run on the **Mac**:

```bash
cd /Users/natedemchak/Desktop/security-corpus
mkdir -p reports/web-corpora
rsync -av \
  --include='/*/' --include='/*/profile-v1/' \
  --include='/*/profile-v1/summary.json' \
  --include='/*/profile-v1/inspection_sample.jsonl' \
  --include='/comparison-v1.json' --exclude='*' \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/web-corpora/ \
  reports/web-corpora/
```

Reports provide raw and exact-unique token counts, within-source duplicate
counts, cross-source overlap, baseline overlap, missing metadata, token-length
distributions, upstream relevance strata, top source domains, and full-text
diagnostic samples. Each sample has source shard, zero-based row, revision, and
selection probability. Four uniformly sampled records per shard is a diagnostic
default, not a filtering threshold or a representative unweighted yield sample.

The local fixtures test the input formats, including Primus's `content` field.
The completed profiles and 6,896 transferred full-text samples have been checked
locally; bulk raw data remain on Marlowe. Unexpected schemas or malformed JSON
fail explicitly instead of silently dropping records. Semantic filtering and
final retained outputs still need implementation and validation.

Next: review actual samples and RedSage's upstream relevance strata, calibrate
the content classifier against the approved scope, measure throughput on a GPU
pilot, then run semantic filtering and near-deduplication across selected web,
YouTube, and baseline material. Only retained, deduplicated tokens count toward
3B. Novel exact-unique raw tokens are an upper bound, not a final yield forecast.

## References

- [Marlowe Slurm accounts, partitions, and preemption](https://marlowe-research.stanford.edu/documentation/slurm/)
- [RedSage source card](https://huggingface.co/datasets/RISys-Lab/RedSage-CFW)
- [Primus source card](https://huggingface.co/datasets/trendmicro-ailab/Primus-FineWeb)
