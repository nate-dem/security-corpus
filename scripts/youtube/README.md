# Download and profile every YouTube-Commons shard on Marlowe

**2026-09-14:** downloads and metadata profiling are complete. The next step is
the new full English CPU preparation job; see the
[current run commands](../../docs/curation_protocol.md#next-marlowe-commands).
That job uses the project checkout and existing web CPU environment. The
standalone download/profile instructions below remain historical setup guidance.

The download and metadata-profile scripts can run independently of the rest of security-corpus.
Python 3.10+ and the small Hugging Face Hub dependency are sufficient for the
download. The optional CPU profiling step adds DuckDB and PyArrow. Neither
step requires PyTorch, CUDA, or a tokenizer.

The default dataset snapshot is
`PleIAs/YouTube-Commons@9addbabbfcd7409acbcd11a3b59ec2aef6da7eb0`.
Its live inventory was checked on 2026-09-10: **439 Parquet shards,
162,854,932,262 bytes (162.85 GB / 151.67 GiB)**. Allow at least 200 GB of
available project storage for this raw download; later normalized outputs need
additional space. The script enumerates every Parquet file in the snapshot,
including nested paths, rather than assuming 439 numbered filenames.

This stage downloads the complete raw data. It applies **no channel, language,
length, keyword, or LLM filtering** and does not generate normalized records.

## 1. Copy the directory

To restore the **whole project**, including the local corpus and reports, run
this from the Mac:

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/marlowe/sync_all.sh
```

It copies to `/scratch/m000091/natedem/security-corpus/` and excludes Mac Python
environments, Git history, local credentials, and caches. It never deletes
remote files and can be rerun to resume. With that layout, use
`/scratch/m000091/natedem/security-corpus/scripts/youtube` wherever the standalone
instructions below say `~/youtube-download` or `$HOME/youtube-download`.
The filesystem path is `/scratch/m000091/`. The commands below use the working
`marlowe-m000091` account and `preempt` partition. On 2026-09-10, the researcher
confirmed that `marlowe-m000091-pm05` has `GrpSubmitJobs=0`, which blocks new
submissions even with an empty queue. The base account successfully downloaded
and verified all 439 shards.

To copy **only the standalone downloader directory** instead:

From the Mac, assuming your existing SSH alias is `marlowe`:

```bash
rsync -av --exclude=.venv --exclude=__pycache__ \
  /Users/natedemchak/Desktop/security-corpus/scripts/youtube/ \
  marlowe:youtube-download/
```

Alternatively, copy this directory using your usual file-transfer tool. The
remaining commands assume it is at `~/youtube-download` on Marlowe.

## 2. Set up a small environment and start tmux

On Marlowe's login node:

```bash
cd ~/youtube-download
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
tmux new -s youtube-download
```

If the available Python is older than 3.10, first load an available Python
module or your existing Python environment. GPU/CUDA modules are unnecessary.

Inside tmux, still on the login node:

```bash
cd ~/youtube-download
export YOUTUBE_DATA_DIR="/scratch/m000091/${USER}/youtube-transcripts"
mkdir -p "$YOUTUBE_DATA_DIR"

# Small metadata-only request: shows the pinned shard inventory and size.
bash run_download.sh --plan-only

# Sustained download + SHA-256 verification runs on a CPU allocation.
srun --account=marlowe-m000091 --partition=preempt \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=4:00:00 \
  bash "$HOME/youtube-download/run_download.sh"
```

Use a different `YOUTUBE_DATA_DIR` if your allocated scratch/project path
differs. The script's default is the same scratch path shown above. Do not put
the dataset in your home directory, whose documented quota is only 32 GB.
Verify your project quota separately: filesystem-wide free space does not
guarantee that your allocation has enough remaining quota.

**No GPU is requested.** The four-hour limit is a scheduling allowance, not an
estimate of download time. The same command resumes after preemption or time
limit expiration.

Marlowe's operating guidance permits moving data on login nodes but directs
sustained work to compute nodes. This full transfer and hashing pass fits that
category. Keep tmux on the login node and let `srun` execute the downloader on
the allocated node. Tmux survives SSH disconnects; it does not extend a Slurm
job's time limit or protect it from preemption.

Detach with **Ctrl-B, then D**. Reconnect after logging in again:

```bash
tmux attach -t youtube-download
```

## 3. Progress, completion, and retries

The downloader logs to both the terminal and
`$YOUTUBE_DATA_DIR/download.log`. From another terminal:

```bash
tail -f "/scratch/m000091/${USER}/youtube-transcripts/download.log"
```

Successful completion prints `COMPLETE: 439/439 shards verified` for the
default snapshot and exits 0. `download-summary.json` records `complete: true`,
the snapshot, manifest checksum, verified shard count, total bytes, and each
file's SHA-256. A missing or failed shard makes the run exit nonzero.

If interrupted or a transfer fails, **rerun the same `srun` command**. It:

- Reuses the original `manifest.json`; it does not follow a moving `main`.
- Rehashes completed files and skips those that match upstream checksums.
- Resumes partial HTTP downloads using `raw/.cache/huggingface/`.
- Replaces files whose size or content checksum fails verification.
- Retries individual failed transfers and reports unresolved failures.

Do not remove `raw/.cache/` while the download is incomplete. Do not run two
downloaders against the same output directory; an OS lock prevents this.
Ctrl-C may wait for active HTTP workers to stop. On a hard job termination,
the summary can remain marked `running`; only `complete: true` proves that all
files passed the preceding run. Rerun to verify and finish.

The script uses two concurrent HTTP transfers with Xet acceleration disabled
to keep CPU use modest and avoid an additional Xet cache. There is no second
full copy of the dataset in your home Hugging Face cache. Change concurrency
with `--workers N` only alongside an appropriate CPU allocation.

For a later **offline** full recheck without changing or downloading files:

```bash
srun --account=marlowe-m000091 --partition=preempt \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=4:00:00 \
  bash "$HOME/youtube-download/run_download.sh" --verify-only
```

There is no need to run that immediately after successful download: the
download command already verifies every file. Checksums verify source bytes;
transcript parsing, corpus profiling, normalization, and quality filtering
are subsequent stages.

## 4. After download: profile the complete inventory

The next step inventories every row's metadata and exports one complete
transcript per nonempty shard for diagnostic inspection. It reports record,
video, channel, language, license, and source-reported length statistics.
Repeated video/language keys are counted separately; these do not prove text
duplication. No rows are excluded from the metadata inventory. Raw shards are
unchanged. This is not normalized ingestion, full-text deduplication, token
counting, or LLM classification.

For the existing full-project installation, copy the updated scripts from the
Mac (no need to transfer the corpus again):

```bash
rsync -av --exclude='.venv/' --exclude='__pycache__/' \
  /Users/natedemchak/Desktop/security-corpus/scripts/youtube/ \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/security-corpus/scripts/youtube/
```

On Marlowe, inside the existing tmux session:

```bash
cd /scratch/m000091/natedem/security-corpus/scripts/youtube
export YOUTUBE_PYTHON="$PWD/.venv/bin/python"
export YOUTUBE_DATA_DIR=/scratch/m000091/natedem/youtube-transcripts

# Install and check the same interpreter the job will use before joining the queue.
"$YOUTUBE_PYTHON" -m pip install -r requirements-profile.txt && \
bash run_profile.sh --check-env && \
srun --account=marlowe-m000091 --partition=preempt \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=4:00:00 \
  bash run_profile.sh
```

For a standalone installation, use its directory instead of the full-project
path. The wrapper uses this directory's `.venv/bin/python`; set
`YOUTUBE_PYTHON` to use another environment with the profiling dependencies.
The lightweight `--check-env` check does not read corpus data or need a compute
allocation. The downloader's `requirements.txt` does not include DuckDB or
PyArrow; profiling needs the separate `requirements-profile.txt` installation.
If `ModuleNotFoundError` appears before profiling starts, install into the
interpreter printed by the wrapper, verify it, then resubmit. A failed import
does not modify the downloaded shards or start a profile.

Successful completion prints `PROFILE COMPLETE` and writes:

- `$YOUTUBE_DATA_DIR/profile-v1/summary.json`: aggregate statistics and actual
  per-shard schemas. Length sums and percentiles use source-reported words and
  characters, **not cl100k_base token counts**. Missing and invalid supplied
  counts are reported; invalid counts do not contribute to length aggregates.
- `$YOUTUBE_DATA_DIR/profile-v1/inspection_sample.jsonl`: complete, untruncated
  text with metadata, shard/row provenance, revision, and selection probability.
  The default yields up to 439 examples. Selection is deterministic within each
  shard; different shard sizes mean different sampling probabilities. This is
  an inspection sample, not a labeled classifier evaluation set or an
  unweighted estimate of the whole corpus.
- `metadata/`, `samples/`, and `checkpoints/`: per-shard products for restart.

The profiler requires a complete matching download report and checks current
file sizes. It relies on the downloader's SHA-256 verification; it does not
repeat the full 163 GB hash pass. On resume it checks raw file size/mtime and
checksums its own cached outputs. Changed source files fail explicitly;
missing or corrupt profile outputs are rebuilt. Rerun the same `srun` command
after interruption. Completed shards are reused; final aggregation is rerun.
Only `summary.json` with `complete: true` establishes a completed profile.

`--samples-per-shard N` changes diagnostic sample size, not training eligibility.
Use a new `--output-dir` if changing the seed, sample size, code, or dependency
versions; existing profiles are bound to their configuration. Allow additional
scratch space for metadata, samples, and DuckDB temporary files.

After completion, copy only the two review artifacts back to the Mac:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/youtube/profile-v1
rsync -av --include='/summary.json' --include='/inspection_sample.jsonl' --exclude='*' \
  natedem@login.marlowe.stanford.edu:/scratch/m000091/natedem/youtube-transcripts/profile-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/youtube/profile-v1/
```

The profile has been reviewed and the researcher approved an English pilot,
including translations, covering security plus directly supporting technical
material. Continue with the [content-classification pilot](../youtube_filter/README.md).
Production keep/drop rules remain open; the teammate's metadata/channel and
50-word filters are not applied.

## Output layout

```text
youtube-transcripts/
  manifest.json              # Immutable upstream revision, files, sizes, hashes
  download-summary.json      # Complete/incomplete state and verification results
  download.log               # Append-only progress log
  profile-v1/                # CPU metadata inventory and diagnostic sample
  raw/
    cctube_0.parquet
    ...                      # Every Parquet file from the manifest
    .cache/huggingface/       # Transfer state; retain for partial-download resume
```

Keep the raw shards and manifests for ingestion. Scratch is not backed up;
copy completed work to approved persistent project storage or another backup
before deleting scratch files. The downloader never cleans up completed data.

## Verified documentation

- [Marlowe login/compute-node guidance](https://marlowe-research.stanford.edu/documentation/violations/#do-not-degrade-the-shared-nodes)
- [Marlowe Slurm partitions and limits](https://marlowe-research.stanford.edu/documentation/slurm/)
- [Marlowe filesystems and quotas](https://marlowe-research.stanford.edu/documentation/getting-started/filesystems/)
- [Pinned YouTube-Commons snapshot](https://huggingface.co/datasets/PleIAs/YouTube-Commons/tree/9addbabbfcd7409acbcd11a3b59ec2aef6da7eb0)
