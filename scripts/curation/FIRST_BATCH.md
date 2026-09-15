# Current run: model comparison and first complete partitions

Jobs **486312** and **486326** completed. Their reports have been transferred and
reviewed. Do not repeat preparation or the old 8B/32B diagnostic jobs. See
[the review](../../docs/curation_gpu_review.md).

This run compares **Qwen3.6-35B-A3B-FP8** (throughput candidate) with
**Qwen3.8-27B-FP8** (quality comparator). Each uses **one H100**, with identical
compact v4 prompts, non-thinking JSON output, a batch size of 32, an 8,192-token
context, CUDA graphs, text-only loading and pinned official model revisions.
Each model runs the 89 original controls and the 48 newly reviewed accepted cases,
then **4,187 complete candidate documents / 4,841,659 cl100k_base tokens** from
one prepared partition per source. There is no automatic full-corpus launch.

The source partitions are deterministic workload clusters: Primus index 1785
(1,094 documents), RedSage index 2686 (2,048), YouTube index 9211 (1,045).
They measure actual workload behavior. They are **not** a representative
document sample or a held-out quality benchmark. Control examples influenced
prompt development; references are assistant-authored, not human ground truth.
No existing filter threshold or source scope changes. A new prompt/model is
experimental until its actual outputs are reviewed.

## Copy on the Mac

```bash
cd /Users/natedemchak/Desktop/security-corpus
bash scripts/curation/sync_first_batch.sh
```

The helper uses `natedem@login-01.marlowe.stanford.edu` following the working
login-01 connection. Override `REMOTE_HOST` with your exact working SSH target if
you use an SSH alias. Only code, tokenizer assets and the small control packets
are copied. No bulk data or model weights go to the Mac.

## Submit on Marlowe

```bash
cd /scratch/m000091/natedem/security-corpus
bash scripts/curation/submit_first_batch.sh
```

The script checks the completed preparation manifest and control bindings, then
submits a CPU setup job and a dependent two-task GPU array:

- Setup: **4 CPUs, 32 GB RAM, no GPUs, at most 4 hours**. Installs a separate
  `scripts/curation/.venv-next` and downloads **68.33 GB of model weights** to
  `/scratch/m000091/natedem/curation-model-cache`. Allow about 130 GB additional
  scratch for the environment, installation cache, models and bounded results.
- Comparison: **1 GPU, 8 CPUs, 96 GB RAM per task, at most 4 hours per task**;
  at most two tasks concurrently, so at most eight allocated GPU-hours. Each
  model loads once and stays resident across the five input packets. Four hours
  is a cap, not a measured runtime prediction.

The current transferred GPU log reports NVIDIA driver 580.173.02 and CUDA 13.0.
The pinned vLLM 0.29.0 PyPI build uses CUDA 13.0; Transformers is pinned at 5.10.4.
Actual model/kernel execution must still be checked on Marlowe. Existing
YouTube/web environments, model caches and corpus files remain available.
No Hugging Face login is needed for these publicly available model repositories.

These are `sbatch` jobs; they survive logout and do not require an active tmux
session. Monitor `squeue -u natedem` and `logs/curation/`. If setup fails, the
dependent tasks are cancelled rather than left waiting indefinitely. Return the
setup log before changing dependencies. On inference preemption, resubmit only
the affected array index with the same code/environment/output, for example
`sbatch --array=0 scripts/curation/first_batch.sbatch`. Successful spans resume;
invalid spans remain unresolved and get retried. Never change model/rubric settings
inside an existing run directory.

## Return the small compressed results

After both GPU tasks finish, on the **Mac**:

```bash
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/first-batch-v1
rsync -av \
  --include='/*/' --include='/*/summary.json' \
  --include='/*/bundle-summary.json' --include='/*/review-bundle.tar.gz' \
  --exclude='*' \
  natedem@login-01.marlowe.stanford.edu:/scratch/m000091/natedem/curation/first-batch-v1/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/first-batch-v1/
```

Bundles contain only the bounded experiment's packets, raw responses, requests,
configurations, full installed-package versions and evaluations. Raw 19M-document preparation and model weights
remain on scratch. `bundle-summary.json` binds each archive's size and SHA-256.
If a task fails before creating a bundle, return its log and available summary.

Slurm completion and parsed outputs are not accuracy certificates. Reports show
errors admitted on development controls, missed usable controls, actual generation
time, unresolved spans and provisional document/token routes. Timing is explicitly
per invocation; a resumed run's generation time does not include its earlier work.
No automatic model winner, final selection, release Parquet or new retained-token
claim is produced. The baseline remains **1,467,709,789 retained working tokens**.

## Model and runtime sources

Checked 2026-09-15:

- [Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8):
  35B total / 3B active parameters; official FP8 weights about 37.46 GB.
- [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8): official
  dense-model FP8 weights about 30.87 GB; configurable non-thinking operation.
- [vLLM 0.29.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0):
  CUDA 13.0 PyPI build; newer Qwen architecture support. Versions are isolated
  from the old vLLM 0.13 environment.
- [Qwen3.5/3.6 serving recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
  and [Qwen3.8 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B): text-only serving
  and inference options. Published general benchmarks and parameter counts do
  not prove better filtering or faster throughput on our workload.
