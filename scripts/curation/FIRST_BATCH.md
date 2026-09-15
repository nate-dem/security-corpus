# Current run: model comparison and first complete partitions

**Retry after job 486877:** both tasks passed the compiler and Ninja checks, then
the early FlashInfer sampler check failed because system NVCC could not find
`curand.h`. The launcher now sets `VLLM_USE_FLASHINFER_SAMPLER=0`, vLLM's supported
native sampling backend. Classification still uses temperature-zero greedy
decoding. Startup verifies native dispatch and known GPU outputs for both the
warm-up sampler and greedy decoder before loading model weights. The existing
CCCL/compiler and Ninja checks remain. No reinstall or model download is needed.
The retry uses **`first-batch-v4`**; preserve failed v1/v2/v3 outputs. Local tests
cover routing and failure handling; full GPU inference still needs to complete
on Marlowe.

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
bash scripts/curation/submit_first_batch.sh --reuse-setup
```

The script checks the completed preparation manifest, controls, setup report,
installed versions, header location and `ninja --version`, then submits only the
two-task GPU array. The startup sampler check uses synthetic logits at batch sizes
2 and 32, exercising native PyTorch/Triton paths and greedy decoding. Both must
return alternating token IDs 7 and 99; it does not process corpus data.
Compiler, Ninja and sampler details are recorded with the run configuration.

- Comparison: **1 GPU, 8 CPUs, 96 GB RAM per task, at most 4 hours per task**;
  at most two tasks concurrently, so at most eight allocated GPU-hours. Each
  model loads once and stays resident across the five input packets. Four hours
  is a cap, not a measured runtime prediction.

For a fresh installation only, omit `--reuse-setup`: the script first submits a
4-CPU/32-GB/4-hour setup job and makes the GPU tasks depend on its success. That
setup downloads 68.33 GB of weights to the scratch model cache and installs
`scripts/curation/.venv-next`; allow about 130 GB for all dependencies and models.
Job 486840 already completed this work.

The current transferred GPU log reports NVIDIA driver 580.173.02 and CUDA 13.0.
The pinned vLLM 0.29.0 PyPI build uses CUDA 13.0; Transformers is pinned at 5.10.4.
Both models loaded successfully (33.42 and 27.67 GiB of GPU memory respectively).
The retry logs identify the system compiler as `/usr/local/cuda-12.9/bin/nvcc`;
the CCCL preflight passed with this compiler. Graph capture and GPU cache allocation
also completed before the missing-Ninja error. Earlier Triton cache warnings were
followed by those successful stages; they were not the terminating exception.
The driver version is not the system NVCC version. Existing YouTube/web
environments, model caches and corpus files remain available.
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
mkdir -p /Users/natedemchak/Desktop/security-corpus/reports/curation/first-batch-v4
rsync -av \
  --include='/*/' --include='/*/summary.json' \
  --include='/*/bundle-summary.json' --include='/*/review-bundle.tar.gz' \
  --exclude='*' \
  natedem@login-01.marlowe.stanford.edu:/scratch/m000091/natedem/curation/first-batch-v4/ \
  /Users/natedemchak/Desktop/security-corpus/reports/curation/first-batch-v4/
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

The compiler fix uses NVIDIA's documented
[NVCC environment flags](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-environment-variables).
The pinned vLLM build's [DeepGEMM compiler](https://github.com/deepseek-ai/DeepGEMM/blob/8b1392b978f5a03c828dd1711090d7fb50958b8a/csrc/jit/compiler.hpp)
honors `DG_JIT_NVCC_COMPILER`; the runner pins that to the compiler tested by the
preflight. No model weights, precision, prompt, scope or quality policy changed.

Python's [virtual-environment documentation](https://docs.python.org/3/library/venv.html#how-venvs-work)
distinguishes selecting an interpreter from adding installed executables to PATH.
FlashInfer's [Ninja launcher](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/cpp_ext.py)
invokes the bare `ninja` command, which requires the latter.

vLLM 0.29's [sampler backend selection](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/sample/ops/topk_topp_sampler.py)
supports `VLLM_USE_FLASHINFER_SAMPLER=0`. Its
[greedy decoding path](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/sample/sampler.py)
returns argmax before random sampling when all requests are greedy. This flag
only changes the random-sampling backend; other FlashInfer operations used by
the model are unaffected. No general stochastic-output equivalence is claimed.
