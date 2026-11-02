"""Check build tools, expose CCCL headers, and optionally exercise the sampler.

The compiler check needs no GPU; check_sampler explicitly executes a tiny GPU
kernel without model weights. Neither check downloads or installs anything.
NVIDIA documents
NVCC_PREPEND_FLAGS for adding -I flags to child compiler invocations. Discover
headers from wheel metadata because CUDA 13 wheels changed their directory layout.
"""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile


PROBE = """#include <nv/target>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/std/type_traits>
#include <cub/block/block_reduce.cuh>
#include <cutlass/half.h>
__global__ void header_probe(__half* output) {
    if (threadIdx.x == 0) output[0] = __float2half(1.0f);
}
"""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_ninja():
    """Check the executable child processes use, not just its Python package."""
    found = shutil.which("ninja")
    expected = Path(sysconfig.get_path("scripts")) / "ninja"
    if found is None:
        raise RuntimeError(
            f"ninja is not on PATH; prepend {expected.parent} before submitting"
        )
    path = Path(found).resolve()
    if not expected.is_file() or path != expected.resolve():
        raise RuntimeError(
            f"Expected this environment's ninja at {expected}, found {path}"
        )
    # Use the same bare command that FlashInfer launches.
    version = subprocess.run(
        ["ninja", "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()
    if not version:
        raise RuntimeError("ninja --version returned empty output")
    return {"path": str(path), "version": version, "sha256": sha256(path)}


def check_sampler():
    """Exercise vLLM's native warm-up and greedy paths before model loading.

    The optional FlashInfer random sampler needs cuRAND development headers
    absent from Marlowe's system toolkit. vLLM 0.29 supports opting out; our
    temperature-zero classification uses the same greedy argmax either way.
    Check the selected dispatch, not a directly called fallback that could hide
    a different backend being selected when the actual engine starts.
    """
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError(
            "Set VLLM_USE_FLASHINFER_SAMPLER=0 before importing vLLM; "
            "use run_first_batch.sh"
        )
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    from vllm.v1.sample.sampler import Sampler

    if not torch.cuda.is_available():
        raise RuntimeError("Sampler preflight requires a GPU allocation")
    sampler = TopKTopPSampler()
    if sampler.forward != sampler.forward_native:
        raise RuntimeError("vLLM did not select the native sampler; stop before model loading")
    print(
        "Sampler preflight: checking vLLM native and greedy outputs at batch sizes 2 and 32",
        flush=True,
    )
    checks = []
    # Native top-k/top-p dispatch uses PyTorch for small batches and Triton for
    # batches >= 8. Both are exercised, including our configured batch size 32.
    for batch_size in (2, 32):
        expected = [7, 99] * (batch_size // 2)
        logits = torch.full((batch_size, 128), -100.0, dtype=torch.float32, device="cuda")
        for i, token in enumerate(expected):
            logits[i, token] = 1.0
        k = torch.full((batch_size,), 1, dtype=torch.int32, device="cuda")
        p = torch.full((batch_size,), 1.0, dtype=torch.float32, device="cuda")
        greedy = Sampler.greedy_sample(logits).cpu().tolist()
        result, _ = sampler(logits, generators={}, k=k, p=p)
        observed = result.cpu().tolist()
        if observed != expected or greedy != expected:
            raise RuntimeError(
                f"Native sampler preflight returned native={observed}, greedy={greedy}; "
                f"expected {expected}"
            )
        checks.append({"batch_size": batch_size, "native": observed, "greedy": greedy})
        del logits, result, k, p
    del sampler
    torch.cuda.empty_cache()
    print(
        "Sampler preflight passed: native and greedy outputs matched at both batch sizes",
        flush=True,
    )
    return {
        "version": "vllm-native-sampler-preflight-v1",
        "vllm_version": importlib.metadata.version("vllm"),
        "backend": "forward_native",
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
        "gpu": torch.cuda.get_device_name(0),
        "checks": checks,
    }


def cccl_include():
    dist = importlib.metadata.distribution("nvidia-cuda-cccl")
    roots = {
        Path(dist.locate_file(p)).resolve().parent.parent
        for p in dist.files or ()
        if str(p).endswith("/nv/target")
    }
    required = ("nv/target", "cuda/std/type_traits", "cub/block/block_reduce.cuh")
    roots = [p for p in roots if all((p / name).is_file() for name in required)]
    if not roots:
        raise RuntimeError("Installed nvidia-cuda-cccl lacks a complete include tree")
    # CUDA 13.3 provides a cccl subtree plus forwarding headers above it.
    root = min(roots, key=lambda p: (p.name != "cccl", str(p)))
    return root, dist.version


def nvcc_path():
    # Match the pinned DeepGEMM CUDA-home discovery and compiler override.
    explicit = os.environ.get("DG_JIT_NVCC_COMPILER")
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    found = shutil.which("nvcc")
    path = Path(
        explicit
        or (
            str(Path(home) / "bin/nvcc")
            if home
            else found or "/usr/local/cuda/bin/nvcc"
        )
    )
    path = path.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"DeepGEMM CUDA compiler is unavailable: {path}")
    return path


def configure_and_check():
    ninja = check_ninja()
    include, version = cccl_include()
    nvcc = nvcc_path()
    # DeepGEMM constructs its NVCC command as shell text. Marlowe paths have no
    # whitespace; fail explicitly instead of generating ambiguous compiler flags.
    if any(c.isspace() or c in "'\"`$;|&<>\\" for p in (include, nvcc) for c in str(p)):
        raise ValueError(
            "CUDA compiler/header paths must be shell-safe paths without whitespace"
        )
    flag = f"-I{include}"
    previous = os.environ.get("NVCC_PREPEND_FLAGS", "")
    if flag not in previous.split():
        os.environ["NVCC_PREPEND_FLAGS"] = f"{flag} {previous}".strip()
    os.environ["DG_JIT_NVCC_COMPILER"] = str(nvcc)
    vllm = importlib.metadata.distribution("vllm")
    cutlass = Path(vllm.locate_file("vllm/third_party/deep_gemm/include")).resolve()
    if not (cutlass / "cutlass/half.h").is_file():
        raise RuntimeError(f"Vendored DeepGEMM CUTLASS header missing: {cutlass}")
    print(f"CUDA compiler preflight: {nvcc}; CCCL {version} at {include}", flush=True)
    compiler_version = subprocess.run(
        [str(nvcc), "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="curation-cuda-probe-") as folder:
        probe = Path(folder) / "probe.cu"
        output = Path(folder) / "probe.cubin"
        probe.write_text(PROBE)
        result = subprocess.run(
            [
                str(nvcc),
                str(probe),
                "--cubin",
                "-std=c++20",
                "-arch=sm_90a",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                f"-I{cutlass}",
                "-o",
                str(output),
            ],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode or not output.is_file() or not output.stat().st_size:
            raise RuntimeError(
                "CUDA header compilation failed before model loading. "
                "Preserve the log; do not reinstall models.\n"
                + result.stdout
                + result.stderr
            )
    print(
        "CUDA compiler preflight passed (H100 cubin compiled; no GPU execution).",
        flush=True,
    )
    return {
        "version": "cccl-include-preflight-v1",
        "helper_sha256": sha256(Path(__file__)),
        "nvcc": str(nvcc),
        "nvcc_sha256": sha256(nvcc),
        "nvcc_version": compiler_version,
        "ninja": ninja,
        "cccl_version": version,
        "cccl_include": str(include),
        "nv_target_sha256": sha256(include / "nv/target"),
        "cutlass_half_sha256": sha256(cutlass / "cutlass/half.h"),
        "probe_sha256": hashlib.sha256(PROBE.encode()).hexdigest(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_HOME",
                "CUDA_PATH",
                "DG_JIT_NVCC_COMPILER",
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
                "NVCC_CCBIN",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
            )
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(configure_and_check(), indent=2))
