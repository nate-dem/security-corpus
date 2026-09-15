import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.curation import cuda_preflight as cuda


def installed_headers(tmp_path, monkeypatch):
    include = tmp_path / "nvidia/cu13/include/cccl"
    names = ("nv/target", "cuda/std/type_traits", "cub/block/block_reduce.cuh")
    for name in names:
        path = include / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// fixture header")
    # New wheels also carry a forwarding nv/target; do not select an incomplete tree.
    forwarding = include.parent / "nv/target"
    forwarding.parent.mkdir()
    forwarding.write_text("// forwarding header")
    cutlass = tmp_path / "vllm/third_party/deep_gemm/include/cutlass/half.h"
    cutlass.parent.mkdir(parents=True)
    cutlass.write_text("// fixture cutlass")
    dist = SimpleNamespace(
        version="13.3.4.2.1",
        files=[
            (include / "nv/target").relative_to(tmp_path),
            forwarding.relative_to(tmp_path),
        ],
        locate_file=lambda p: tmp_path / p,
    )
    monkeypatch.setattr(cuda.importlib.metadata, "distribution", lambda name: dist)
    return include


def test_cccl_current_wheel_layout_and_missing_header(tmp_path, monkeypatch):
    include = installed_headers(tmp_path, monkeypatch)
    assert cuda.cccl_include() == (include, "13.3.4.2.1")
    (include / "cuda/std/type_traits").unlink()
    with pytest.raises(RuntimeError, match="complete include tree"):
        cuda.cccl_include()


def test_deepgemm_compiler_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("DG_JIT_NVCC_COMPILER", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    root = tmp_path / "cuda"
    nvcc = root / "bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("fixture compiler")
    nvcc.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(root))
    monkeypatch.setattr(cuda.shutil, "which", lambda _: "/ignored/nvcc")
    assert cuda.nvcc_path() == nvcc
    monkeypatch.setenv("DG_JIT_NVCC_COMPILER", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="compiler is unavailable"):
        cuda.nvcc_path()


@pytest.mark.parametrize("compile_ok", [True, False])
def test_compile_receives_headers_and_reports_real_failure(
    tmp_path, monkeypatch, compile_ok
):
    include = installed_headers(tmp_path, monkeypatch)
    nvcc = tmp_path / "nvcc"
    nvcc.write_text("fixture compiler")
    monkeypatch.setattr(cuda, "nvcc_path", lambda: nvcc)
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-DKEPT=1")
    monkeypatch.delenv("DG_JIT_NVCC_COMPILER", raising=False)
    # Track environment mutations so they cannot leak to another test.
    monkeypatch.setenv("DG_JIT_NVCC_COMPILER", "")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert os.environ["DG_JIT_NVCC_COMPILER"] == str(nvcc)
        assert os.environ["NVCC_PREPEND_FLAGS"] == f"-I{include} -DKEPT=1"
        if "--version" in args:
            return SimpleNamespace(
                stdout="Cuda compilation tools, release 13.0, V13.0.88"
            )
        assert "--cubin" in args and "-arch=sm_90a" in args
        assert "#include <cutlass/half.h>" in Path(args[1]).read_text()
        assert kwargs["cwd"] == str(Path(args[1]).parent)
        if compile_ok:
            Path(args[args.index("-o") + 1]).write_bytes(b"compiled fixture")
        return SimpleNamespace(
            returncode=0 if compile_ok else 1,
            stdout="",
            stderr="fatal error: nv/target",
        )

    monkeypatch.setattr(cuda.subprocess, "run", run)
    if compile_ok:
        report = cuda.configure_and_check()
        assert report["cccl_include"] == str(include)
        assert report["nv_target_sha256"] == cuda.sha256(include / "nv/target")
        assert (
            report == cuda.configure_and_check()
        )  # Stable resume binding, no duplicated flag.
    else:
        with pytest.raises(RuntimeError, match="before model loading.*\n.*nv/target"):
            cuda.configure_and_check()
    assert len(calls) == (4 if compile_ok else 2)
