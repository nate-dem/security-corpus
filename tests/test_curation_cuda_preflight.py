import os
from pathlib import Path
import subprocess
import sys
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
    monkeypatch.setattr(cuda, "check_ninja", lambda: {"version": "fixture"})
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


def test_ninja_must_be_visible_to_child_processes(tmp_path, monkeypatch):
    scripts = tmp_path / "venv/bin"
    scripts.mkdir(parents=True)
    ninja = scripts / "ninja"
    ninja.write_text("#!/bin/sh\nprintf '1.13.2\\n'\n")
    ninja.chmod(0o755)
    monkeypatch.setattr(cuda.sysconfig, "get_path", lambda key: str(scripts))
    monkeypatch.setenv("PATH", str(tmp_path / "unrelated"))
    with pytest.raises(RuntimeError, match="not on PATH"):
        cuda.check_ninja()
    monkeypatch.setenv("PATH", str(scripts))
    report = cuda.check_ninja()
    assert report == {
        "path": str(ninja),
        "version": "1.13.2",
        "sha256": cuda.sha256(ninja),
    }
    wrong = tmp_path / "wrong/bin"
    wrong.mkdir(parents=True)
    (wrong / "ninja").write_bytes(ninja.read_bytes())
    (wrong / "ninja").chmod(0o755)
    monkeypatch.setenv("PATH", f"{wrong}:{scripts}")
    with pytest.raises(RuntimeError, match="Expected this environment"):
        cuda.check_ninja()


def test_launcher_exposes_environment_tools_with_absolute_python(tmp_path):
    """Reproduce the missing-PATH condition in the actual shell launcher."""
    scripts = tmp_path / "scripts/curation"
    scripts.mkdir(parents=True)
    source = Path(cuda.__file__).with_name("run_first_batch.sh")
    launcher = scripts / source.name
    launcher.write_bytes(source.read_bytes())
    bin_dir = scripts / ".venv-next/bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\ncommand -v ninja\nninja --version\n")
    python.chmod(0o755)
    ninja = bin_dir / "ninja"
    ninja.write_text("#!/bin/sh\nprintf 'fixture-ninja\\n'\n")
    ninja.chmod(0o755)
    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["/bin/bash", str(launcher), "--check-env"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [str(ninja), "fixture-ninja"]


@pytest.mark.parametrize("observed", [[7, 99], [7, 98]])
def test_sampler_preflight_requires_known_gpu_outputs(monkeypatch, observed):
    assigned = {}
    freed = []

    class Logits:
        def __setitem__(self, index, value):
            assigned[index] = value

    def full(shape, value, **kwargs):
        assert shape == (2, 128) and value == -100.0
        assert kwargs == {"dtype": "float32", "device": "cuda"}
        return Logits()

    def sample(logits, **kwargs):
        assert assigned == {(0, 7): 1.0, (1, 99): 1.0}
        assert kwargs == {"top_k": 1, "top_p": 1.0}
        return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: observed))

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            full=full,
            float32="float32",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _: "fixture H100",
                empty_cache=lambda: freed.append(True),
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.sampling",
        SimpleNamespace(top_k_top_p_sampling_from_logits=sample),
    )
    monkeypatch.setattr(cuda.importlib.metadata, "version", lambda _: "0.6.18")
    if observed == [7, 99]:
        assert cuda.check_sampler()["observed"] == observed
        assert freed == [True]
    else:
        with pytest.raises(RuntimeError, match="expected \\[7, 99\\]"):
            cuda.check_sampler()
