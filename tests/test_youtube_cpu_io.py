"""Compatibility with frozen audit digests and interrupted production audits."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading

import pytest

from scripts.curation_audit import run as audit
from scripts.curation_single import run as single
from scripts.youtube.profile import _sha256
from scripts.youtube_cpu_io import run as io
from test_curation_audit import (
    finished as _finished,
    single_env as _single_env,
    production_env as _production_env,
)

finished, single_env, production_env = _finished, _single_env, _production_env


def test_identical_digest_inventory_and_order(finished, monkeypatch):
    _, plan, _, _, _ = finished
    directory = single.folder(plan, 0)
    nested = directory / ".hidden/deep"
    nested.mkdir(parents=True)
    (nested / "extra").write_text("Additional evidence is bound too")
    (nested / ".download.lock").write_text("Excluded advisory lock")
    (directory / "file-link").symlink_to(nested / "extra")
    (directory / "directory-link").symlink_to(nested, target_is_directory=True)
    (directory / "dangling-link").symlink_to(nested / "missing")
    expected_files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.name != ".download.lock"
    )
    assert io.part_files(directory) == expected_files
    entry = {"slot": 0, "summary_sha256": _sha256(directory / "summary.json")}
    expected = audit.fingerprint(plan, entry)
    original = audit._sha256
    # Force two reads to overlap, without relying on filesystem timing. The
    # result must retain the serial path order, regardless of completion order.
    barrier = threading.Barrier(2, timeout=10)
    paired = set(expected_files[:2])

    def synchronized(path):
        if threading.current_thread() is not threading.main_thread() and path in paired:
            barrier.wait()
        return original(path)

    monkeypatch.setattr(audit, "_sha256", synchronized)
    assert io.fingerprint(plan, entry, threads=2) == expected
    (nested / "extra").write_text("Changed evidence")
    assert io.fingerprint(plan, entry, threads=2) != expected


def test_timeout_after_last_partition_reuses_frozen_checkpoints(finished, monkeypatch):
    path, plan, output, loads, calls = finished
    original = audit.fingerprint
    visits = 0

    def timeout_in_final_scan(plan, entry):
        nonlocal visits
        visits += 1
        if visits == 5:  # Two parts, each checked before/after its fresh audit.
            raise TimeoutError("Allocation ended during final verification")
        return original(plan, entry)

    monkeypatch.setattr(audit, "fingerprint", timeout_in_final_scan)
    with pytest.raises(TimeoutError):
        audit.execute(path, output)
    monkeypatch.setattr(audit, "fingerprint", original)
    frozen = _sha256(output / "frozen.json")
    saved = {p: _sha256(p) for p in (output / "parts").rglob("*") if p.is_file()}
    code = audit.auditor_code()
    inference = len(loads), list(calls)
    monkeypatch.setattr(
        single, "audit_part", lambda *a: pytest.fail("Cached audit was reparsed")
    )
    with io.parallel_fingerprints(2):
        assert audit.execute(path, output)["audit_complete"]
    assert audit.fingerprint is original
    assert _sha256(output / "frozen.json") == frozen
    assert audit.auditor_code() == code
    assert saved == {p: _sha256(p) for p in saved}
    assert inference == (len(loads), calls)
    raw = next((single.folder(plan, 0) / "work/score/decisions").glob("*.json"))
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Cached audit"):
        with io.parallel_fingerprints(2):
            audit.execute(path, output)
    assert audit.fingerprint is original
    assert not json.loads((output / "review/summary.json").read_text())[
        "audit_complete"
    ]


@pytest.mark.parametrize("changed", ["summary", "source", "unreadable"])
def test_changed_or_unreadable_inputs_stop(finished, monkeypatch, changed):
    _, plan, _, _, _ = finished
    directory = single.folder(plan, 0)
    summary = directory / "summary.json"
    entry = {"slot": 0, "summary_sha256": _sha256(summary)}
    if changed == "summary":
        summary.write_bytes(summary.read_bytes() + b" ")
        error, message = ValueError, "worker summary changed"
    elif changed == "source":
        source = Path(plan["input_dir"]) / plan["parts"][0]["path"]
        source.write_bytes(source.read_bytes() + b" ")
        error, message = ValueError, "Prepared input changed"
    else:
        original = audit._sha256

        def unreadable(path):
            if threading.current_thread() is not threading.main_thread():
                raise OSError("fixture read failure")
            return original(path)

        monkeypatch.setattr(audit, "_sha256", unreadable)
        error, message = OSError, "fixture read failure"
    with pytest.raises(error, match=message):
        io.fingerprint(plan, entry, threads=2)


def test_cli_forwards_arguments_and_restores_on_failure(monkeypatch, caplog):
    from scripts.youtube_finalize import pipeline

    original, arguments = audit.fingerprint, sys.argv

    def target():
        assert sys.argv == [
            "scripts.youtube_finalize.pipeline",
            "--manifest",
            "m.json",
            "--output",
            "out",
        ]
        assert audit.fingerprint is not original
        raise RuntimeError("fixture stop")

    monkeypatch.setattr(pipeline, "main", target)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="fixture stop"):
        io.main(
            ["--threads", "2", "pipeline", "--manifest", "m.json", "--output", "out"]
        )
    assert audit.fingerprint is original and sys.argv is arguments
    assert _sha256(Path(io.__file__)) in caplog.text
    with pytest.raises(SystemExit):
        io.main(["--threads", "9", "pipeline"])


def test_threads_are_closed_before_returning(finished, monkeypatch):
    _, plan, _, _, _ = finished
    directory = single.folder(plan, 0)
    pools = []

    class TrackingPool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

    monkeypatch.setattr(io, "ThreadPoolExecutor", TrackingPool)
    io.fingerprint(
        plan, {"slot": 0, "summary_sha256": _sha256(directory / "summary.json")}, 2
    )
    assert pools and all(not t.is_alive() for pool in pools for t in pool._threads)
