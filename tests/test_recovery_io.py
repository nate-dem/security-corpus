from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.arxiv import download_sources, harvest_citation_metadata
from scripts.arxiv.normalize_sources import _normalize_one
from scripts.release.audit_corpus_integrity import main as audit
from scripts.release.audit_source_licenses import _classify_license, _discover_parquet


def test_integrity_recomputes_instead_of_trusting_stored_counts(tmp_path):
    content = "Security audit text."
    row = {"source_id": "test", "source_record_id": "1", "record_id": "test:1",
           "content": content, "content_hash": compute_content_hash(content),
           "content_length": compute_token_count(content), "license": "test",
           "source_url": "https://example.org/1", "ingested_at": datetime.now(timezone.utc)}
    data, report = tmp_path / "data.parquet", tmp_path / "audit.json"
    pq.write_table(pa.Table.from_pylist([row]), data)
    assert audit([str(data), "--output", str(report)]) == 0
    assert json.loads(report.read_text())["integrity_passed"]
    row["content_length"] += 1
    row["content_hash"] = "0" * 64
    pq.write_table(pa.Table.from_pylist([row]), data)
    assert audit([str(data), "--output", str(report)]) == 2
    assert json.loads(report.read_text())["issues"] == {
        "content_length_mismatch": 1, "content_hash_mismatch": 1,
    }
    row["content_hash"] = compute_content_hash(content)
    pq.write_table(pa.Table.from_pylist([row]), data)
    assert audit([str(data), "--skip-token-check", "--output", str(report)]) == 0
    assert not json.loads(report.read_text())["integrity_passed"]


def test_license_audit_rejects_missing_requested_input_and_unknown_state(tmp_path):
    with pytest.raises(FileNotFoundError):
        _discover_parquet([tmp_path / "missing"])
    assert _classify_license({"state": "typo"}, "license")[0] == "unknown"


def test_metadata_harvest_recovers_failed_legacy_checkpoint(tmp_path):
    checkpoint = tmp_path / ".checkpoint"
    checkpoint.write_text("2401.00001\n2401.00002\n")
    called = []
    def get_record(**kwargs):
        called.append(kwargs["identifier"])
        if kwargs["identifier"].endswith("2"):
            raise OSError("transient network failure")
        return SimpleNamespace(raw="<record><header><identifier>oai:arXiv.org:2401.00001</identifier>"
                                   "</header><metadata><arXiv><id>2401.00001</id><title>Paper</title>"
                                   "</arXiv></metadata></record>")
    client = SimpleNamespace(GetRecord=get_record)
    assert harvest_citation_metadata.harvest_by_id(client, ["2401.00001", "2401.00002"],
                                                  tmp_path, checkpoint, 0) == 1
    assert len(called) == 2
    assert (tmp_path / "2401.jsonl").read_text().strip()
    called.clear()
    harvest_citation_metadata.harvest_by_id(client, ["2401.00001", "2401.00002"], tmp_path, checkpoint, 0)
    assert called == ["oai:arXiv.org:2401.00002"]


def test_download_is_atomic_and_retries_after_interruption(tmp_path, monkeypatch):
    payload = gzip.compress(b"\\documentclass{article}\n")
    class Response:
        status_code = 200
        headers = {}
        interrupted = True
        def raise_for_status(self):
            pass
        def iter_content(self, **kwargs):
            yield payload[:5]
            if self.interrupted:
                raise OSError("interrupted")
            yield payload[5:]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    response = Response()
    session = SimpleNamespace(get=lambda *a, **kw: response, close=lambda: None)
    monkeypatch.setattr(download_sources, "_make_session", lambda: session)
    assert download_sources.download_papers(["2401.00001"], tmp_path, 0) == 1
    assert not (tmp_path / "2401/2401.00001.tar.gz").exists()
    assert not list(tmp_path.rglob("*.tmp"))
    response.interrupted = False
    assert download_sources.download_papers(["2401.00001"], tmp_path, 0) == 0
    assert (tmp_path / "2401/2401.00001.tar.gz").read_bytes() == payload
    assert not (tmp_path / "2401/2401.00001.failed").exists()


def test_normalizer_resume_checks_archive_hash_and_content_file(tmp_path):
    archive = tmp_path / "2401/2401.00001.gz"
    archive.parent.mkdir()
    archive.write_bytes(gzip.compress(b"\\documentclass{article}\n\\begin{document}First\\end{document}"))
    output = tmp_path / "normalized"
    assert _normalize_one((archive, output))[1:] == (1, 0, 0)
    assert _normalize_one((archive, output))[1:] == (0, 1, 0)
    content = output / "2401/2401.00001/main.tex"
    content.unlink()
    assert _normalize_one((archive, output))[1:] == (1, 0, 0)
    archive.write_bytes(gzip.compress(b"\\documentclass{article}\n\\begin{document}Changed\\end{document}"))
    assert _normalize_one((archive, output))[1:] == (1, 0, 0)
    assert "Changed" in content.read_text()


@pytest.mark.parametrize("script,shard,count", [
    ("qwen_qa_array.sbatch", "42", "100"),
    ("qwen_citation_abstract_array.sbatch", "7", "24"),
])
def test_partial_slurm_retry_preserves_shard_modulus(tmp_path, script, shard, count):
    activate = tmp_path / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.touch()
    fake_python = tmp_path / ".venv/bin/python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    env = {**os.environ, "PROJECT_DIR": str(tmp_path), "SLURM_ARRAY_TASK_ID": shard,
           "SLURM_ARRAY_TASK_COUNT": "1", "PATH": f"{fake_python.parent}:{os.environ['PATH']}"}
    env.pop("NUM_SHARDS", None)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "scripts/classify/slurm" / script)],
                            env=env, capture_output=True, text=True, check=True)
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index("--num-shards") + 1] == count
    assert arguments[arguments.index("--shard-id") + 1] == shard
