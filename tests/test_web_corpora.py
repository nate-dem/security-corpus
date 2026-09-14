"""Offline tests for gated downloads, full-text accounting, and safe resumes."""

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.web_corpora import compare, download, profile
from scripts.web_corpora.sources import SOURCES


def file_entry(path, body):
    return {"path": path, "size": len(body), "hash_algorithm": "sha256",
            "hash": hashlib.sha256(body).hexdigest()}


def fixture_source(root, source, rows):
    directory = root / source
    raw = directory / "raw"
    raw.mkdir(parents=True)
    name = "nested/data" + SOURCES[source]["suffix"]
    path = raw / name
    path.parent.mkdir()
    if name.endswith(".parquet"):
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                # Emulate Primus's observed upstream schema; its text is content.
                if source == "primus-fineweb" and "text" in row:
                    row = {("content" if k == "text" else k): v for k, v in row.items()}
                handle.write(json.dumps(row) + "\n")
    entry = file_entry(name, path.read_bytes())
    manifest = {"format_version": 1, "source": source, "repo_id": SOURCES[source]["repo_id"],
                "revision": SOURCES[source]["revision"], "files": [entry], "total_bytes": entry["size"]}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "download-summary.json").write_text(json.dumps({
        "complete": True, "failed": [], "repo_id": manifest["repo_id"], "revision": manifest["revision"],
        "manifest_sha256": profile.sha256(directory / "manifest.json"),
        "verified": [{"path": name, "bytes": entry["size"], "sha256": entry["hash"]}],
    }))
    return directory, manifest


@pytest.fixture
def tiny_manifests(monkeypatch):
    # File verification, summary coverage, reads, tokens, checkpoints, and SQL
    # remain real. Only substitute a small offline inventory for the Hub corpus.
    monkeypatch.setattr(profile, "load_manifest", lambda root, source, **kw:
                        json.loads((root / "manifest.json").read_text()))


@pytest.mark.parametrize("source,count", [("redsage-cfw", 125), ("primus-fineweb", 1599)])
def test_bundled_snapshots_are_pinned_and_resumed_without_network(tmp_path, source, count):
    manifest = download.load_manifest(tmp_path, source)
    assert len(download.data_files(manifest)) == count
    assert manifest["revision"] == SOURCES[source]["revision"]
    assert download.load_manifest(tmp_path, source, offline=True) == manifest
    changed = deepcopy(manifest)
    changed["files"][0]["hash"] = "0" * len(changed["files"][0]["hash"])
    (tmp_path / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="inventory changed"):
        download.load_manifest(tmp_path, source)


def test_download_authenticates_and_reuses_verified_files(tmp_path, monkeypatch):
    source = "primus-fineweb"
    root, manifest = fixture_source(tmp_path, source, [{"text": "networking"}])
    monkeypatch.setattr(download, "load_manifest", lambda *a, **kw: manifest)
    checked = []
    monkeypatch.setattr(download, "check_access", lambda value: checked.append(value["repo_id"]))
    monkeypatch.setattr(download, "authenticated_download", lambda **kw: pytest.fail("already downloaded"))
    assert download.main(["--source", source, "--data-dir", str(tmp_path)]) == 0
    assert checked == [SOURCES[source]["repo_id"]]
    assert json.loads((root / "download-summary.json").read_text())["complete"]
    checked.clear()
    assert download.main(["--source", source, "--data-dir", str(tmp_path), "--verify-only"]) == 0
    assert checked == []


def test_auth_failure_is_preflighted_and_credentials_not_logged(tmp_path, monkeypatch, caplog):
    class GatedError(Exception):
        pass
    GatedError.__module__ = "huggingface_hub.errors"
    def fail(_):
        raise GatedError("https://signed.example/?secret=DO_NOT_LOG")
    monkeypatch.setattr(download, "check_access", fail)
    monkeypatch.setattr(download, "authenticated_download", lambda **kw: pytest.fail("access denied"))
    assert download.main(["--source", "redsage-cfw", "--data-dir", str(tmp_path), "--check-access"]) == 1
    assert "DO_NOT_LOG" not in caplog.text
    assert "Accept access conditions" in caplog.text
    assert not (tmp_path / "redsage-cfw/raw").exists()


def test_mixed_formats_count_all_rows_without_adopting_upstream_labels(tmp_path, tiny_manifests):
    rows = [
        {"id": "1", "text": "TCP retransmits lost segments.",
         "metadata": {"url": "https://example.org/tcp", "relevant": True, "token_count": 999}},
        {"id": "2", "text": "TCP retransmits lost segments.",
         "metadata": {"url": "https://example.org/copy", "relevant": False, "token_count": 999}},
        {"id": "3", "text": "   ", "metadata": None},
        {"id": "4", "text": None, "metadata": None},
        {"id": "5", "text": "A second complete explanation.", "metadata": None},
    ]
    for source in SOURCES:
        fixture_source(tmp_path, source, rows)
    assert profile.main(["--data-dir", str(tmp_path), "--workers", "1", "--samples-per-shard", "2"]) == 0
    for source in SOURCES:
        output = tmp_path / source / "profile-v1"
        report = json.loads((output / "summary.json").read_text())
        assert report["complete"]
        assert report["totals"]["records"] == 5
        assert report["totals"]["valid_text_records"] == 3
        assert report["totals"]["raw_tokens"] == sum(compute_token_count(r["text"]) for r in rows if r["text"] is not None)
        assert report["exact_deduplication"]["extra_exact_copies"] == 1
        assert report["exact_deduplication"]["exact_unique_tokens"] == sum(
            compute_token_count(t) for t in {r["text"] for r in rows if r["text"] and r["text"].strip()})
        features = pq.read_table(output / "metadata/00000.parquet").to_pylist()
        assert features[0]["upstream_token_count"] == "999"
        assert features[0]["content_length"] != 999
        assert features[1]["upstream_relevant"] == "false"
        assert features[0]["upstream_license"] is None
        sample = [json.loads(line) for line in (output / "inspection_sample.jsonl").read_text().splitlines()]
        assert len(sample) == 2
        assert all(s["selection_probability"] == 0.4 for s in sample)
        for s in sample:
            assert s["raw_record"][profile.text_field(source)] == rows[s["source_row"]]["text"]


def test_checkpoint_resume_repairs_outputs_but_rejects_raw_corruption(tmp_path, tiny_manifests, monkeypatch):
    root, manifest = fixture_source(tmp_path, "redsage-cfw", [{"text": "System calls cross a privilege boundary."}])
    args = ["--source", "redsage-cfw", "--data-dir", str(tmp_path), "--workers", "1"]
    assert profile.main(args) == 0
    sidecar = root / "profile-v1/metadata/00000.parquet"
    original = sidecar.read_bytes()
    tokenize = profile.compute_token_count
    monkeypatch.setattr(profile, "compute_token_count", lambda t: pytest.fail("completed shard must be reused"))
    assert profile.main(args) == 0
    monkeypatch.setattr(profile, "compute_token_count", tokenize)
    sidecar.write_bytes(b"corrupt")
    assert profile.main(args) == 0
    assert sidecar.read_bytes() == original
    assert profile.main(args + ["--seed", "7"]) == 1
    raw = root / "raw" / manifest["files"][0]["path"]
    raw.write_bytes(b"X" * raw.stat().st_size)
    assert profile.main(args) == 1
    assert json.loads((root / "profile-v1/summary.json").read_text())["complete"] is False


def test_profile_requires_full_download_coverage(tmp_path, tiny_manifests):
    root, _ = fixture_source(tmp_path, "primus-fineweb", [{"text": "Example"}])
    report = json.loads((root / "download-summary.json").read_text())
    report["verified"] = []
    (root / "download-summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exactly once"):
        profile.load_inputs(root, "primus-fineweb")


def test_unknown_text_schema_and_invalid_json_fail_explicitly(tmp_path, tiny_manifests):
    fixture_source(tmp_path, "primus-fineweb", [{"unexpected_body": "must not silently count zero"}])
    assert profile.main(["--data-dir", str(tmp_path), "--source", "primus-fineweb", "--workers", "1"]) == 1
    invalid = tmp_path / "invalid.jsonl.gz"
    with gzip.open(invalid, "wt") as handle:
        handle.write('{"text":"valid"}\nnot json\n')
    with pytest.raises(ValueError, match="Invalid JSON"):
        list(profile.iter_rows(invalid))


def test_cross_source_overlap_and_baseline_hash_recomputed(tmp_path, tiny_manifests):
    shared, novel = "Already in the corpus.", "New security mechanism."
    for source in SOURCES:
        fixture_source(tmp_path, source, [{"text": shared}, {"text": novel}])
    assert profile.main(["--data-dir", str(tmp_path), "--workers", "1"]) == 0
    baseline = tmp_path / "baseline.parquet"
    pq.write_table(pa.Table.from_pylist([{"content": shared, "content_hash": "wrong stored hash",
                                        "content_length": compute_token_count(shared)}]), baseline)
    args = ["--data-dir", str(tmp_path), "--baseline", str(baseline)]
    assert compare.main(args) == 0
    report = json.loads((tmp_path / "comparison-v1.json").read_text())
    assert report["exact_unique_texts"] == 2
    assert report["cross_source_exact_texts"] == 2
    assert report["extra_exact_copies"] == 2
    assert report["baseline_comparison"]["matching_tokens"] == compute_token_count(shared)
    assert report["baseline_comparison"]["novel_exact_unique_tokens"] == compute_token_count(novel)
    assert report["baseline_comparison"]["baseline_tokens_recomputed"] is False
    assert report["baseline_comparison"]["target_is_release_gate"] is False
    assert report["baseline_comparison"]["aspirational_target_tokens"] == 3_000_000_000
    assert "release_minimum_tokens" not in report["baseline_comparison"]
    metadata = tmp_path / "redsage-cfw/profile-v1/metadata/00000.parquet"
    metadata.write_bytes(b"corrupted metadata")
    with pytest.raises(ValueError, match="changed or reordered"):
        compare.main(args)


def test_features_preserve_nested_metadata_and_literal_special_tokens():
    text = "Explain <|endoftext|> as literal data."
    result = profile.features({"text": text, "metadata": json.dumps({"url": "https://EXAMPLE.org/a", "license": "CC-BY-4.0"})},
                              "redsage-cfw", "a.parquet", 17)
    assert result["domain"] == "example.org"
    assert result["upstream_license"] == "CC-BY-4.0"
    assert result["content_hash"] == compute_content_hash(text)
    assert result["content_length"] == compute_token_count(text)


def test_multiprocess_profile_handles_empty_compressed_shards(tmp_path, tiny_manifests):
    fixture_source(tmp_path, "primus-fineweb", [])
    assert profile.main(["--data-dir", str(tmp_path), "--source", "primus-fineweb", "--workers", "2"]) == 0
    report = json.loads((tmp_path / "primus-fineweb/profile-v1/summary.json").read_text())
    assert report["totals"]["records"] == 0
    assert report["exact_deduplication"]["exact_unique_tokens"] == 0
    assert report["inspection_sample"]["records"] == 0


def test_slurm_wrapper_uses_checkout_and_preserves_arguments(tmp_path, monkeypatch):
    runner = tmp_path / "scripts/web_corpora/run.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    job = Path(__file__).parents[1] / "scripts/web_corpora/job.sbatch"
    result = subprocess.run(["bash", str(job), "profile", "--source", "primus-fineweb"],
                            check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ["profile", "--source", "primus-fineweb"]
    result = subprocess.run(["bash", str(job)], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ["full"]


def test_primus_content_mapping_and_mixed_profile_versions_preserve_redsage(tmp_path, tiny_manifests, monkeypatch):
    text = "Search poisoning directs users to malicious sites."
    fixture_source(tmp_path, "redsage-cfw", [{"text": text}])
    assert profile.main(["--data-dir", str(tmp_path), "--source", "redsage-cfw", "--workers", "1"]) == 0
    redsage_report = tmp_path / "redsage-cfw/profile-v1/summary.json"
    original = redsage_report.read_bytes()
    fixture_source(tmp_path, "primus-fineweb", [
        {"source": "FineWeb-Cybersecurity-Filtered", "url": "http://example.org/article",
         "content": text, "time": "2013-12-31"}, {"content": ""}, {"content": None}, {"content": 12}])
    read = profile.iter_rows
    def primus_only(path):
        assert path.name.endswith('.jsonl.gz'), "Completed RedSage data must not be scanned again"
        yield from read(path)
    monkeypatch.setattr(profile, "iter_rows", primus_only)
    assert profile.main(["--data-dir", str(tmp_path), "--source", "primus-fineweb",
                         "--profile-name", "profile-v2", "--workers", "1"]) == 0
    assert redsage_report.read_bytes() == original
    report = json.loads((tmp_path / "primus-fineweb/profile-v2/summary.json").read_text())
    assert report["text_field"] == "content"
    assert report["totals"]["records"] == 4
    assert report["totals"]["valid_text_records"] == 1
    assert report["totals"]["raw_tokens"] == compute_token_count(text)
    metadata = pq.read_table(tmp_path / "primus-fineweb/profile-v2/metadata/00000.parquet").to_pylist()
    assert metadata[0]["content_hash"] == compute_content_hash(text)
    assert metadata[0]["url"] == "http://example.org/article"
    assert metadata[0]["upstream_license"] is None
    assert [r["text_state"] for r in metadata] == ["valid", "blank", "missing", "non_string"]
    args = ["--data-dir", str(tmp_path), "--source-profile", "primus-fineweb=profile-v2"]
    assert compare.main(args) == 0
    comparison = json.loads((tmp_path / "comparison-v1.json").read_text())
    assert comparison["cross_source_exact_texts"] == 1
    assert comparison["exact_unique_tokens"] == compute_token_count(text)
    assert comparison["profile_names"] == {"redsage-cfw": "profile-v1", "primus-fineweb": "profile-v2"}
    assert comparison["profile_reports"]["redsage-cfw"] == hashlib.sha256(original).hexdigest()
    for invalid in ["unknown=profile-v2", "primus-fineweb=../raw", "primus-fineweb=raw"]:
        with pytest.raises(SystemExit):
            compare.main(["--data-dir", str(tmp_path), "--source-profile", invalid])
