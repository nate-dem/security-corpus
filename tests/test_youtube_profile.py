"""Offline Parquet tests for the resumable, unfiltered YouTube inventory."""

import json
import os

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.youtube import download, profile


def make_download(root, tables):
    files = []
    verified = []
    for index, table in enumerate(tables):
        name = f"nested/shard_{index}.parquet"
        path = root / "raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=2)
        digest = profile._sha256(path)
        size = path.stat().st_size
        files.append({"path": name, "size": size, "hash_algorithm": "sha256", "hash": digest})
        verified.append({"path": name, "bytes": size, "sha256": digest})
    manifest = {"format_version": 1, "repo_id": download.REPO_ID,
                "revision": download.DEFAULT_REVISION, "files": files,
                "total_bytes": sum(row["size"] for row in files)}
    download._write_json(root / "manifest.json", manifest)
    download._write_json(root / "download-summary.json", {
        "complete": True, "failed": [], "verified": verified,
        "revision": download.DEFAULT_REVISION,
        "manifest_sha256": profile._sha256(root / "manifest.json"),
    })
    return manifest


def table():
    return pa.table({
        "video_id": ["v1", "v1", "v1", "v2", "", None],
        "channel_id": ["c1", "c1", "c1", "c2", "", None],
        "transcription_language": ["en", "fr", "en", "es", None, "en"],
        "original_language": ["en"] * 6,
        "license": ["CC BY"] * 4 + [None, ""],
        "word_count": ["10", "20", "30", "-1", "bad", "1.5"],
        "character_count": [50, 100, 150, 5, 0, None],
        "text": ["complete text " * 10000, "bonjour", "duplicate key, different text",
                 "hola", "", None],
    })


def run(root, *extra):
    return profile.main(["--data-dir", str(root), "--workers", "2", *extra])


def report(root):
    return json.loads((root / "profile-v1/summary.json").read_text())


def test_all_rows_and_languages_retained_with_exact_full_text_provenance(tmp_path):
    original = table()
    sparse = pa.table({"text": ["another shard"], "video_id": ["v3"]})
    empty = pa.table({"text": pa.array([], type=pa.string())})
    make_download(tmp_path, [original, sparse, empty])
    before = {p: profile._sha256(p) for p in (tmp_path / "raw").rglob("*.parquet")}
    assert run(tmp_path, "--samples-per-shard", "10") == 0
    summary = report(tmp_path)
    assert summary["complete"] is True
    assert summary["shards"] == 3
    assert summary["totals"] == {
        "records": 7, "distinct_video_ids": 3, "distinct_channel_ids": 2,
        "missing_video_ids": 2, "missing_channel_ids": 3, "missing_licenses": 3,
        "missing_or_unparseable_word_counts": 2, "invalid_word_counts": 3,
        "invalid_character_counts": 0, "reported_word_count_sum": 60,
        "reported_character_count_sum": 305,
    }
    assert summary["repeated_video_language_keys"] == {"groups": 1, "additional_rows": 1}
    languages = {row["transcription_language"]: row for row in summary["transcription_languages"]}
    assert languages["en"]["records"] == 3
    assert languages["en"]["reported_word_count_p50_p90_p99"] == [20.0, 28.0, 29.8]
    assert set(languages) == {"en", "fr", "es", None}
    samples = [json.loads(line) for line in (tmp_path / "profile-v1/inspection_sample.jsonl").read_text().splitlines()]
    assert len(samples) == 7
    for row in samples:
        source = pq.read_table(tmp_path / "raw" / row["source_shard"])
        assert row["text"] == source["text"][row["source_row"]].as_py()
        assert row["selection_probability"] == 1.0
        assert row["dataset_revision"] == download.DEFAULT_REVISION
    assert any(len(row["text"] or "") == 140000 for row in samples)
    assert before == {p: profile._sha256(p) for p in before}
    with duckdb.connect() as con:
        assert con.read_parquet(str(tmp_path / "profile-v1/metadata/*.parquet")).count("*").fetchone()[0] == 7


def test_sampling_is_deterministic_and_resume_reuses_valid_checkpoints(tmp_path, monkeypatch):
    make_download(tmp_path, [table(), table()])
    assert run(tmp_path) == 0
    output = tmp_path / "profile-v1"
    sample = (output / "inspection_sample.jsonl").read_bytes()
    rows = [json.loads(line) for line in sample.splitlines()]
    assert len(rows) == 2
    assert all(row["selection_probability"] == 1 / 6 for row in rows)
    tracked = list((output / "metadata").iterdir()) + list((output / "checkpoints").iterdir())
    times = {path: path.stat().st_mtime_ns for path in tracked}
    monkeypatch.setattr(profile, "_sample_texts", lambda *_: pytest.fail("Should reuse samples"))
    assert run(tmp_path) == 0
    assert sample == (output / "inspection_sample.jsonl").read_bytes()
    assert times == {path: path.stat().st_mtime_ns for path in tracked}


@pytest.mark.parametrize("artifact", ["metadata/00000.parquet", "samples/00000.jsonl"])
def test_corrupt_checkpoint_output_is_rebuilt_deterministically(tmp_path, artifact):
    make_download(tmp_path, [table()])
    assert run(tmp_path) == 0
    sample = (tmp_path / "profile-v1/inspection_sample.jsonl").read_bytes()
    (tmp_path / "profile-v1" / artifact).write_bytes(b"corrupt")
    assert run(tmp_path) == 0
    assert sample == (tmp_path / "profile-v1/inspection_sample.jsonl").read_bytes()
    assert report(tmp_path)["complete"] is True


@pytest.mark.parametrize("change", ["incomplete", "missing_evidence", "checksum", "missing_file"])
def test_rejects_download_without_complete_verification(tmp_path, change):
    manifest = make_download(tmp_path, [table()])
    path = tmp_path / "download-summary.json"
    summary = json.loads(path.read_text())
    if change == "incomplete":
        summary["complete"] = False
    elif change == "missing_evidence":
        summary["verified"] = []
    elif change == "checksum":
        summary["verified"][0]["sha256"] = "0" * 64
    else:
        (tmp_path / "raw" / manifest["files"][0]["path"]).unlink()
    download._write_json(path, summary)
    assert run(tmp_path) == 1
    assert not (tmp_path / "profile-v1").exists()


def test_source_mutation_on_resume_is_rejected(tmp_path):
    manifest = make_download(tmp_path, [table()])
    assert run(tmp_path) == 0
    path = tmp_path / "raw" / manifest["files"][0]["path"]
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert run(tmp_path) == 1
    assert report(tmp_path)["complete"] is False


def test_rejects_changed_sampling_config_without_overwriting_results(tmp_path):
    make_download(tmp_path, [table()])
    assert run(tmp_path) == 0
    old_report = report(tmp_path)
    assert run(tmp_path, "--seed", "1") == 1
    assert report(tmp_path) == old_report


def test_bad_shard_is_failure_and_never_silently_skipped(tmp_path):
    make_download(tmp_path, [table(), pa.table({"missing_text": [1]})])
    assert run(tmp_path) == 1
    assert report(tmp_path)["complete"] is False


def test_row_group_and_batch_offsets_preserve_selected_text(tmp_path):
    path = tmp_path / "groups.parquet"
    pq.write_table(pa.table({"text": [f"transcript {i}" for i in range(80)]}), path, row_group_size=30)
    assert profile._sample_texts(path, [0, 17, 30, 58, 79]) == {
        i: f"transcript {i}" for i in [0, 17, 30, 58, 79]
    }
