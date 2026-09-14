"""Offline failure/restart tests for the standalone shard downloader."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.youtube import download as downloader


REVISION = downloader.DEFAULT_REVISION


def entry(name="cctube_0.parquet", body=b"test shard"):
    return {"path": name, "size": len(body), "hash_algorithm": "sha256",
            "hash": hashlib.sha256(body).hexdigest()}


def manifest(files=None):
    files = files if files is not None else [entry()]
    return {"format_version": 1, "repo_id": downloader.REPO_ID,
            "revision": REVISION, "files": files,
            "total_bytes": sum(item["size"] for item in files)}


def save_manifest(root, files=None):
    root.mkdir(parents=True, exist_ok=True)
    result = manifest(files)
    (root / "manifest.json").write_text(json.dumps(result))
    return result


def test_inventory_enumerates_every_parquet_and_pins_revision(tmp_path, monkeypatch):
    calls = []

    def dataset_info(repo, **kwargs):
        calls.append((repo, kwargs))
        return SimpleNamespace(sha=REVISION, siblings=[
            SimpleNamespace(rfilename="README.md"),
            SimpleNamespace(rfilename="nested/train-00001.parquet", size=10,
                            lfs={"sha256": "a" * 64}),
            SimpleNamespace(rfilename="cctube_999.parquet", size=12,
                            lfs=None, blob_id="b" * 40),
        ])

    monkeypatch.setattr(downloader, "_hub_api", lambda: SimpleNamespace(dataset_info=dataset_info))
    result = downloader._load_or_create_manifest(tmp_path, None, offline=False)
    assert [row["path"] for row in result["files"]] == [
        "cctube_999.parquet", "nested/train-00001.parquet",
    ]
    assert result["total_bytes"] == 22
    assert calls[0][1]["revision"] == REVISION
    assert calls[0][1]["files_metadata"] is True
    assert json.loads((tmp_path / "manifest.json").read_text()) == result


def test_existing_manifest_is_reused_offline_and_cannot_change_revision(tmp_path, monkeypatch):
    expected = save_manifest(tmp_path)
    monkeypatch.setattr(downloader, "_hub_api", lambda: pytest.fail("No network on resume"))
    assert downloader._load_or_create_manifest(tmp_path, None, offline=True) == expected
    with pytest.raises(ValueError, match="revision changed"):
        downloader._load_or_create_manifest(tmp_path, "b" * 40, offline=False)


@pytest.mark.parametrize("name", ["../escape.parquet", "/escape.parquet", "a/../b.parquet",
                                 "a\\b.parquet", "a//b.parquet", ".hidden.parquet"])
def test_rejects_unsafe_shard_paths(name):
    with pytest.raises(ValueError):
        downloader._validate_manifest(manifest([entry(name)]))


@pytest.mark.parametrize("change", ["duplicate", "bad_hash", "wrong_total", "no_files", "wrong_repo"])
def test_rejects_incomplete_or_inconsistent_inventory(change):
    value = deepcopy(manifest())
    if change == "duplicate":
        value["files"] *= 2
    elif change == "bad_hash":
        value["files"][0]["hash"] = "wrong"
    elif change == "wrong_total":
        value["total_bytes"] += 1
    elif change == "no_files":
        value["files"] = []
    else:
        value["repo_id"] = "unrelated/repo"
    with pytest.raises(ValueError):
        downloader._validate_manifest(value)


def test_verify_checks_content_not_just_size_and_supports_git_blob_hash(tmp_path):
    path = tmp_path / "cctube_0.parquet"
    body = b"test shard"
    path.write_bytes(body)
    assert downloader._verify_file(path, entry())["sha256"] == hashlib.sha256(body).hexdigest()
    git_entry = {**entry(), "hash_algorithm": "git_blob_sha1",
                 "hash": hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()}
    assert downloader._verify_file(path, git_entry)["bytes"] == len(body)
    path.write_bytes(b"X" * len(body))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader._verify_file(path, entry())


def test_valid_file_is_rehashed_and_skipped_without_network(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/cctube_0.parquet").write_bytes(b"test shard")
    monkeypatch.setattr(downloader, "_hub_download", lambda **_: pytest.fail("Already complete"))
    result = downloader._download_one(entry(), manifest(), tmp_path, 2, False)
    assert result["action"] == "already_verified"


def test_corrupt_file_repaired_and_partial_transfer_preserved_on_retry(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "cctube_0.parquet").write_bytes(b"X" * len(b"test shard"))
    partial = raw / ".partial-test"
    calls = []

    def transfer(**kwargs):
        calls.append(kwargs)
        # Default Hub behavior resumes partial files; no force_download flag.
        assert "force_download" not in kwargs
        assert kwargs["revision"] == REVISION
        assert kwargs["local_dir"] == raw
        if len(calls) == 1:
            partial.write_bytes(b"partial")
            raise ConnectionError("interrupted")
        assert partial.read_bytes() == b"partial"
        destination = raw / kwargs["filename"]
        destination.write_bytes(b"test shard")
        return str(destination)

    monkeypatch.setattr(downloader, "_hub_download", transfer)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    assert downloader._download_one(entry(), manifest(), tmp_path, 2, False)["action"] == "downloaded"
    assert len(calls) == 2


def test_bad_download_is_not_accepted_and_can_be_retried(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()

    def transfer(**kwargs):
        path = kwargs["local_dir"] / kwargs["filename"]
        path.write_bytes(b"X" * len(b"test shard"))
        return str(path)

    monkeypatch.setattr(downloader, "_hub_download", transfer)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        downloader._download_one(entry(), manifest(), tmp_path, 1, False)
    assert not (tmp_path / "raw/cctube_0.parquet").exists()


def test_verify_only_is_offline_and_never_repairs(tmp_path, monkeypatch):
    save_manifest(tmp_path)
    (tmp_path / "raw").mkdir()
    path = tmp_path / "raw/cctube_0.parquet"
    path.write_bytes(b"bad")
    monkeypatch.setattr(downloader, "_hub_download", lambda **_: pytest.fail("Offline"))
    assert downloader.main(["--output-dir", str(tmp_path), "--verify-only"]) == 1
    report = json.loads((tmp_path / "download-summary.json").read_text())
    assert report["complete"] is False
    assert report["failed"][0]["path"] == "cctube_0.parquet"
    assert path.read_bytes() == b"bad"


def test_end_to_end_resume_reports_all_shards_and_keeps_unrelated_files(tmp_path, monkeypatch):
    files = [entry(), entry("nested/other.parquet", b"second")]
    save_manifest(tmp_path, files)
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("preserve")
    attempts = []

    def transfer(**kwargs):
        attempts.append(kwargs["filename"])
        path = kwargs["local_dir"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"second" if kwargs["filename"].startswith("nested") else b"test shard")
        return str(path)

    monkeypatch.setattr(downloader, "_hub_download", transfer)
    args = ["--output-dir", str(tmp_path)]
    assert downloader.main(args) == 0
    assert downloader.main(args) == 0
    report = json.loads((tmp_path / "download-summary.json").read_text())
    assert report["complete"] is True
    assert report["verified_shards"] == 2
    assert report["verified_bytes"] == 16
    assert len(attempts) == 2
    assert unrelated.read_text() == "preserve"


def test_failures_leave_run_incomplete_but_good_shards_survive(tmp_path, monkeypatch):
    save_manifest(tmp_path, [entry(), entry("missing.parquet")])
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/cctube_0.parquet").write_bytes(b"test shard")
    monkeypatch.setattr(downloader, "_hub_download", lambda **_: (_ for _ in ()).throw(ConnectionError()))
    assert downloader.main(["--output-dir", str(tmp_path), "--retries", "1"]) == 1
    report = json.loads((tmp_path / "download-summary.json").read_text())
    assert report["verified_shards"] == 1
    assert report["complete"] is False
    assert report["failed"] == [{"path": "missing.parquet", "error_type": "ConnectionError"}]


def test_second_process_cannot_use_same_output_directory(tmp_path):
    with downloader._directory_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another downloader"):
            with downloader._directory_lock(tmp_path):
                pytest.fail("second lock acquired")
    with downloader._directory_lock(tmp_path):
        pass  # releasing a lock must not require deleting the lock file


def test_symlink_cannot_redirect_a_shard(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"test shard")
    (raw / "cctube_0.parquet").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        downloader._shard_path(raw, entry())


def test_plan_only_never_downloads(tmp_path, monkeypatch):
    save_manifest(tmp_path)
    monkeypatch.setattr(downloader, "_hub_download", lambda **_: pytest.fail("plan only"))
    assert downloader.main(["--output-dir", str(tmp_path), "--plan-only"]) == 0
    assert not (tmp_path / "raw").exists()


def test_no_manifest_offline_fails(tmp_path):
    assert downloader.main(["--output-dir", str(tmp_path), "--verify-only"]) == 1


def test_interruption_leaves_incomplete_report(tmp_path, monkeypatch):
    save_manifest(tmp_path)
    monkeypatch.setattr(downloader, "_download_one", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert downloader.main(["--output-dir", str(tmp_path)]) == 130
    report = json.loads((tmp_path / "download-summary.json").read_text())
    assert report["status"] == "interrupted"
    assert report["complete"] is False


def test_low_disk_space_stops_before_transfer(tmp_path, monkeypatch):
    save_manifest(tmp_path)
    monkeypatch.setattr(downloader.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    monkeypatch.setattr(downloader, "_hub_download", lambda **_: pytest.fail("out of space"))
    assert downloader.main(["--output-dir", str(tmp_path)]) == 1

