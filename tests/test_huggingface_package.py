import hashlib
from types import SimpleNamespace

import pytest

from scripts.release.upload_huggingface_private import upload_private, verify_package


def sample_package(root):
    (root / "data/000").mkdir(parents=True)
    path = root / "data/000/000000.parquet"
    path.write_bytes(b"test payload")
    (root / "README.md").write_text("# CyberOpen\n")
    (root / ".gitattributes").write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n")
    (root / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n"
        for p in sorted(root.rglob("*")) if p.is_file()))
    return path


def test_private_upload_refuses_existing_public_repository(tmp_path):
    sample_package(tmp_path)
    calls = []
    api = SimpleNamespace(
        create_repo=lambda **kw: calls.append(kw),
        dataset_info=lambda **kw: SimpleNamespace(private=False),
        upload_folder=lambda **kw: pytest.fail("Must not upload to a public repository"),
    )
    with pytest.raises(ValueError, match="Repository is public"):
        upload_private(tmp_path, "lab/corpus", api)
    assert calls[0]["private"] is True and calls[0]["repo_type"] == "dataset"


def test_modified_payload_is_rejected_before_remote_creation(tmp_path):
    path = sample_package(tmp_path)
    path.write_bytes(b"changed")
    api = SimpleNamespace(create_repo=lambda **kw: pytest.fail("No network on invalid package"))
    with pytest.raises(ValueError, match="checksum failed"):
        upload_private(tmp_path, "lab/corpus", api)


def test_extra_training_file_is_rejected_but_resume_cache_is_allowed(tmp_path):
    sample_package(tmp_path)
    (tmp_path / ".cache/huggingface").mkdir(parents=True)
    (tmp_path / ".cache/huggingface/resume").write_text("cache")
    verify_package(tmp_path)
    (tmp_path / "data/000/excluded.parquet").write_bytes(b"must not be uploaded")
    with pytest.raises(ValueError, match="unlisted"):
        verify_package(tmp_path)


def test_internal_artifact_rejected_even_if_checksum_is_valid(tmp_path):
    sample_package(tmp_path)
    artifact = tmp_path / "upload-manifest.json"
    artifact.write_text('{"source_root": "/Users/researcher/work"}')
    with (tmp_path / "SHA256SUMS").open("a") as handle:
        handle.write(f"{hashlib.sha256(artifact.read_bytes()).hexdigest()}  upload-manifest.json\n")
    with pytest.raises(ValueError, match="outside the CyberOpen release"):
        verify_package(tmp_path)


def test_existing_development_artifacts_on_hub_block_upload(tmp_path):
    sample_package(tmp_path)
    api = SimpleNamespace(
        create_repo=lambda **kw: None,
        dataset_info=lambda **kw: SimpleNamespace(private=True),
        list_repo_files=lambda **kw: ["README.md", "provenance/v10/notes.md"],
        upload_folder=lambda **kw: pytest.fail("Must not leave unwanted remote artifacts"),
    )
    with pytest.raises(ValueError, match="outside this release"):
        upload_private(tmp_path, "lab/CyberOpen", api)


def test_private_clean_repository_uploads_only_release_folder(tmp_path):
    sample_package(tmp_path)
    calls = []
    api = SimpleNamespace(
        create_repo=lambda **kw: None,
        dataset_info=lambda **kw: SimpleNamespace(private=True),
        list_repo_files=lambda **kw: [".gitattributes"],
        upload_folder=lambda **kw: calls.append(kw),
    )
    upload_private(tmp_path, "lab/CyberOpen", api)
    assert len(calls) == 1
    assert calls[0]["folder_path"] == str(tmp_path)
    assert calls[0]["commit_message"] == "Upload CyberOpen"


def test_union_features_preserve_source_fields_and_exclude_evidence(tmp_path):
    datasets = pytest.importorskip("datasets")
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    (tmp_path / "data/000").mkdir(parents=True)
    (tmp_path / "provenance/v10/evidence").mkdir(parents=True)
    pq.write_table(pa.table({"content": ["question"], "score": [5]}), tmp_path / "data/000/0.parquet")
    pq.write_table(pa.table({"content": ["rule"], "rule_source": ["YAML"]}), tmp_path / "data/000/1.parquet")
    pq.write_table(pa.table({"content": ["excluded"]}), tmp_path / "provenance/v10/evidence/excluded.parquet")
    features = datasets.Features.from_arrow_schema(pa.schema([
        ("content", pa.string()), ("score", pa.int64()), ("rule_source", pa.string())]))
    meta = {"configs": [{"config_name": "default", "data_files": [
        {"split": "train", "path": "data/*/*.parquet"}]}],
        "dataset_info": {"features": features._to_yaml_list()}}
    (tmp_path / "README.md").write_text("---\n" + yaml.safe_dump(meta) + "---\n# Corpus\n")
    data = datasets.load_dataset(str(tmp_path), split="train", streaming=True,
                                 cache_dir=str(tmp_path / "cache"))
    assert list(data) == [
        {"content": "question", "score": 5, "rule_source": None},
        {"content": "rule", "score": None, "rule_source": "YAML"},
    ]
