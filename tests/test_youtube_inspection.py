import hashlib
import json

import pytest

from scripts.youtube.inspect_profile import analyze, read_verified_sample


def save_profile(root, *, complete=True, rows=None):
    if rows is None:
        rows = [{"source_shard": "a.parquet", "source_row": 0, "shard_rows": 1,
                 "dataset_revision": "abc", "selection_probability": 1, "text": "hello"}]
    body = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    (root / "inspection_sample.jsonl").write_bytes(body)
    (root / "summary.json").write_text(json.dumps({
        "complete": complete, "dataset_revision": "abc", "shard_schemas": [{"shard": "a.parquet"}],
        "inspection_sample": {"sha256": hashlib.sha256(body).hexdigest(), "records": len(rows)},
    }))


def test_copied_sample_checksum_is_verified_before_analysis(tmp_path):
    save_profile(tmp_path)
    assert read_verified_sample(tmp_path)[1][0]["text"] == "hello"
    (tmp_path / "inspection_sample.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        read_verified_sample(tmp_path)


def test_incomplete_profile_rejected(tmp_path):
    save_profile(tmp_path, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        read_verified_sample(tmp_path)


@pytest.mark.parametrize("change", ["duplicate", "revision", "bounds", "probability", "unknown_shard"])
def test_invalid_sample_provenance_rejected(tmp_path, change):
    row = {"source_shard": "a.parquet", "source_row": 0, "shard_rows": 1,
           "dataset_revision": "abc", "selection_probability": 1, "text": "hello"}
    rows = [row]
    if change == "duplicate":
        rows.append(row.copy())
    elif change == "revision":
        row["dataset_revision"] = "other"
    elif change == "bounds":
        row["source_row"] = 1
    elif change == "probability":
        row["selection_probability"] = 0
    else:
        row["source_shard"] = "unknown.parquet"
    save_profile(tmp_path, rows=rows)
    with pytest.raises(ValueError):
        read_verified_sample(tmp_path)


def test_sample_counts_distinguish_blanks_duplicates_and_languages():
    rows = [{"source_shard": "a.parquet", "source_row": i, "video_id": str(i),
             "transcription_language": language, "original_language": "en",
             "text": text, "character_count": len(text), "selection_probability": 0.1}
            for i, (language, text) in enumerate([
                ("en", "hello"), ("en", "hello"), ("fr", ""), ("fr", ""),
            ])]
    stats, features = analyze(rows)
    assert stats["records"] == 4
    assert stats["blank_text_records"] == 2
    assert stats["nonempty_exact_text_duplicate_extra_rows"] == 1
    assert stats["cl100k_base_tokens"] == 2
    assert stats["reported_character_count_mismatches"] == 0
    assert features[0]["content_sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert stats["by_language"][0]["cl100k_base_tokens"] == 2
    assert stats["by_language"][1]["blank_text_records"] == 2
