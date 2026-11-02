from datetime import datetime, timezone
import json

from ingest.utils import compute_content_hash, compute_token_count
from scripts.release.inspect_final_corpus import cloudtrail_errors, inspect_row, run, save, sha

POLICY = {"max_repeated_line_fraction": .5, "max_bad_character_fraction": .05}


def record(text, **extra):
    return dict(source_id="test", source_record_id="1", record_id="test:1", content=text,
                content_hash=compute_content_hash(text), content_length=compute_token_count(text),
                license="CC-BY-4.0", ingested_at=datetime.now(timezone.utc), **extra)


def test_ai_quote_is_diagnostic_not_an_authorship_or_exclusion_decision():
    row = record('This security paper quotes the phrase "as an AI language model" in an experiment.')
    result = inspect_row(row, POLICY)
    assert "ai_self_identification" in result["flags"]
    assert result["errors"] == []
    assert "should_keep" not in result


def test_invalid_numerical_metadata_and_text_drift_are_detected():
    row = record("An example CVE description.", cvss_score=11, cve_id="CVE-2020-1234")
    row.update(source_id="nvd", source_record_id="CVE-2020-1234", record_id="nvd:CVE-2020-1234", content_length=100)
    result = inspect_row(row, POLICY)
    assert set(result["errors"]) == {"cvss_out_of_range", "content_length_mismatch"}


def test_optional_nulls_are_not_invalid_values_but_naive_time_is_flagged():
    row = record("Valid content.", title=None, published_at=datetime(2000, 1, 1), raw=None)
    result = inspect_row(row, POLICY)
    assert result["errors"] == []
    assert "timezone_missing_published_at" in result["flags"]


def test_cloudtrail_actual_events_are_compared_to_summary_fields():
    event = dict(eventTime="2020-01-01T00:00:00Z", eventName="ListBuckets", eventSource="s3.amazonaws.com", awsRegion="us-east-1", sourceIPAddress="example", userIdentity={"type":"Root"})
    content = "# Session\n## Events\n" + json.dumps(event)
    row = dict(event_count=1, actions=["ListBuckets"], aws_services=["s3.amazonaws.com"], regions=["us-east-1"], principals=["Root"], has_errors=False, source_ip="example", session_duration_seconds=0, published_at=datetime(2020,1,1,tzinfo=timezone.utc))
    assert cloudtrail_errors(row, content) == []
    row["event_count"] = 2
    assert cloudtrail_errors(row, content) == ["cloudtrail_event_count_mismatch"]


def test_raw_json_and_empty_content_are_checked():
    row = record("[deleted]", raw="{broken")
    result = inspect_row(row, POLICY)
    assert set(result["errors"]) == {"raw_invalid_json", "content_is_placeholder"}


def test_complete_scan_binds_manifest_and_finds_cross_row_duplicates(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "corpus"
    (root / "records").mkdir(parents=True)
    row = record("Identical content.", title=None, published_at=None, source_url=None, raw=None)
    second = dict(row, source_record_id="2", record_id="test:2")
    path = root / "records/000000.parquet"
    pq.write_table(pa.Table.from_pylist([row, second]), path)
    totals = dict(records=2, tokens=row["content_length"] * 2, bytes=path.stat().st_size)
    save(root / "corpus-manifest.json", dict(files=[dict(relative_path="records/000000.parquet", sha256=sha(path), **totals)], totals=totals))
    save(root / "final-corpus-complete.json", dict(manifest_sha256=sha(root / "corpus-manifest.json")))
    output = tmp_path / "report"
    run(root, output, workers=1)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["totals"]["records"] == 2
    assert summary["duplicate_groups"] == {"record_id_groups": 0, "content_hash_groups": 1}
    assert summary["structural_error_records"] == 0
    assert pq.read_table(output / "record-diagnostics.parquet").num_rows == 2
