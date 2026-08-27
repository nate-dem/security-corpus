from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.release.audit_qwen_human_labels import main


def _row(record_id: str, qwen: bool, manual: bool | None) -> dict:
    return {
        "source_id": "source",
        "record_id": record_id,
        "content_hash": f"hash-{record_id}",
        "qwen_should_keep": qwen,
        "manual_should_keep": manual,
        "manual_notes": "reviewed",
        "reviewer": "researcher",
        "reviewed_at": "2026-08-26T12:00:00-07:00",
    }


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_human_label_audit_reports_agreement_without_threshold(tmp_path):
    audit = tmp_path / "audit.parquet"
    report = tmp_path / "report.json"
    _write(audit, [_row("a", True, True), _row("b", False, True)])

    assert main(["--input", str(audit), "--output", str(report)]) == 0
    payload = __import__("json").loads(report.read_text(encoding="utf-8"))
    assert payload["agreement_rate"] == 0.5
    assert payload["agreement_threshold_applied"] is False
    assert payload["release_blocking_issues"] == {}


def test_human_label_audit_blocks_incomplete_review(tmp_path):
    audit = tmp_path / "audit.parquet"
    report = tmp_path / "report.json"
    _write(audit, [_row("a", True, None)])

    assert main(["--input", str(audit), "--output", str(report)]) == 2
