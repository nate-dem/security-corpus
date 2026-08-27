from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from ingest.connectors.web.fineweb import (
    FINEWEB_LICENSE,
    FINEWEB_SOURCE_ID,
    DsirScorer,
    audit_fineweb_output,
    build_slurm_script,
    fit_dsir_scorer,
    normalize_fineweb_record,
    write_fineweb_records,
)
from scripts.ingest_fineweb import main as ingest_fineweb_main
from securityclip.paths import root_for_source, virtual_dir_for_row
from securityclip.web.operations import ALLOWED_ROOTS
from securityclip.web.routing import deterministic_route


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _tiny_scorer() -> DsirScorer:
    return fit_dsir_scorer(
        [
            "SQL injection exploit payload authentication bypass vulnerability",
            "malware command control privilege escalation exploit",
        ],
        [
            "sourdough bread recipe flour starter kitchen oven",
            "tennis match score forehand tournament player",
        ],
        min_feature_count=1,
    )


def test_dsir_scorer_ranks_security_text_above_generic_text():
    scorer = _tiny_scorer()

    security_score = scorer.score("remote code execution exploit and SQL injection vulnerability")
    generic_score = scorer.score("bread recipe with flour and kitchen oven notes")

    assert scorer.metadata["version"] == "dsir_log_ratio_v1"
    assert security_score > generic_score


def test_fineweb_normalization_fields():
    record = {
        "id": "doc-1",
        "title": "  Browser Sandbox Escape  ",
        "url": "https://example.test/sandbox",
        "text": "Browser sandbox escape analysis with exploit mitigation details.",
        "language": "en",
    }

    normalized = normalize_fineweb_record(record, score=1.25)

    assert normalized.source_id == FINEWEB_SOURCE_ID
    assert normalized.source_record_id == "doc-1"
    assert normalized.record_id == "fineweb-security:doc-1"
    assert normalized.title == "Browser Sandbox Escape"
    assert normalized.source_url == "https://example.test/sandbox"
    assert normalized.license == FINEWEB_LICENSE
    assert normalized.raw is None
    assert normalized.dsir_score == 1.25
    assert normalized.language == "en"
    assert normalized.content_length and normalized.content_length > 0
    assert normalized.content_hash


def test_local_ingest_writes_partitioned_parquet_and_filters_by_score(tmp_path: Path):
    input_path = tmp_path / "fineweb.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": "sec-1", "text": "SQL injection exploit vulnerability authentication bypass", "url": "https://a.test"},
            {"id": "gen-1", "text": "bread recipe flour starter kitchen oven", "url": "https://b.test"},
        ],
    )
    scorer = _tiny_scorer()
    scorer_path = tmp_path / "dsir_scorer.pkl"
    scorer.save(scorer_path)
    threshold = (scorer.score("SQL injection exploit vulnerability") + scorer.score("bread recipe flour oven")) / 2

    rc = ingest_fineweb_main(
        [
            "--scorer",
            str(scorer_path),
            "--input-glob",
            str(input_path),
            "--output-dir",
            str(tmp_path / "normalized"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--min-score",
            str(threshold),
            "--min-words",
            "1",
            "--overwrite",
        ]
    )

    assert rc == 0
    output_path = tmp_path / "normalized" / "source_id=fineweb-security" / "local_part_00000.parquet"
    table = pq.ParquetFile(output_path).read()
    rows = table.to_pylist()
    assert table.num_rows == 1
    assert rows[0]["source_id"] == "fineweb-security"
    assert rows[0]["source_record_id"] == "sec-1"
    assert rows[0]["dsir_score"] >= threshold


def test_task_sharding_is_deterministic_and_non_overlapping(tmp_path: Path):
    input_path = tmp_path / "fineweb.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": f"doc-{idx}", "text": f"security exploit vulnerability payload {idx}"}
            for idx in range(6)
        ],
    )
    scorer_path = tmp_path / "dsir_scorer.pkl"
    _tiny_scorer().save(scorer_path)
    output_dir = tmp_path / "normalized"
    report_dir = tmp_path / "reports"

    for task_id in (0, 1):
        assert ingest_fineweb_main(
            [
                "--scorer",
                str(scorer_path),
                "--input-glob",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--report-dir",
                str(report_dir),
                "--task-id",
                str(task_id),
                "--tasks",
                "2",
                "--min-score",
                "-999",
                "--min-words",
                "1",
                "--overwrite",
            ]
        ) == 0

    task0 = pq.ParquetFile(output_dir / "source_id=fineweb-security" / "task_00000_part_00000.parquet").read().to_pylist()
    task1 = pq.ParquetFile(output_dir / "source_id=fineweb-security" / "task_00001_part_00000.parquet").read().to_pylist()
    ids0 = {row["source_record_id"] for row in task0}
    ids1 = {row["source_record_id"] for row in task1}

    assert ids0 == {"doc-0", "doc-2", "doc-4"}
    assert ids1 == {"doc-1", "doc-3", "doc-5"}
    assert ids0.isdisjoint(ids1)
    state = json.loads((report_dir / "task_00000_state.json").read_text(encoding="utf-8"))
    assert state["completed"] is True
    assert state["next_chunk"] == 1


def test_audit_detects_duplicate_hashes_and_writes_artifacts(tmp_path: Path):
    output_dir = tmp_path / "normalized"
    report_dir = tmp_path / "reports"
    content = "shared exploit analysis content"
    records = [
        normalize_fineweb_record({"id": "a", "text": content}, score=2.0),
        normalize_fineweb_record({"id": "b", "text": content}, score=0.5),
    ]
    assert write_fineweb_records(records, output_dir, shard_name="part-00000", overwrite=True) == 2

    summary = audit_fineweb_output(output_dir, report_dir)

    assert summary["rows"] == 2
    assert summary["duplicate_hashes"] == 1
    assert (report_dir / "summary.md").exists()
    assert (report_dir / "summary.json").exists()
    audit_text = (report_dir / "audit_sample.csv").read_text(encoding="utf-8")
    assert "high_score" in audit_text
    assert "borderline_kept" in audit_text


def test_slurm_script_uses_marlowe_preempt_defaults():
    script = build_slurm_script(
        tasks=64,
        array_concurrency=8,
        account="marlowe-m000091",
        partition="preempt",
        qos="normal",
        cpus_per_task=4,
        mem="32G",
        time_limit="04:00:00",
        command="python3 scripts/ingest_fineweb.py --scorer data/fineweb/dsir_scorer.pkl",
    )

    assert "#SBATCH --account=marlowe-m000091" in script
    assert "#SBATCH --partition=preempt" in script
    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --array=0-63%8" in script
    assert "logs/fineweb/task_%a.out" in script


def test_security_scope_web_root_mapping_and_routing():
    root, segment = root_for_source("fineweb-security")
    row = {"source_id": "fineweb-security", "source_record_id": "abc123"}

    assert root == "/web"
    assert segment == "fineweb-security"
    assert virtual_dir_for_row(row) == "/web/fineweb-security/abc123"
    assert "/web" in ALLOWED_ROOTS

    route = deterministic_route("Find FineWeb documents about malware")
    assert route is not None
    assert route.likely_roots == ["/web"]


def test_frontend_root_label_includes_web():
    text = Path("web/src/lib/types.ts").read_text(encoding="utf-8")

    assert '"/web"' in text
    assert '"/web": "Web"' in text
