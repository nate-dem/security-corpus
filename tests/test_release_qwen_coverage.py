from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.release.audit_qwen_coverage import main


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _corpus_row(record_id: str) -> dict:
    return {
        "source_id": "source",
        "record_id": record_id,
        "content_hash": record_id * 4,
        "content_length": 10,
    }


def _decision_row(record_id: str) -> dict:
    return {
        "source_id": "source",
        "record_id": record_id,
        "content_hash": record_id * 4,
        "qwen_should_keep": True,
        "qwen_parse_status": "ok",
        "qwen_model": "Qwen/Qwen3-8B",
        "qwen_model_revision": REVISION,
        "qwen_prompt_version": "test",
        "qwen_scored_at": "2026-08-26T00:00:00+00:00",
        "qwen_task": "qa",
    }


def test_qwen_coverage_passes_complete_decisions(tmp_path):
    corpus = tmp_path / "corpus.parquet"
    decisions = tmp_path / "decisions.parquet"
    _write(corpus, [_corpus_row("a")])
    _write(decisions, [_decision_row("a")])

    assert main([
        "--corpus", str(corpus),
        "--decisions", str(decisions),
        "--expected-revision", REVISION,
    ]) == 0


def test_qwen_coverage_blocks_missing_decision(tmp_path):
    corpus = tmp_path / "corpus.parquet"
    decisions = tmp_path / "decisions.parquet"
    _write(corpus, [_corpus_row("a"), _corpus_row("b")])
    _write(decisions, [_decision_row("a")])

    assert main([
        "--corpus", str(corpus),
        "--decisions", str(decisions),
    ]) == 2
