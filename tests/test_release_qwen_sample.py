from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.release.sample_qwen_audit import main


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _corpus_row(source_id: str, record_id: str) -> dict:
    return {
        "source_id": source_id,
        "record_id": record_id,
        "content_hash": f"hash-{record_id}",
        "content": f"content {record_id}",
        "content_length": 2,
    }


def _decision_row(source_id: str, record_id: str, keep: bool) -> dict:
    return {
        "source_id": source_id,
        "record_id": record_id,
        "content_hash": f"hash-{record_id}",
        "qwen_should_keep": keep,
        "qwen_parse_status": "ok",
        "qwen_model": "Qwen/Qwen3-8B",
        "qwen_model_revision": REVISION,
        "qwen_prompt_version": "test",
    }


def test_qwen_audit_sample_is_deterministic_and_stratified(tmp_path):
    corpus = tmp_path / "corpus.parquet"
    decisions = tmp_path / "decisions.parquet"
    output_a = tmp_path / "sample-a.parquet"
    output_b = tmp_path / "sample-b.parquet"
    corpus_rows = []
    decision_rows = []
    for source_id in ("one", "two"):
        for keep in (False, True):
            for index in range(3):
                record_id = f"{source_id}-{keep}-{index}"
                corpus_rows.append(_corpus_row(source_id, record_id))
                decision_rows.append(_decision_row(source_id, record_id, keep))
    _write(corpus, corpus_rows)
    _write(decisions, decision_rows)

    common = [
        "--corpus", str(corpus),
        "--decisions", str(decisions),
        "--per-stratum", "1",
        "--seed", "17",
    ]
    assert main([*common, "--output", str(output_a)]) == 0
    assert main([*common, "--output", str(output_b)]) == 0

    table_a = pq.read_table(output_a).sort_by("record_id")
    table_b = pq.read_table(output_b).sort_by("record_id")
    assert table_a.num_rows == 4
    assert table_a.column("record_id").to_pylist() == table_b.column("record_id").to_pylist()
    assert table_a.column("manual_should_keep").null_count == 4


def test_qwen_audit_sample_rejects_incomplete_coverage(tmp_path):
    corpus = tmp_path / "corpus.parquet"
    decisions = tmp_path / "decisions.parquet"
    _write(corpus, [_corpus_row("one", "a"), _corpus_row("one", "b")])
    _write(decisions, [_decision_row("one", "a", True)])

    try:
        main([
            "--corpus", str(corpus),
            "--decisions", str(decisions),
            "--output", str(tmp_path / "sample.parquet"),
            "--per-stratum", "1",
        ])
    except ValueError as error:
        assert "missing decisions=1" in str(error)
    else:
        raise AssertionError("Incomplete coverage should have failed")
