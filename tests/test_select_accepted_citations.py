from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.arxiv.select_accepted_citations import DEFAULT_REVISION, main


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _universe_row(arxiv_id: str) -> dict:
    return {
        "source_id": "arxiv",
        "record_id": f"arxiv:{arxiv_id}",
        "content_hash": f"hash-{arxiv_id}",
        "arxiv_id": arxiv_id,
    }


def _decision_row(arxiv_id: str, keep: bool) -> dict:
    return {
        "source_id": "arxiv",
        "record_id": f"arxiv:{arxiv_id}",
        "content_hash": f"hash-{arxiv_id}",
        "qwen_should_keep": keep,
        "qwen_parse_status": "ok",
        "qwen_model": "Qwen/Qwen3-8B",
        "qwen_model_revision": DEFAULT_REVISION,
        "qwen_prompt_version": "test",
        "qwen_task": "arxiv_abstract",
    }


def test_select_accepted_citations_requires_complete_decisions(tmp_path):
    universe = tmp_path / "universe.parquet"
    decisions = tmp_path / "decisions.parquet"
    output = tmp_path / "accepted.txt"
    _write(universe, [_universe_row("2401.00001"), _universe_row("2401.00002")])
    _write(decisions, [_decision_row("2401.00001", True)])

    with pytest.raises(ValueError, match="incomplete"):
        main([
            "--universe", str(universe),
            "--decisions", str(decisions),
            "--output", str(output),
        ])


def test_select_accepted_citations_writes_sorted_kept_ids(tmp_path):
    universe = tmp_path / "universe.parquet"
    decisions = tmp_path / "decisions.parquet"
    output = tmp_path / "accepted.txt"
    rows = [
        _universe_row("2401.00002"),
        _universe_row("2401.00001"),
        _universe_row("2401.00003"),
    ]
    _write(universe, rows)
    _write(
        decisions,
        [
            _decision_row("2401.00002", True),
            _decision_row("2401.00001", False),
            _decision_row("2401.00003", True),
        ],
    )

    assert main([
        "--universe", str(universe),
        "--decisions", str(decisions),
        "--output", str(output),
    ]) == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "2401.00002",
        "2401.00003",
    ]
