import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[1]


def test_select_qa_qwen_candidates_uses_explicit_thresholds(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "stackoverflow",
                    "record_id": "stackoverflow:1",
                    "content_hash": "h1",
                    "score": 0,
                    "answer_count": 0,
                    "has_accepted_answer": False,
                    "closed": False,
                },
                {
                    "source_id": "stackoverflow",
                    "record_id": "stackoverflow:2",
                    "content_hash": "h2",
                    "score": 3,
                    "answer_count": 1,
                    "has_accepted_answer": True,
                    "closed": False,
                },
            ]
        ),
        corpus_path,
    )
    quality_path = tmp_path / "quality.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "stackoverflow",
                    "record_id": "stackoverflow:1",
                    "content_hash": "h1",
                    "qa_quality_binary_score": 0.2,
                    "qa_quality_binary_predicted_label": "0",
                    "qa_quality_binary_prob_0": 0.9,
                    "qa_quality_binary_prob_1": 0.1,
                    "classifier_model": "tfidf_logreg",
                    "classifier_version": "test",
                    "scored_at": "2026-05-25T00:00:00+00:00",
                },
                {
                    "source_id": "stackoverflow",
                    "record_id": "stackoverflow:2",
                    "content_hash": "h2",
                    "qa_quality_binary_score": 0.8,
                    "qa_quality_binary_predicted_label": "1",
                    "qa_quality_binary_prob_0": 0.2,
                    "qa_quality_binary_prob_1": 0.8,
                    "classifier_model": "tfidf_logreg",
                    "classifier_version": "test",
                    "scored_at": "2026-05-25T00:00:00+00:00",
                },
            ]
        ),
        quality_path,
    )
    output_path = tmp_path / "candidates.parquet"

    subprocess.run(
        [
            sys.executable,
            "scripts/classify/select_qa_qwen_candidates.py",
            "--corpus",
            str(corpus_path),
            "--quality-sidecar",
            str(quality_path),
            "--output",
            str(output_path),
            "--min-quality-score",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    rows = {
        row["record_id"]: row
        for row in pq.read_table(output_path).to_pylist()
    }
    assert rows["stackoverflow:1"]["qa_candidate_for_qwen"] is False
    assert rows["stackoverflow:1"]["qa_candidate_reason"] == "below_researcher_thresholds"
    assert rows["stackoverflow:2"]["qa_candidate_for_qwen"] is True
    assert rows["stackoverflow:2"]["qa_candidate_reason"] == "quality_score_threshold"
