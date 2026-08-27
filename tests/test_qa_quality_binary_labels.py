import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest


pytest.importorskip("pandas")

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_derive_qa_quality_binary_labels_maps_4_class_labels(tmp_path):
    input_path = tmp_path / "qa_quality.parquet"
    output_path = tmp_path / "qa_quality_binary.parquet"
    frame = pd.DataFrame(
        [
            {
                "source_id": "stackoverflow",
                "record_id": "stackoverflow:0",
                "content_hash": "h0",
                "content": "junk",
                "qa_quality_label": 0,
                "label_notes": "broken",
                "score": 0,
            },
            {
                "source_id": "reddit-netsec",
                "record_id": "reddit-netsec:1",
                "content_hash": "h1",
                "content": "thin",
                "qa_quality_label": 1,
                "label_notes": "thin",
                "score": 1,
            },
            {
                "source_id": "stackexchange-infosec",
                "record_id": "stackexchange-infosec:2",
                "content_hash": "h2",
                "content": "usable",
                "qa_quality_label": 2,
                "label_notes": "usable",
                "score": 2,
            },
            {
                "source_id": "stackoverflow",
                "record_id": "stackoverflow:3",
                "content_hash": "h3",
                "content": "substantive",
                "qa_quality_label": 3,
                "label_notes": "strong",
                "score": 3,
            },
        ]
    )
    frame.to_parquet(input_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/classify/derive_qa_quality_binary_labels.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    rows = pq.read_table(output_path).to_pylist()
    assert [row["qa_quality_binary_label"] for row in rows] == [0, 0, 1, 1]
    assert [row["qa_quality_label"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["label_notes"] == "broken"
    assert rows[0]["score"] == 0
    assert rows[0]["qa_quality_binary_label_source"] == "derived_from_qa_quality_label"


def test_derive_qa_quality_binary_labels_rejects_invalid_source_labels(tmp_path):
    input_path = tmp_path / "qa_quality.parquet"
    output_path = tmp_path / "qa_quality_binary.parquet"
    pd.DataFrame(
        [
            {
                "source_id": "stackoverflow",
                "record_id": "stackoverflow:4",
                "content_hash": "h4",
                "content": "bad label",
                "qa_quality_label": 4,
            }
        ]
    ).to_parquet(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/classify/derive_qa_quality_binary_labels.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "qa_quality_label must contain labels in 0..3" in result.stderr
    assert not output_path.exists()
