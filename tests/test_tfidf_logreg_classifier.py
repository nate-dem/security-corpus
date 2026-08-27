import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


pytest.importorskip("joblib")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import pandas as pd

from classify.tfidf_logreg import TfidfLogRegConfig, score_parquet, train_model


def test_tfidf_logreg_trains_and_scores_qa_quality_sidecar(tmp_path):
    labels = pd.DataFrame(
        [
            {
                "source_id": "stackoverflow",
                "record_id": "stackoverflow:1",
                "content_hash": "h1",
                "content": "detailed answer about TLS certificate validation",
                "qa_quality_label": 1,
            },
            {
                "source_id": "stackoverflow",
                "record_id": "stackoverflow:2",
                "content_hash": "h2",
                "content": "excellent explanation of password hashing and salts",
                "qa_quality_label": 1,
            },
            {
                "source_id": "reddit-test",
                "record_id": "reddit-test:1",
                "content_hash": "h3",
                "content": "lol same problem bump",
                "qa_quality_label": 0,
            },
            {
                "source_id": "reddit-test",
                "record_id": "reddit-test:2",
                "content_hash": "h4",
                "content": "thanks nevermind fixed it",
                "qa_quality_label": 0,
            },
        ]
    )
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)

    config = TfidfLogRegConfig(
        task_name="qa_quality",
        min_df=1,
        max_df=1.0,
        max_features=None,
        max_iter=200,
        random_state=7,
        validation_fraction=0.5,
    )
    model_dir = tmp_path / "model"
    train_model(labels_path, model_dir, config)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["training_rows"] == 2
    assert metrics["validation_rows"] == 2
    assert metrics["classes"] == [0, 1]
    assert isinstance(metrics["accuracy"], float)
    assert set(metrics["per_class"]) == {"0", "1"}

    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "stackoverflow",
                    "record_id": "stackoverflow:3",
                    "content_hash": "h5",
                    "content": "how does TLS certificate pinning work",
                },
                {
                    "source_id": "nvd",
                    "record_id": "nvd:CVE-2026-0001",
                    "content_hash": "h6",
                    "content": "CVE description should not be scored by qa_quality",
                }
            ]
        ),
        input_path,
    )
    output_path = tmp_path / "scores.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "old",
                    "record_id": "old:1",
                    "content_hash": "oldhash",
                }
            ]
        ),
        output_path,
    )

    score_parquet(input_path, output_path, model_dir, config, batch_size=1)

    outputs = pq.read_table(output_path).to_pylist()
    assert len(outputs) == 1
    output = outputs[0]
    assert output["source_id"] == "stackoverflow"
    assert "qa_quality_score" in output
    assert "qa_quality_prob_0" in output
    assert "qa_quality_prob_1" in output
    assert output["classifier_model"] == "tfidf_logreg"

    before_zero_match = pq.read_table(output_path).to_pylist()
    with pytest.raises(ValueError, match="existing output left untouched"):
        score_parquet(
            input_path,
            output_path,
            model_dir,
            config,
            batch_size=1,
            source_ids=["does-not-exist"],
        )
    assert pq.read_table(output_path).to_pylist() == before_zero_match


def test_qa_quality_training_rejects_non_numeric_labels(tmp_path):
    labels = pd.DataFrame(
        [
            {"content": "substantive answer", "qa_quality_label": "1"},
            {"content": "thin chatter", "qa_quality_label": "0"},
        ]
    )
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)

    config = TfidfLogRegConfig(task_name="qa_quality", min_df=1)
    with pytest.raises(ValueError, match="qa_quality labels must be numeric"):
        train_model(labels_path, tmp_path / "model", config)


def test_qa_quality_training_rejects_out_of_range_labels(tmp_path):
    labels = pd.DataFrame(
        [
            {"content": "substantive answer", "qa_quality_label": 4},
            {"content": "thin chatter", "qa_quality_label": 0},
        ]
    )
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)

    config = TfidfLogRegConfig(task_name="qa_quality", min_df=1)
    with pytest.raises(ValueError, match="allowed set 0, 1, 2, 3"):
        train_model(labels_path, tmp_path / "model", config)


def test_tfidf_logreg_trains_and_scores_qa_quality_binary_sidecar(tmp_path):
    labels = pd.DataFrame(
        [
            {
                "source_id": "stackoverflow",
                "record_id": "stackoverflow:1",
                "content_hash": "h1",
                "content": "detailed answer about TLS certificate validation",
                "qa_quality_binary_label": 1,
            },
            {
                "source_id": "stackoverflow",
                "record_id": "stackoverflow:2",
                "content_hash": "h2",
                "content": "excellent explanation of password hashing and salts",
                "qa_quality_binary_label": 1,
            },
            {
                "source_id": "reddit-test",
                "record_id": "reddit-test:1",
                "content_hash": "h3",
                "content": "lol same problem bump",
                "qa_quality_binary_label": 0,
            },
            {
                "source_id": "reddit-test",
                "record_id": "reddit-test:2",
                "content_hash": "h4",
                "content": "thanks nevermind fixed it",
                "qa_quality_binary_label": 0,
            },
        ]
    )
    labels_path = tmp_path / "binary_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    config = TfidfLogRegConfig(
        task_name="qa_quality_binary",
        min_df=1,
        max_df=1.0,
        max_features=None,
        max_iter=200,
        random_state=7,
        validation_fraction=0.5,
    )
    model_dir = tmp_path / "binary_model"
    train_model(labels_path, model_dir, config)

    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_id": "stackoverflow",
                    "record_id": "stackoverflow:3",
                    "content_hash": "h5",
                    "content": "how does TLS certificate pinning work",
                }
            ]
        ),
        input_path,
    )
    output_path = tmp_path / "binary_scores.parquet"

    score_parquet(input_path, output_path, model_dir, config, batch_size=1)

    output = pq.read_table(output_path).to_pylist()[0]
    assert output["source_id"] == "stackoverflow"
    assert "qa_quality_binary_predicted_label" in output
    assert "qa_quality_binary_score" in output
    assert "qa_quality_binary_prob_0" in output
    assert "qa_quality_binary_prob_1" in output
    assert output["classifier_model"] == "tfidf_logreg"


def test_qa_quality_binary_training_rejects_out_of_range_labels(tmp_path):
    labels = pd.DataFrame(
        [
            {"content": "substantive answer", "qa_quality_binary_label": 2},
            {"content": "thin chatter", "qa_quality_binary_label": 0},
        ]
    )
    labels_path = tmp_path / "binary_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    config = TfidfLogRegConfig(task_name="qa_quality_binary", min_df=1)
    with pytest.raises(ValueError, match="allowed set 0, 1"):
        train_model(labels_path, tmp_path / "binary_model", config)


def test_qa_quality_binary_training_rejects_non_numeric_labels(tmp_path):
    labels = pd.DataFrame(
        [
            {"content": "substantive answer", "qa_quality_binary_label": "1"},
            {"content": "thin chatter", "qa_quality_binary_label": "0"},
        ]
    )
    labels_path = tmp_path / "binary_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    config = TfidfLogRegConfig(task_name="qa_quality_binary", min_df=1)
    with pytest.raises(ValueError, match="qa_quality_binary labels must be numeric"):
        train_model(labels_path, tmp_path / "binary_model", config)
