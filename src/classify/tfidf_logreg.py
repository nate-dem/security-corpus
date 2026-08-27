"""TF-IDF plus logistic regression classifier implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
import re
from typing import Any, Sequence
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from classify.io import ensure_parent, read_json, write_json
from classify.schema import (
    CLASSIFIER_MODEL_COLUMN,
    CLASSIFIER_VERSION_COLUMN,
    ID_COLUMNS,
    SCORED_AT_COLUMN,
    TEXT_COLUMN,
    get_task,
)


MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"
CLASSIFIER_MODEL_NAME = "tfidf_logreg"
QA_LABEL_VALUES_BY_TASK = {
    "qa_quality": {0, 1, 2, 3},
    "qa_quality_binary": {0, 1},
}
QA_CLASSIFIER_TASKS = frozenset(QA_LABEL_VALUES_BY_TASK)
QA_SOURCE_IDS = ("stackoverflow",)
QA_SOURCE_PATTERNS = ("stackexchange-*", "reddit-*")


@dataclass(frozen=True)
class TfidfLogRegConfig:
    """Configuration for one TF-IDF plus logistic regression task."""

    task_name: str
    text_column: str = TEXT_COLUMN
    label_column: str | None = None
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.95
    max_features: int | None = 500_000
    class_weight: str | None = "balanced"
    max_iter: int = 1000
    random_state: int = 13
    validation_fraction: float = 0.2

    def resolved_label_column(self) -> str:
        """Return the explicit or task-default label column."""
        if self.label_column:
            return self.label_column
        return get_task(self.task_name).label_column


def build_pipeline(config: TfidfLogRegConfig):
    """Build a scikit-learn TF-IDF plus logistic regression pipeline."""
    _validate_config(config)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(config.ngram_min, config.ngram_max),
                    min_df=config.min_df,
                    max_df=config.max_df,
                    max_features=config.max_features,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight=config.class_weight,
                    max_iter=config.max_iter,
                    n_jobs=-1,
                    random_state=config.random_state,
                    solver="saga",
                ),
            ),
        ]
    )


def train_model(labels_path: Path, model_dir: Path, config: TfidfLogRegConfig) -> None:
    """Train and persist one classifier model.

    Expected label input columns:
    - source_id
    - record_id
    - content_hash
    - content, or config.text_column
    - config.resolved_label_column()
    """
    _validate_config(config)
    task = get_task(config.task_name)
    label_column = config.resolved_label_column()

    frame = _read_labels(labels_path)
    _require_columns(frame, [config.text_column, label_column], labels_path)

    training_frame = frame[[config.text_column, label_column]].copy()
    training_frame = training_frame.dropna(subset=[config.text_column, label_column])
    training_frame[config.text_column] = training_frame[config.text_column].astype(str)
    training_frame = training_frame[training_frame[config.text_column].str.strip() != ""]
    training_frame = _validate_and_prepare_labels(training_frame, label_column, config)

    if training_frame.empty:
        raise ValueError(f"No usable training rows found in {labels_path}")

    label_counts = training_frame[label_column].value_counts(dropna=False)
    if len(label_counts) < 2:
        raise ValueError(
            f"Need at least two classes in {label_column}; found {len(label_counts)}"
        )

    metrics, final_training_frame = _train_validation_metrics(training_frame, config, label_column)

    pipeline = build_pipeline(config)
    pipeline.fit(final_training_frame[config.text_column], final_training_frame[label_column])

    model_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    import sklearn

    joblib.dump(pipeline, model_dir / MODEL_FILENAME)

    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "classifier_model": CLASSIFIER_MODEL_NAME,
        "classifier_version": _default_classifier_version(config.task_name, trained_at),
        "trained_at": trained_at,
        "task_name": config.task_name,
        "task_description": task.description,
        "text_column": config.text_column,
        "label_column": label_column,
        "score_column": task.score_column,
        "probability_prefix": task.probability_prefix,
        "classes": [_json_scalar(value) for value in pipeline.classes_.tolist()],
        "class_counts": {
            str(_json_scalar(label)): int(count)
            for label, count in label_counts.items()
        },
        "training_rows": int(len(training_frame)),
        "config": _json_ready(asdict(config)),
        "sklearn_version": sklearn.__version__,
    }
    write_json(model_dir / METADATA_FILENAME, metadata)
    write_json(model_dir / METRICS_FILENAME, metrics)


def score_parquet(
    input_path: Path,
    output_path: Path,
    model_dir: Path,
    config: TfidfLogRegConfig,
    batch_size: int,
    *,
    source_ids: Sequence[str] | None = None,
    source_like: Sequence[str] | None = None,
) -> None:
    """Score normalized corpus Parquet and write sidecar score Parquet.

    The output should be keyed by source_id, record_id, and content_hash, with
    classifier metadata and task score/probability columns.
    """
    _validate_config(config)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    task = get_task(config.task_name)
    if config.task_name in QA_CLASSIFIER_TASKS and not source_ids and not source_like:
        source_ids = QA_SOURCE_IDS
        source_like = QA_SOURCE_PATTERNS
    metadata = read_json(model_dir / METADATA_FILENAME)
    if metadata.get("task_name") != config.task_name:
        raise ValueError(
            f"Model task {metadata.get('task_name')!r} does not match "
            f"requested task {config.task_name!r}"
        )

    import joblib

    model = joblib.load(model_dir / MODEL_FILENAME)
    classes = list(model.classes_)
    probability_columns = _probability_columns(task.probability_prefix, classes)
    numeric_classes = _numeric_classes(classes)

    dataset = ds.dataset(str(input_path), format="parquet", partitioning="hive")
    required_columns = [*ID_COLUMNS, config.text_column]
    missing = [name for name in required_columns if name not in dataset.schema.names]
    if missing:
        raise ValueError(
            f"Input {input_path} is missing required columns: {', '.join(missing)}"
        )

    ensure_parent(output_path)
    temp_output_path = output_path.with_name(
        f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    )

    classifier_model = metadata.get("classifier_model", CLASSIFIER_MODEL_NAME)
    classifier_version = metadata.get("classifier_version", "unknown")
    scored_at = datetime.now(timezone.utc).isoformat()

    writer: pq.ParquetWriter | None = None
    scored_rows = 0
    scanner = dataset.scanner(columns=required_columns, batch_size=batch_size)

    try:
        try:
            for batch in scanner.to_batches():
                if batch.num_rows == 0:
                    continue

                frame = batch.to_pandas()
                frame = _filter_sources(frame, source_ids=source_ids, source_like=source_like)
                if frame.empty:
                    continue
                texts = frame[config.text_column].fillna("").astype(str)
                probabilities = model.predict_proba(texts)
                predicted_indexes = np.argmax(probabilities, axis=1)
                predicted_labels = [
                    str(_json_scalar(classes[index]))
                    for index in predicted_indexes
                ]

                output_frame = frame[list(ID_COLUMNS)].copy()
                output_frame[CLASSIFIER_MODEL_COLUMN] = classifier_model
                output_frame[CLASSIFIER_VERSION_COLUMN] = classifier_version
                output_frame[SCORED_AT_COLUMN] = scored_at
                output_frame[f"{task.name}_predicted_label"] = predicted_labels
                output_frame[task.score_column] = _score_from_probabilities(
                    probabilities,
                    numeric_classes,
                )

                for index, column in enumerate(probability_columns):
                    output_frame[column] = probabilities[:, index]

                table = pa.Table.from_pandas(output_frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp_output_path, table.schema)
                writer.write_table(table)
                scored_rows += len(frame)
        finally:
            if writer is not None:
                writer.close()

        if scored_rows == 0:
            raise ValueError(
                f"No rows found to score in {input_path}; existing output left untouched"
            )
        temp_output_path.replace(output_path)
    except Exception:
        _unlink_if_exists(temp_output_path)
        raise


def _validate_config(config: TfidfLogRegConfig) -> None:
    if config.ngram_min < 1:
        raise ValueError("ngram_min must be >= 1")
    if config.ngram_max < config.ngram_min:
        raise ValueError("ngram_max must be >= ngram_min")
    if config.min_df < 1:
        raise ValueError("min_df must be >= 1")
    if not 0 < config.max_df <= 1:
        raise ValueError("max_df must be in the interval (0, 1]")
    if config.max_features is not None and config.max_features < 1:
        raise ValueError("max_features must be positive or None")
    if config.max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if config.class_weight not in ("balanced", None):
        raise ValueError("class_weight must be 'balanced' or None")
    if not 0 <= config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in the interval [0, 1)")


def _validate_and_prepare_labels(
    frame: pd.DataFrame,
    label_column: str,
    config: TfidfLogRegConfig,
) -> pd.DataFrame:
    allowed_values = QA_LABEL_VALUES_BY_TASK.get(config.task_name)
    if allowed_values is None:
        return frame

    if not pd.api.types.is_numeric_dtype(frame[label_column]):
        allowed_display = _allowed_label_display(allowed_values)
        raise ValueError(
            f"{config.task_name} labels must be numeric ordinal values {allowed_display}; "
            f"column {label_column!r} has dtype {frame[label_column].dtype}"
        )

    numeric = pd.to_numeric(frame[label_column], errors="raise")
    integral = numeric.dropna().map(lambda value: float(value).is_integer())
    if not bool(integral.all()):
        allowed_display = _allowed_label_display(allowed_values)
        raise ValueError(
            f"{config.task_name} labels must be integer ordinal values {allowed_display}"
        )

    invalid = sorted(set(int(value) for value in numeric if int(value) not in allowed_values))
    if invalid:
        allowed_display = _allowed_label_display(allowed_values)
        raise ValueError(
            f"{config.task_name} labels must be in the allowed set {allowed_display}; "
            f"found invalid labels: {invalid}"
        )

    prepared = frame.copy()
    prepared[label_column] = numeric.astype(int)
    return prepared


def _allowed_label_display(allowed_values: set[int]) -> str:
    return ", ".join(str(value) for value in sorted(allowed_values))


def _train_validation_metrics(
    training_frame: pd.DataFrame,
    config: TfidfLogRegConfig,
    label_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    classes = sorted(_json_scalar(value) for value in training_frame[label_column].unique().tolist())
    split = _make_validation_split(training_frame, config, label_column)
    if split is None:
        metrics = {
            "classifier_model": CLASSIFIER_MODEL_NAME,
            "task_name": config.task_name,
            "training_rows": int(len(training_frame)),
            "validation_rows": 0,
            "classes": classes,
            "accuracy": None,
            "per_class": {},
            "confusion_matrix": [],
            "validation_status": "skipped",
            "validation_reason": _validation_skip_reason(training_frame, config, label_column),
        }
        return metrics, training_frame

    train_frame, validation_frame = split
    eval_pipeline = build_pipeline(config)
    eval_pipeline.fit(train_frame[config.text_column], train_frame[label_column])
    predictions = eval_pipeline.predict(validation_frame[config.text_column])

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    report = classification_report(
        validation_frame[label_column],
        predictions,
        labels=eval_pipeline.classes_,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        validation_frame[label_column],
        predictions,
        labels=eval_pipeline.classes_,
    )
    metrics = {
        "classifier_model": CLASSIFIER_MODEL_NAME,
        "task_name": config.task_name,
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "classes": [_json_scalar(value) for value in eval_pipeline.classes_.tolist()],
        "accuracy": float(accuracy_score(validation_frame[label_column], predictions)),
        "per_class": _per_class_metrics(report, eval_pipeline.classes_),
        "confusion_matrix": matrix.astype(int).tolist(),
        "validation_status": "ok",
        "validation_reason": None,
    }
    return metrics, training_frame


def _make_validation_split(
    training_frame: pd.DataFrame,
    config: TfidfLogRegConfig,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if _validation_skip_reason(training_frame, config, label_column) is not None:
        return None

    from sklearn.model_selection import train_test_split

    train_frame, validation_frame = train_test_split(
        training_frame,
        test_size=config.validation_fraction,
        random_state=config.random_state,
        stratify=training_frame[label_column],
    )
    return train_frame, validation_frame


def _validation_skip_reason(
    training_frame: pd.DataFrame,
    config: TfidfLogRegConfig,
    label_column: str,
) -> str | None:
    if config.validation_fraction <= 0:
        return "validation_fraction_is_zero"
    label_counts = training_frame[label_column].value_counts(dropna=False)
    num_classes = int(len(label_counts))
    if num_classes < 2:
        return "need_at_least_two_classes"
    if int(label_counts.min()) < 2:
        return "need_at_least_two_examples_per_class_for_stratified_validation"
    validation_rows = int(np.ceil(len(training_frame) * config.validation_fraction))
    training_rows = len(training_frame) - validation_rows
    if validation_rows < num_classes:
        return "validation_split_too_small_for_all_classes"
    if training_rows < num_classes:
        return "training_split_too_small_for_all_classes"
    return None


def _per_class_metrics(report: dict[str, Any], classes: Sequence[Any]) -> dict[str, dict[str, float]]:
    per_class: dict[str, dict[str, float]] = {}
    for class_value in classes:
        class_key = str(_json_scalar(class_value))
        class_report = report.get(class_key, {})
        per_class[class_key] = {
            "precision": float(class_report.get("precision", 0.0)),
            "recall": float(class_report.get("recall", 0.0)),
            "f1": float(class_report.get("f1-score", 0.0)),
            "support": int(class_report.get("support", 0)),
        }
    return per_class


def _filter_sources(
    frame: pd.DataFrame,
    *,
    source_ids: Sequence[str] | None,
    source_like: Sequence[str] | None,
) -> pd.DataFrame:
    exact = {source_id for source_id in (source_ids or []) if source_id}
    patterns = [pattern for pattern in (source_like or []) if pattern]
    if not exact and not patterns:
        return frame

    source_values = frame["source_id"].fillna("").astype(str)
    mask = pd.Series(False, index=frame.index)
    if exact:
        mask = mask | source_values.isin(exact)
    for pattern in patterns:
        mask = mask | source_values.map(lambda value, pattern=pattern: fnmatchcase(value, pattern))
    return frame[mask].copy()


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _read_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)

    raise ValueError(f"Unsupported labels format: {path}. Use .csv, .csv.gz, or .parquet")


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _default_classifier_version(task_name: str, trained_at: str) -> str:
    compact_time = re.sub(r"[^0-9A-Za-z]+", "", trained_at)
    return f"{CLASSIFIER_MODEL_NAME}-{task_name}-{compact_time}"


def _probability_columns(prefix: str, classes: list[Any]) -> list[str]:
    columns: list[str] = []
    seen: dict[str, int] = {}
    for class_value in classes:
        base = f"{prefix}_{_safe_column_suffix(class_value)}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        columns.append(base if count == 0 else f"{base}_{count}")
    return columns


def _safe_column_suffix(value: Any) -> str:
    suffix = re.sub(r"[^0-9A-Za-z_]+", "_", str(_json_scalar(value))).strip("_")
    return suffix or "class"


def _numeric_classes(classes: list[Any]) -> np.ndarray | None:
    try:
        return np.asarray([float(value) for value in classes], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _score_from_probabilities(
    probabilities: np.ndarray,
    numeric_classes: np.ndarray | None,
) -> np.ndarray:
    if numeric_classes is None:
        return np.max(probabilities, axis=1)
    return probabilities @ numeric_classes


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
