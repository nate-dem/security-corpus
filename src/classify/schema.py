"""Classifier task and sidecar score schema conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


ID_COLUMNS: Final[tuple[str, ...]] = ("source_id", "record_id", "content_hash")
TEXT_COLUMN: Final[str] = "content"
CLASSIFIER_VERSION_COLUMN: Final[str] = "classifier_version"
CLASSIFIER_MODEL_COLUMN: Final[str] = "classifier_model"
SCORED_AT_COLUMN: Final[str] = "scored_at"


@dataclass(frozen=True)
class TaskSpec:
    """A single classifier target."""
    name: str
    label_column: str
    score_column: str
    probability_prefix: str
    description: str


DEFAULT_TASKS: Final[dict[str, TaskSpec]] = {
    "qa_quality": TaskSpec(
        name="qa_quality",
        label_column="qa_quality_label",
        score_column="qa_quality_score",
        probability_prefix="qa_quality_prob",
        description=(
            "Conservative Q&A/social quality score used only to remove "
            "obvious low-value records before Qwen review."
        ),
    ),
    "qa_quality_binary": TaskSpec(
        name="qa_quality_binary",
        label_column="qa_quality_binary_label",
        score_column="qa_quality_binary_score",
        probability_prefix="qa_quality_binary_prob",
        description=(
            "Binary Q&A/social prefilter: whether a record is high enough "
            "quality to spend Qwen inference on."
        ),
    ),
    "security_relevance": TaskSpec(
        name="security_relevance",
        label_column="security_relevance_label",
        score_column="security_relevance_score",
        probability_prefix="security_relevance_prob",
        description="How directly security-relevant the record is.",
    ),
    "quality": TaskSpec(
        name="quality",
        label_column="quality_label",
        score_column="quality_score",
        probability_prefix="quality_prob",
        description="How coherent, substantive, and useful the record is as text.",
    )
}


def get_task(name: str) -> TaskSpec:
    """Return a known task spec by name."""
    return DEFAULT_TASKS[name]
