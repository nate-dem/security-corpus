"""Fail-closed checks shared by Qwen release gates.

These validate decision integrity, not semantic quality or acceptance thresholds.
"""

from __future__ import annotations

import duckdb


def decision_issues(
    connection: duckdb.DuckDBPyConnection,
    *,
    expected_model: str | None = None,
    expected_revision: str | None = None,
    expected_prompt_version: str | None = None,
    expected_task: str | None = None,
) -> dict[str, int]:
    """Inspect the caller's ``decisions`` view, preserving SQL NULL failures."""
    columns = dict(
        (row[0], row[1])
        for row in connection.execute("DESCRIBE decisions").fetchall()
    )
    required = {
        "source_id", "record_id", "content_hash", "qwen_should_keep",
        "qwen_parse_status", "qwen_model", "qwen_model_revision",
        "qwen_prompt_version", "qwen_scored_at", "qwen_task",
    }
    missing = required - columns.keys()
    if missing:
        raise ValueError("Decisions are missing columns: " + ", ".join(sorted(missing)))

    counts = connection.execute("""
        SELECT
            count(*),
            count(*) FILTER (WHERE qwen_should_keep IS NULL),
            count(*) FILTER (WHERE qwen_parse_status IS NULL
                OR qwen_parse_status NOT IN ('ok', 'extracted_json')),
            count(*) FILTER (WHERE
                source_id IS NULL OR trim(source_id) = ''
                OR record_id IS NULL OR trim(record_id) = ''
                OR content_hash IS NULL OR trim(content_hash) = ''),
            count(*) FILTER (WHERE
                qwen_model IS NULL OR trim(qwen_model) = ''
                OR qwen_model_revision IS NULL
                OR NOT regexp_full_match(qwen_model_revision, '[0-9a-f]{40}')
                OR qwen_prompt_version IS NULL OR trim(qwen_prompt_version) = ''
                OR try_cast(qwen_scored_at AS TIMESTAMPTZ) IS NULL
                OR qwen_task IS NULL
                OR qwen_task NOT IN ('qa', 'arxiv_abstract', 'arxiv_full')),
            count(DISTINCT (qwen_model, qwen_model_revision,
                           qwen_prompt_version, qwen_task))
        FROM decisions
    """).fetchone()
    issues = {
        "empty_decisions": int(counts[0] == 0),
        "undecided_rows": int(counts[1]),
        "invalid_parse_rows": int(counts[2]),
        "invalid_decision_keys": int(counts[3]),
        "invalid_provenance_rows": int(counts[4]),
        "mixed_decision_configurations": max(int(counts[5]) - 1, 0),
        "non_boolean_decision_column": int(columns["qwen_should_keep"] != "BOOLEAN"),
    }
    clauses, parameters = [], []
    for column, expected in (
        ("qwen_model", expected_model),
        ("qwen_model_revision", expected_revision),
        ("qwen_prompt_version", expected_prompt_version),
        ("qwen_task", expected_task),
    ):
        if expected is not None:
            clauses.append(f"{column} IS DISTINCT FROM ?")
            parameters.append(expected)
    if clauses:
        issues["expected_configuration_mismatches"] = int(connection.execute(
            "SELECT count(*) FROM decisions WHERE " + " OR ".join(clauses), parameters,
        ).fetchone()[0])
    return {name: count for name, count in issues.items() if count}
