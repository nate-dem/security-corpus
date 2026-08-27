"""Local SQLite run history for Security Scope web queries."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunHistory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS web_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    route_json TEXT NOT NULL,
                    router_model TEXT NOT NULL,
                    planner_model TEXT NOT NULL,
                    synthesis_model TEXT NOT NULL,
                    validated_operations_json TEXT NOT NULL,
                    rendered_command_trace_json TEXT NOT NULL,
                    operation_outputs_json TEXT NOT NULL,
                    answer_markdown TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    planner_repair_json TEXT,
                    error_details_json TEXT
                )
                """
            )
            _ensure_columns(con, "web_runs", {"planner_repair_json": "TEXT", "error_details_json": "TEXT"})

    def save(self, run: dict[str, Any]) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO web_runs (
                    run_id, created_at, user_query, route_json, router_model,
                    planner_model, synthesis_model, validated_operations_json,
                    rendered_command_trace_json, operation_outputs_json,
                    answer_markdown, citations_json, errors_json, latency_ms,
                    planner_repair_json, error_details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run["run_id"],
                    run.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    run.get("query", ""),
                    json.dumps(run.get("route", {}), sort_keys=True),
                    run.get("router_model", ""),
                    run.get("planner_model", ""),
                    run.get("synthesis_model", ""),
                    json.dumps(run.get("operations", []), sort_keys=True),
                    json.dumps(run.get("command_trace", []), sort_keys=True),
                    json.dumps(run.get("operation_outputs", []), sort_keys=True),
                    run.get("answer_markdown", ""),
                    json.dumps(run.get("citations", []), sort_keys=True),
                    json.dumps(run.get("errors", []), sort_keys=True),
                    int(run.get("latency_ms") or 0),
                    json.dumps(run["planner_repair"], sort_keys=True) if run.get("planner_repair") is not None else None,
                    json.dumps(run.get("error_details", []), sort_keys=True),
                ),
            )

    def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT run_id, created_at, user_query, answer_markdown, errors_json, latency_ms
                FROM web_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "query": row["user_query"],
                "answer_preview": row["answer_markdown"][:240],
                "errors": json.loads(row["errors_json"]),
                "latency_ms": row["latency_ms"],
            }
            for row in rows
        ]

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM web_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        operation_outputs = json.loads(row["operation_outputs_json"])
        return {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "status": "done" if not json.loads(row["errors_json"]) else "done_with_errors",
            "query": row["user_query"],
            "route": json.loads(row["route_json"]),
            "router_model": row["router_model"],
            "planner_model": row["planner_model"],
            "synthesis_model": row["synthesis_model"],
            "operations": json.loads(row["validated_operations_json"]),
            "command_trace": json.loads(row["rendered_command_trace_json"]),
            "operation_outputs": operation_outputs,
            "answer_markdown": row["answer_markdown"],
            "citations": json.loads(row["citations_json"]),
            "sources": _sources_from_outputs(operation_outputs),
            "errors": json.loads(row["errors_json"]),
            "error_details": json.loads(row["error_details_json"]) if row["error_details_json"] else [],
            "planner_repair": json.loads(row["planner_repair_json"]) if row["planner_repair_json"] else None,
            "latency_ms": row["latency_ms"],
        }


def _ensure_columns(con: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}
    for name, declared_type in columns.items():
        if name not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declared_type}")


def _sources_from_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for output in outputs:
        for source in output.get("sources", []):
            key = source.get("path")
            if not key or key in seen:
                continue
            seen.add(key)
            sources.append(source)
    return sources
