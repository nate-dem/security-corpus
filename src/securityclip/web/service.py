"""Structured agentic retrieval orchestration for Security Scope web."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from securityclip.web.errors import ErrorCollector, classify_operation_error
from securityclip.web.executor import OperationExecutor, OperationOutput, collect_citations, collect_sources
from securityclip.web.history import RunHistory
from securityclip.web.llm import ModelClient, OpenAIModelClient
from securityclip.web.operations import (
    ALLOWED_ROOTS,
    OperationValidationError,
    ValidatedOperation,
    render_command,
    validate_operation,
    validate_operations,
)
from securityclip.web.routing import Route, deterministic_operations, deterministic_route
from securityclip.web.settings import WebSettings


ROUTER_SYSTEM = """You classify natural-language security corpus queries for a retrieval app.
Return JSON with route_type, entities, likely_roots, confidence, and reason.
Route types: exact_identifier, topic_research, source_list, paper_research,
log_or_rule_search, path_inspection, general."""

PLANNER_SYSTEM = """You plan structured Security Scope retrieval operations.
Return only a JSON array of operation objects. Never return shell commands.
Allowed tools:
search(query, limit)
grep(pattern, path, ignore_case, limit)
grep_from(handle, pattern, ignore_case, limit)
cat(path)
head(path, count)
ls(path, limit)
Allowed roots: /papers, /qa, /vulns, /rules, /logs, /knowledge, /transcripts, /web.
If the input lists allowed_roots, only plan operations against those roots.
Use bounded operations. Treat 'all documents' as top matching documents up to the cap."""

ROOT_DISPLAY_NAMES: dict[str, str] = {
    "/papers": "Academic Papers",
    "/qa": "Community Q&A",
    "/vulns": "Vulnerabilities",
    "/rules": "Detection Rules",
    "/knowledge": "Knowledge Base",
    "/logs": "Logs",
    "/transcripts": "Transcripts",
    "/web": "Web",
}

SYNTHESIS_SYSTEM = """You synthesize security corpus retrieval results into a report-style Markdown answer.

Formatting rules:
- Organize the answer into `##` sections following the provided answer skeleton; omit sections with no evidence.
- Refer to corpus areas by display name, never by raw root path:
  /papers = Academic Papers, /qa = Community Q&A, /vulns = Vulnerabilities, /rules = Detection Rules,
  /knowledge = Knowledge Base, /logs = Logs, /transcripts = Transcripts.
  /web = Web.
- The app separately renders the full clickable list of matched documents grouped by source type;
  do not enumerate every matched document. Summarize what was found in each corpus area
  (using `citations_by_root`, `sources_by_root`, and `counts_by_root`) and call out only the most important records.
- Keep prose concise. Use a GitHub-flavored Markdown table only when comparing a handful of records.

Citation rules:
- Cite every substantive claim with an exact citation string from the evidence, wrapped in backticks,
  e.g. `/papers/arx_2510.19844/content.lines:L9`.
- Never invent citations or paths. Only use citation strings that appear in the evidence.
- Avoid vague references like "see the sources list"; point at specific cited lines.

Closing rules:
- The answer must read as a complete report. Do not end with a question. Do not ask the user what they want next.
- When useful, end with a `## Suggested follow-up commands` section containing Security Scope CLI commands in a fenced code block.
- Mention when results are capped (see `cap_notice`).
- If a `root_filter` note is present, state that results were restricted to those corpus areas."""

SYNTHESIS_TEMPLATES: dict[str, str] = {
    "source_list": """## Summary

## Where It Appears
(one short bullet per corpus area, with counts and the most important citation)

## Notes""",
    "exact_identifier": """## Summary

## Key Records

## Notes""",
    "topic_research": """## Overview

## Key Findings
(subsections per corpus area, using display names)

## Evidence Highlights""",
    "paper_research": """## Overview

## Key Papers

## Common Themes""",
    "log_or_rule_search": """## Summary

## Detections and Activity

## Evidence Highlights""",
    "path_inspection": """## Document

## Contents

## Key Lines""",
    "general": """## Summary

## Key Findings

## Evidence Highlights""",
}


class QueryService:
    def __init__(
        self,
        settings: WebSettings,
        *,
        model_client: ModelClient | None = None,
        history: RunHistory | None = None,
    ):
        self.settings = settings
        self.model_client = model_client or OpenAIModelClient()
        self.history = history or RunHistory(settings.web_db)

    def run_query(
        self,
        query: str,
        *,
        max_steps: int | None = None,
        max_results: int | None = None,
        roots: list[str] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        run_id = "run_" + uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()
        collector = ErrorCollector()
        max_steps = _bounded(max_steps, default=self.settings.max_commands, upper=self.settings.max_commands)
        max_results = _bounded(max_results, default=self.settings.max_results, upper=self.settings.max_limit)
        roots = normalize_requested_roots(roots)

        _emit(on_event, "routing")
        route, route_used_model = self._route(query, collector)
        if roots:
            route["likely_roots"] = [root for root in route.get("likely_roots") or [] if root in roots]

        _emit(on_event, "planning")
        raw_operations, planner_used_model = self._plan(
            query, route, max_steps=max_steps, max_results=max_results, roots=roots, collector=collector
        )

        _emit(on_event, "validating")
        operations, validation_warnings, repair_debug = self._validate_with_repair(
            query, route, raw_operations, collector, roots=roots, deterministic_plan=not planner_used_model
        )
        for warning in validation_warnings:
            collector.add("validation_failed", warning)

        executor = OperationExecutor(self.settings.index_dir, self.settings, roots=roots)
        operation_outputs: list[OperationOutput] = []
        for index, operation in enumerate(operations, start=1):
            _emit(
                on_event,
                "executing",
                operation_index=index,
                operation_count=len(operations),
                command=render_command(operation),
            )
            operation_outputs.append(executor.execute(operation))
        for output in operation_outputs:
            if not output.ok and output.error:
                collector.add(classify_operation_error(output.error), output.error)

        citations = collect_citations(operation_outputs)
        sources = collect_sources(operation_outputs)
        if not citations and not sources and all(output.ok for output in operation_outputs):
            collector.add("no_results", "no matching documents found", detail_only=True)

        _emit(on_event, "synthesizing")
        answer_markdown = self._synthesize(
            query, route, operations, operation_outputs, citations, sources, max_results, collector, roots=roots
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        errors = collector.messages
        response = {
            "run_id": run_id,
            "created_at": created_at,
            "status": "done" if not errors else "done_with_errors",
            "query": query,
            "route": route,
            "route_used_model": route_used_model,
            "planner_used_model": planner_used_model,
            "router_model": self.settings.router_model,
            "planner_model": self.settings.planner_model,
            "synthesis_model": self.settings.synthesis_model,
            "operations": [operation.to_json() for operation in operations],
            "command_trace": [render_command(operation) for operation in operations],
            "operation_outputs": [output.to_json() for output in operation_outputs],
            "answer_markdown": answer_markdown,
            "citations": citations,
            "sources": sources,
            "errors": errors,
            "error_details": collector.details,
            "planner_repair": repair_debug,
            "requested_roots": roots,
            "latency_ms": latency_ms,
            "caps": {"max_steps": max_steps, "max_results": max_results},
        }
        self.history.save(response)
        return response

    def _route(self, query: str, collector: ErrorCollector) -> tuple[dict[str, Any], bool]:
        deterministic = deterministic_route(query)
        if deterministic is not None and deterministic.confidence >= 0.95:
            return deterministic.to_json(), False
        if not self.settings.openai_api_key_present:
            if deterministic is not None:
                return deterministic.to_json(), False
            collector.add("missing_api_key", "OPENAI_API_KEY is not set; using deterministic fallback search")
            return Route("topic_research", confidence=0.3, deterministic=True, reason="missing OPENAI_API_KEY").to_json(), False
        try:
            route = self.model_client.generate_json(
                model=self.settings.router_model,
                system=ROUTER_SYSTEM,
                user=json.dumps({"query": query, "deterministic_hint": deterministic.to_json() if deterministic else None}),
            )
            if not isinstance(route, dict):
                raise ValueError("router response must be a JSON object")
            route.setdefault("deterministic", False)
            return _normalize_route_payload(route), True
        except Exception as exc:  # noqa: BLE001
            collector.add("router_failed", f"router failed: {exc}; using fallback search")
            return Route("topic_research", confidence=0.2, deterministic=True, reason="router failure").to_json(), False

    def _plan(
        self,
        query: str,
        route: dict[str, Any],
        *,
        max_steps: int,
        max_results: int,
        roots: list[str] | None,
        collector: ErrorCollector,
    ) -> tuple[Any, bool]:
        route = _normalize_route_payload(route)
        route_obj = Route(
            route_type=str(route.get("route_type") or "general"),
            entities=route.get("entities") or {},
            likely_roots=route.get("likely_roots") or [],
            confidence=float(route.get("confidence") or 0),
            deterministic=bool(route.get("deterministic")),
            reason=str(route.get("reason") or ""),
        )
        deterministic_plan = deterministic_operations(query, route_obj, max_results=max_results)
        if deterministic_plan and route_obj.confidence >= 0.95:
            return deterministic_plan[:max_steps], False
        if not self.settings.openai_api_key_present:
            collector.add("missing_api_key", "OPENAI_API_KEY is not set; using fallback search operation")
            return [{"tool": "search", "query": query, "limit": max_results}], False
        try:
            payload: dict[str, Any] = {
                "query": query,
                "route": route,
                "max_steps": max_steps,
                "max_results": max_results,
            }
            if roots:
                payload["allowed_roots"] = roots
            operations = self.model_client.generate_json(
                model=self.settings.planner_model,
                system=PLANNER_SYSTEM,
                user=json.dumps(payload, indent=2),
            )
            return operations, True
        except Exception as exc:  # noqa: BLE001
            collector.add("planner_failed", f"planner failed: {exc}; using fallback search operation")
            return [{"tool": "search", "query": query, "limit": max_results}], False

    def _validate_with_repair(
        self,
        query: str,
        route: dict[str, Any],
        raw_operations: Any,
        collector: ErrorCollector,
        *,
        roots: list[str] | None = None,
        deterministic_plan: bool = False,
    ) -> tuple[list[ValidatedOperation], list[str], dict[str, Any] | None]:
        allowed_roots = tuple(roots) if roots else ALLOWED_ROOTS
        try:
            operations, warnings = validate_operations(raw_operations, self.settings, allowed_roots=allowed_roots)
            return operations, warnings, None
        except OperationValidationError as first_error:
            repair_debug: dict[str, Any] = {
                "raw_operations": raw_operations,
                "validation_error": str(first_error),
                "repaired_operations": None,
                "repair_error": None,
                "repair_succeeded": False,
            }
            if deterministic_plan:
                # Deterministic plans must not consume a model repair call;
                # drop the invalid operations (e.g. paths outside selected roots).
                operations, warnings = _validate_dropping_invalid(raw_operations, self.settings, allowed_roots)
                collector.add(
                    "validation_failed",
                    f"deterministic plan dropped invalid operations: {first_error}",
                    detail_only=bool(operations),
                )
                if operations:
                    repair_debug["repaired_operations"] = [operation.to_json() for operation in operations]
                    repair_debug["repair_succeeded"] = True
                    return operations, warnings, repair_debug
                return (*self._fallback_operations(query), repair_debug)
            if not self.settings.openai_api_key_present:
                collector.add("validation_failed", f"planner operation validation failed: {first_error}; using fallback search")
                return (*self._fallback_operations(query), repair_debug)
            try:
                repair_payload: dict[str, Any] = {
                    "query": query,
                    "route": route,
                    "invalid_operations": raw_operations,
                    "validation_error": str(first_error),
                    "instruction": "Return a corrected JSON array of valid operations.",
                }
                if roots:
                    repair_payload["allowed_roots"] = roots
                repaired = self.model_client.generate_json(
                    model=self.settings.planner_model,
                    system=PLANNER_SYSTEM,
                    user=json.dumps(repair_payload, indent=2),
                )
                repair_debug["repaired_operations"] = repaired
                operations, warnings = validate_operations(repaired, self.settings, allowed_roots=allowed_roots)
                repair_debug["repair_succeeded"] = True
                return operations, warnings, repair_debug
            except Exception as second_error:  # noqa: BLE001
                repair_debug["repair_error"] = str(second_error)
                collector.add("repair_failed", f"planner repair failed: {second_error}; using fallback search")
                return (*self._fallback_operations(query), repair_debug)

    def _fallback_operations(self, query: str) -> tuple[list[ValidatedOperation], list[str]]:
        return validate_operations([{"tool": "search", "query": query, "limit": self.settings.max_results}], self.settings)

    def _synthesize(
        self,
        query: str,
        route: dict[str, Any],
        operations: list[ValidatedOperation],
        outputs: list[Any],
        citations: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        max_results: int,
        collector: ErrorCollector,
        *,
        roots: list[str] | None = None,
    ) -> str:
        if not self.settings.openai_api_key_present:
            return fallback_answer(query, citations, sources, max_results, collector.messages)
        _evidence, serialized = self._build_synthesis_evidence(
            query, route, operations, outputs, citations, sources, max_results, roots=roots
        )
        try:
            return self.model_client.generate_text(
                model=self.settings.synthesis_model,
                system=_synthesis_system(str(route.get("route_type") or "general")),
                user=serialized,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            collector.add("synthesis_failed", f"synthesis failed: {exc}; using fallback answer")
            return fallback_answer(query, citations, sources, max_results, collector.messages)

    def _build_synthesis_evidence(
        self,
        query: str,
        route: dict[str, Any],
        operations: list[ValidatedOperation],
        outputs: list[Any],
        citations: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        max_results: int,
        *,
        roots: list[str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Build a compact evidence payload for synthesis, clamped under the total cap.

        Retrieval/display caps (``max_results``) are deliberately decoupled from what the
        synthesis model receives. The full operation output stays in the API response and
        history; the model only sees truncated previews and per-root-capped
        citations/sources so the request body cannot grow large enough to trigger a proxy
        413. Exact citation strings are preserved so the answer keeps precise citations.
        """
        settings = self.settings
        base: dict[str, Any] = {
            "query": query,
            "route": route,
            "commands": [render_command(operation) for operation in operations],
            "counts_by_root": {
                "citations": _root_counts(citations),
                "sources": _root_counts(sources),
            },
            "cap_notice": f"Showing top matching documents up to the configured cap ({max_results}).",
        }
        if roots:
            base["root_filter"] = f"Results were restricted to corpus roots: {', '.join(roots)}."

        # Progressive shrink levels: (output preview chars, snippet chars, sources/root, citations/root).
        # The first level uses the configured caps; later levels shrink previews first, then
        # per-root counts. The final level is tiny and guaranteed under any reasonable budget.
        levels = [
            (
                settings.synthesis_max_output_chars_per_operation,
                240,
                settings.synthesis_max_sources_per_root,
                settings.synthesis_max_citations_per_root,
            ),
            (
                max(500, settings.synthesis_max_output_chars_per_operation // 2),
                160,
                settings.synthesis_max_sources_per_root,
                settings.synthesis_max_citations_per_root,
            ),
            (
                500,
                120,
                max(4, settings.synthesis_max_sources_per_root // 2),
                max(6, settings.synthesis_max_citations_per_root // 2),
            ),
            (250, 80, 3, 4),
            (0, 40, 2, 3),
        ]

        evidence: dict[str, Any] = dict(base)
        serialized = json.dumps(evidence)
        for level, (preview_chars, snippet_chars, source_cap, citation_cap) in enumerate(levels):
            evidence = dict(base)
            evidence["outputs"] = [_compact_output(output.to_json(), preview_chars=preview_chars) for output in outputs]
            evidence["sources_by_root"] = _grouped_compact(sources, source_cap, _compact_source)
            evidence["citations_by_root"] = _grouped_compact(
                citations, citation_cap, lambda item: _compact_citation(item, snippet_chars=snippet_chars)
            )
            if level > 0:
                evidence["evidence_truncated_for_synthesis"] = True
            serialized = json.dumps(evidence, indent=2)
            if len(serialized) <= settings.synthesis_max_total_chars:
                break
        return evidence, serialized


def fallback_answer(
    query: str,
    citations: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    max_results: int,
    errors: list[str] | None = None,
) -> str:
    lines = [
        "## Summary",
        "",
        f"Deterministic results for `{query}`. Showing top matching documents up to the configured cap ({max_results});"
        " the matched documents list below the answer has everything retrieved.",
    ]
    if sources:
        counts = _root_counts(sources)
        parts = [f"{count} in {ROOT_DISPLAY_NAMES.get(root, root)}" for root, count in sorted(counts.items())]
        lines.extend(["", f"Found {len(sources)} documents: {', '.join(parts)}."])
    if citations:
        lines.extend(["", "## Evidence Highlights"])
        for citation in citations[:10]:
            lines.append(f"- `{citation['citation']}`: {citation.get('snippet', '')[:240]}")
    if errors:
        lines.extend(["", "## Notes"])
        for error in errors:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def group_by_root(items: list[dict[str, Any]], *, per_root_cap: int = 15) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        bucket = grouped.setdefault(_root_of(item), [])
        if len(bucket) < per_root_cap:
            bucket.append(item)
    return grouped


def _grouped_compact(
    items: list[dict[str, Any]],
    per_root_cap: int,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        bucket = grouped.setdefault(_root_of(item), [])
        if len(bucket) < per_root_cap:
            bucket.append(transform(item))
    return grouped


def _compact_output(output: dict[str, Any], *, preview_chars: int) -> dict[str, Any]:
    preview = str(output.get("output_text") or "")
    if preview_chars <= 0:
        preview = ""
    elif len(preview) > preview_chars:
        preview = preview[:preview_chars] + "…"
    return {
        "command": output.get("command"),
        "ok": output.get("ok"),
        "error": output.get("error"),
        "latency_ms": output.get("latency_ms"),
        "truncated": output.get("truncated"),
        "source_count": len(output.get("sources") or []),
        "citation_count": len(output.get("citations") or []),
        "output_preview": preview,
    }


def _compact_citation(citation: dict[str, Any], *, snippet_chars: int) -> dict[str, Any]:
    snippet = str(citation.get("snippet") or "")
    if snippet_chars <= 0:
        snippet = ""
    elif len(snippet) > snippet_chars:
        snippet = snippet[:snippet_chars] + "…"
    return {
        "citation": citation.get("citation"),
        "path": citation.get("path"),
        "line_number": citation.get("line_number"),
        "snippet": snippet,
        "source_id": citation.get("source_id"),
        "title": citation.get("title"),
    }


def _compact_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": source.get("path"),
        "title": source.get("title"),
        "source_id": source.get("source_id"),
        "record_id": source.get("record_id"),
        "tokens": source.get("tokens"),
    }


def normalize_requested_roots(raw: list[str] | None) -> list[str] | None:
    """Normalize a user-requested root filter; raises ValueError on unknown roots.

    Returns None when no effective filter applies (no roots, or all roots selected).
    """
    if raw is None:
        return None
    roots: list[str] = []
    invalid: list[str] = []
    for value in raw:
        root = str(value).strip()
        if not root:
            continue
        if not root.startswith("/"):
            root = "/" + root
        root = root.rstrip("/")
        if root in ALLOWED_ROOTS:
            if root not in roots:
                roots.append(root)
        else:
            invalid.append(str(value))
    if invalid:
        raise ValueError(f"unknown corpus roots: {', '.join(invalid)}")
    if not roots or len(roots) == len(ALLOWED_ROOTS):
        return None
    return roots


def _synthesis_system(route_type: str) -> str:
    template = SYNTHESIS_TEMPLATES.get(route_type) or SYNTHESIS_TEMPLATES["general"]
    return f"{SYNTHESIS_SYSTEM}\n\nUse this answer skeleton (omit sections without evidence):\n\n{template}"


def _root_of(item: dict[str, Any]) -> str:
    parts = str(item.get("path") or "").split("/")
    if len(parts) > 1 and parts[0] == "" and parts[1]:
        return "/" + parts[1]
    return "other"


def _root_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        root = _root_of(item)
        counts[root] = counts.get(root, 0) + 1
    return counts


def _validate_dropping_invalid(
    raw_operations: Any,
    settings: WebSettings,
    allowed_roots: tuple[str, ...],
) -> tuple[list[ValidatedOperation], list[str]]:
    if isinstance(raw_operations, dict) and "operations" in raw_operations:
        raw_operations = raw_operations["operations"]
    if not isinstance(raw_operations, list):
        return [], []
    operations: list[ValidatedOperation] = []
    warnings: list[str] = []
    for raw in raw_operations[: settings.max_commands]:
        try:
            operation, op_warnings = validate_operation(raw, settings, allowed_roots=allowed_roots)
        except OperationValidationError:
            continue
        operations.append(operation)
        warnings.extend(op_warnings)
    return operations, warnings


def _emit(on_event: Callable[[dict[str, Any]], None] | None, stage: str, **fields: Any) -> None:
    if on_event is None:
        return
    event = {"stage": stage, "ts": datetime.now(timezone.utc).isoformat(), **fields}
    try:
        on_event(event)
    except Exception:  # noqa: BLE001 - progress callbacks must not break the run
        pass


def _bounded(value: int | None, *, default: int, upper: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, 1), upper)


def _normalize_route_payload(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_type": str(route.get("route_type") or "general"),
        "entities": _normalize_entities(route.get("entities")),
        "likely_roots": _normalize_likely_roots(route.get("likely_roots")),
        "confidence": _normalize_confidence(route.get("confidence")),
        "deterministic": bool(route.get("deterministic")),
        "reason": str(route.get("reason") or ""),
    }


def _normalize_entities(raw: Any) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            values = _string_list(value)
            if values:
                entities[str(key)] = values
        return entities
    values = _string_list(raw)
    if values:
        entities["terms"] = values
    return entities


def _normalize_likely_roots(raw: Any) -> list[str]:
    roots: list[str] = []
    for value in _string_list(raw):
        root = value.strip()
        if not root:
            continue
        if not root.startswith("/"):
            root = "/" + root
        root = root.rstrip("/")
        if root in ALLOWED_ROOTS and root not in roots:
            roots.append(root)
    return roots


def _normalize_confidence(raw: Any) -> float:
    try:
        confidence = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return min(max(confidence, 0.0), 1.0)


def _string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        values: list[str] = []
        for value in raw.values():
            values.extend(_string_list(value))
        return values
    if isinstance(raw, (list, tuple, set)):
        values = []
        for value in raw:
            values.extend(_string_list(value))
        return values
    return [str(raw)]
