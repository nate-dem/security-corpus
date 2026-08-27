from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from securityclip.store import SecurityClipStore
from securityclip.web.executor import OperationExecutor, OperationOutput
from securityclip.web.history import RunHistory
from securityclip.web.operations import OperationValidationError, validate_operations
from securityclip.web.routing import deterministic_route
from securityclip.web.service import QueryService, group_by_root
from securityclip.web.settings import WebSettings, load_settings


def _settings(tmp_path: Path, index_dir: Path, *, key_present: bool = False, timeout: float = 5) -> WebSettings:
    return WebSettings(
        index_dir=index_dir,
        web_db=tmp_path / "history.sqlite",
        router_model="gpt-5-nano",
        planner_model="gpt-5-mini",
        synthesis_model="gpt-5-mini",
        max_commands=8,
        max_results=20,
        max_limit=50,
        max_head_count=100,
        command_timeout_seconds=timeout,
        max_output_chars=10_000,
        openai_api_key_present=key_present,
    )


def test_deterministic_routing_for_cve_arxiv_path_and_root():
    cve = deterministic_route("Give me sources that include CVE-2021-44228")
    assert cve and cve.route_type == "source_list"
    assert cve.entities["cve_ids"] == ["CVE-2021-44228"]

    arxiv = deterministic_route("Read arXiv 2401.00001")
    assert arxiv and arxiv.route_type == "exact_identifier"
    assert arxiv.likely_roots == ["/papers"]

    path = deterministic_route("Open /vulns/nvd/CVE-2021-44228/meta.json")
    assert path and path.route_type == "path_inspection"

    root = deterministic_route("Find Sigma rules for Log4Shell")
    assert root and "/rules" in root.likely_roots


def test_operation_validation_rejects_shell_and_bad_paths(tmp_path: Path):
    settings = _settings(tmp_path, tmp_path / "missing")
    with pytest.raises(OperationValidationError):
        validate_operations(["security-scope search CVE-2021-44228"], settings)
    with pytest.raises(OperationValidationError):
        validate_operations([{"tool": "grep", "pattern": "x", "path": "/etc/passwd"}], settings)
    with pytest.raises(OperationValidationError):
        validate_operations([{"tool": "cat", "path": "/papers/arx_1/content.lines"}], settings)
    with pytest.raises(OperationValidationError):
        validate_operations([{"tool": "search", "query": "x; rm -rf /"}], settings)


def test_operation_validation_clamps_limits(tmp_path: Path):
    settings = _settings(tmp_path, tmp_path / "missing")
    operations, warnings = validate_operations([{"tool": "search", "query": "Log4j", "limit": 999}], settings)
    assert operations[0].params["limit"] == settings.max_limit
    assert warnings


def test_validate_operations_rejects_path_outside_selected_roots(tmp_path: Path):
    settings = _settings(tmp_path, tmp_path / "missing")
    with pytest.raises(OperationValidationError, match="outside selected roots"):
        validate_operations([{"tool": "ls", "path": "/vulns/nvd", "limit": 5}], settings, allowed_roots=("/papers",))
    operations, _ = validate_operations([{"tool": "ls", "path": "/papers", "limit": 5}], settings, allowed_roots=("/papers",))
    assert operations[0].params["path"] == "/papers"


def test_executor_uses_securityclip_store_directly(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index)
    operations, _ = validate_operations(
        [
            {"tool": "search", "query": "CVE-2021-44228", "limit": 5},
            {"tool": "grep", "pattern": "prompt injection", "path": "/papers/", "ignore_case": True, "limit": 5},
        ],
        settings,
    )
    outputs = OperationExecutor(securityclip_fixture_index, settings).execute_many(operations)
    assert outputs[0].ok
    assert outputs[0].data["handle"].startswith("s_")
    assert outputs[1].citations[0]["citation"].startswith("/papers/arx_2401.00001/content.lines:L")


def _multi_root_index(tmp_path: Path, *, n_papers: int, n_vulns: int) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from securityclip.config import SourceSpec
    from securityclip.indexer import build_index

    rows: list[dict[str, Any]] = []
    for idx in range(n_papers):
        rows.append(
            {
                "source_id": "arxiv",
                "source_record_id": f"24{idx:02d}.00001",
                "record_id": f"arxiv:24{idx:02d}.00001",
                "title": f"Paper {idx} on prompt injection",
                "content": f"Paper {idx} studies prompt injection and adversarial retrieval.",
                "content_length": 20,
                "content_hash": f"hp{idx}",
                "license": "CC-BY-4.0",
                "arxiv_id": f"24{idx:02d}.00001",
            }
        )
    for idx in range(n_vulns):
        rows.append(
            {
                "source_id": "nvd",
                "source_record_id": f"CVE-2021-{1000 + idx}",
                "record_id": f"nvd:CVE-2021-{1000 + idx}",
                "title": f"CVE-2021-{1000 + idx} prompt injection",
                "content": f"Vulnerability {idx} involving prompt injection.",
                "content_length": 15,
                "content_hash": f"hv{idx}",
                "license": "Public Domain",
                "source_url": "https://example.test",
                "cve_id": f"CVE-2021-{1000 + idx}",
            }
        )
    parquet = tmp_path / "data" / "docs.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    index_dir = tmp_path / "index"
    build_index(index_dir, source_specs=[SourceSpec("fixture", str(parquet))], fts_max_chars=0)
    return index_dir


def test_search_per_root_limit_caps_each_document_type(tmp_path: Path):
    store = SecurityClipStore(_multi_root_index(tmp_path, n_papers=3, n_vulns=2))
    try:
        # Global limit (CLI behavior) returns everything ranked together.
        _, global_results = store.search("prompt injection", limit=10)
        assert len(global_results) == 5
        # Per-root limit caps each document type independently.
        _, per_root = store.search("prompt injection", per_root_limit=2)
        roots = [result.doc.root for result in per_root]
        assert roots.count("/papers") == 2  # capped from 3
        assert roots.count("/vulns") == 2
        # A roots filter restricts which types are queried, still per-root capped.
        _, papers = store.search("prompt injection", roots=["/papers"], per_root_limit=2)
        assert [result.doc.root for result in papers] == ["/papers", "/papers"]
    finally:
        store.close()


def test_run_query_caps_results_per_document_type(tmp_path: Path):
    index_dir = _multi_root_index(tmp_path, n_papers=3, n_vulns=2)
    settings = replace(_settings(tmp_path, index_dir, key_present=False), max_results=2)
    result = QueryService(settings, model_client=FakeModelClient()).run_query("prompt injection")
    counts: dict[str, int] = {}
    for source in result["sources"]:
        root = "/" + source["path"].split("/")[1]
        counts[root] = counts.get(root, 0) + 1
    assert counts.get("/papers") == 2  # capped at max_results per type
    assert counts.get("/vulns") == 2


def test_search_with_roots_filters_results(securityclip_fixture_index: Path):
    store = SecurityClipStore(securityclip_fixture_index)
    try:
        _, all_results = store.search("injection", limit=10)
        assert {result.doc.root for result in all_results} == {"/papers", "/vulns"}
        _, papers_only = store.search("injection", limit=10, roots=["/papers"])
        assert {result.doc.root for result in papers_only} == {"/papers"}
    finally:
        store.close()


class FakeModelClient:
    def __init__(self, json_responses: list[Any] | None = None, text_response: str = "answer"):
        self.json_responses = list(json_responses or [])
        self.text_response = text_response

    def generate_json(self, *, model: str, system: str, user: str) -> Any:
        if not self.json_responses:
            raise ValueError("no fake JSON response")
        value = self.json_responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        return self.text_response


class RecordingModelClient(FakeModelClient):
    def __init__(self, json_responses: list[Any] | None = None, text_response: str = "answer"):
        super().__init__(json_responses, text_response)
        self.calls: list[tuple[str, str, str]] = []

    def generate_json(self, *, model: str, system: str, user: str) -> Any:
        self.calls.append((model, system, user))
        return super().generate_json(model=model, system=system, user=user)

    def generate_text(self, *, model: str, system: str, user: str) -> str:
        self.calls.append((model, system, user))
        return super().generate_text(model=model, system=system, user=user)


def test_query_service_uses_deterministic_cve_without_openai(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=False)
    result = QueryService(settings, model_client=FakeModelClient()).run_query("Give me sources that include CVE-2021-44228")
    assert result["operations"][0]["tool"] == "search"
    assert result["operation_outputs"][0]["data"]["results"]
    assert result["answer_markdown"]


def test_query_service_repairs_invalid_planner_output(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=True)
    client = FakeModelClient(
        json_responses=[
            {"route_type": "topic_research", "entities": {}, "likely_roots": [], "confidence": 0.5},
            [{"tool": "cat", "path": "/papers/arx_2401.00001/content.lines"}],
            [{"tool": "head", "path": "/papers/arx_2401.00001/content.lines", "count": 10}],
        ],
        text_response="Synthesized answer",
    )
    result = QueryService(settings, model_client=client).run_query("Summarize prompt injection research")
    assert result["operations"][0]["tool"] == "head"
    assert result["answer_markdown"] == "Synthesized answer"


def test_repair_debug_recorded(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=True)
    client = FakeModelClient(
        json_responses=[
            {"route_type": "topic_research", "entities": {}, "likely_roots": [], "confidence": 0.5},
            [{"tool": "cat", "path": "/papers/arx_2401.00001/content.lines"}],
            [{"tool": "head", "path": "/papers/arx_2401.00001/content.lines", "count": 10}],
        ],
        text_response="Synthesized answer",
    )
    service = QueryService(settings, model_client=client)
    result = service.run_query("Summarize prompt injection research")
    repair = result["planner_repair"]
    assert repair["repair_succeeded"] is True
    assert "cat on content.lines" in repair["validation_error"]
    assert repair["raw_operations"][0]["tool"] == "cat"
    assert repair["repaired_operations"][0]["tool"] == "head"
    assert service.history.get(result["run_id"])["planner_repair"]["repair_succeeded"] is True


def test_query_service_normalizes_loose_router_entities(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=True)
    client = FakeModelClient(
        json_responses=[
            {"route_type": "topic_research", "entities": ["prompt injection"], "likely_roots": "papers", "confidence": "0.72"},
            [{"tool": "search", "query": "prompt injection", "limit": 5}],
        ],
        text_response="Synthesized answer",
    )

    result = QueryService(settings, model_client=client).run_query("Find all documents related to prompt injection", max_results=5)

    assert result["route"]["entities"] == {"terms": ["prompt injection"]}
    assert result["route"]["likely_roots"] == ["/papers"]
    assert result["operations"][0]["tool"] == "search"
    assert result["answer_markdown"] == "Synthesized answer"


def test_missing_openai_key_for_ambiguous_query_uses_fallback_with_error(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=False)
    result = QueryService(settings, model_client=FakeModelClient()).run_query("Tell me about unusual retrieval behavior")
    assert result["operations"][0]["tool"] == "search"
    assert any("OPENAI_API_KEY" in error for error in result["errors"])


def test_error_details_for_missing_key(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=False)
    result = QueryService(settings, model_client=FakeModelClient()).run_query("Tell me about unusual retrieval behavior")
    detail = next(detail for detail in result["error_details"] if detail["code"] == "missing_api_key")
    assert "OPENAI_API_KEY" in detail["message"]
    assert "OPENAI_API_KEY" in detail["hint"]
    assert detail["message"] in result["errors"]


def test_error_details_classify_timeout(monkeypatch, securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, timeout=0.01)

    def slow_execute(self, operation, command):
        time.sleep(0.1)
        raise AssertionError("should time out")

    monkeypatch.setattr(OperationExecutor, "_execute_unbounded", slow_execute)
    result = QueryService(settings, model_client=FakeModelClient()).run_query("Give me sources that include CVE-2021-44228")
    assert any(detail["code"] == "operation_timeout" for detail in result["error_details"])
    assert result["status"] == "done_with_errors"


def test_no_results_detail_does_not_flip_status(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=False)
    result = QueryService(settings, model_client=FakeModelClient()).run_query("Give me sources that include CVE-2099-99999")
    assert any(detail["code"] == "no_results" for detail in result["error_details"])
    assert result["status"] == "done"
    assert result["errors"] == []


def test_model_json_parse_failure_falls_back(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=True)
    result = QueryService(settings, model_client=FakeModelClient(json_responses=[ValueError("bad json")])).run_query("Tell me about retrieval")
    assert result["operations"][0]["tool"] == "search"
    assert any("router failed" in error for error in result["errors"])


def test_missing_index_surfaces_operation_error(tmp_path: Path):
    settings = _settings(tmp_path, tmp_path / "missing-index")
    operations, _ = validate_operations([{"tool": "search", "query": "Log4j", "limit": 5}], settings)
    output = OperationExecutor(settings.index_dir, settings).execute(operations[0])
    assert not output.ok
    assert "Security Scope index not found" in (output.error or "")


def test_operation_timeout(monkeypatch, securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, timeout=0.01)
    operations, _ = validate_operations([{"tool": "search", "query": "Log4j", "limit": 5}], settings)

    def slow_execute(self, operation, command):
        time.sleep(0.1)
        raise AssertionError("should time out")

    monkeypatch.setattr(OperationExecutor, "_execute_unbounded", slow_execute)
    output = OperationExecutor(securityclip_fixture_index, settings).execute(operations[0])
    assert not output.ok
    assert "timed out" in (output.error or "")


def test_synthesis_system_includes_route_template(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=True)
    client = RecordingModelClient(text_response="Report")
    QueryService(settings, model_client=client).run_query("Give me sources that include CVE-2021-44228")
    _, system, _ = client.calls[-1]
    assert "Where It Appears" in system
    assert "Community Q&A" in system
    assert "Do not end with a question" in system
    assert "Never invent citations" in system
    assert "Suggested follow-up commands" in system


def test_synthesis_evidence_grouped_by_root(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=True)
    client = RecordingModelClient(text_response="Report")
    result = QueryService(settings, model_client=client).run_query("Give me sources that include CVE-2021-44228")
    _, _, user = client.calls[-1]
    evidence = json.loads(user)
    assert "citations_by_root" in evidence
    assert "/vulns" in evidence["sources_by_root"]
    assert evidence["counts_by_root"]["sources"]["/vulns"] >= 1
    assert "citations" not in evidence
    # Outputs are compact: no full data arrays, only a bounded preview + counts.
    assert set(evidence["outputs"][0].keys()) == {
        "command",
        "ok",
        "error",
        "latency_ms",
        "truncated",
        "source_count",
        "citation_count",
        "output_preview",
    }
    assert isinstance(result["citations"], list)
    assert result["sources"][0]["path"].startswith("/vulns/")


def _synthetic_output(*, text_len: int, n_sources: int, n_citations: int, root: str = "/vulns") -> OperationOutput:
    sources = [
        {
            "path": f"{root}/nvd/CVE-{idx}/content.lines",
            "meta_path": f"{root}/nvd/CVE-{idx}/meta.json",
            "virtual_dir": f"{root}/nvd/CVE-{idx}",
            "source_id": "nvd",
            "title": f"Record {idx}",
            "record_id": f"nvd:CVE-{idx}",
            "source_url": "https://example.test",
            "license": "Public Domain",
            "tokens": 1234,
        }
        for idx in range(n_sources)
    ]
    citations = [
        {
            "path": f"{root}/nvd/CVE-{idx}/content.lines",
            "line_number": 1,
            "citation": f"{root}/nvd/CVE-{idx}/content.lines:L1",
            "snippet": "x" * 400,
            "source_id": "nvd",
            "title": f"Record {idx}",
        }
        for idx in range(n_citations)
    ]
    return OperationOutput(
        operation={"tool": "search", "query": "x", "limit": 50},
        command="security-scope search x -n 50",
        ok=True,
        output_text="A" * text_len,
        data={"handle": "s_abc12345", "results": [{"rank": 1}]},
        citations=citations,
        sources=sources,
    )


def test_build_synthesis_evidence_is_compact(tmp_path: Path):
    settings = _settings(tmp_path, tmp_path / "missing", key_present=True)
    service = QueryService(settings, model_client=FakeModelClient())
    output = _synthetic_output(text_len=20_000, n_sources=30, n_citations=40)
    evidence, serialized = service._build_synthesis_evidence(
        "broad query", {"route_type": "general"}, [], [output], output.citations, output.sources, 50
    )
    # The full raw output_text is never placed in the model request.
    assert "A" * 20_000 not in serialized
    compact = evidence["outputs"][0]
    assert compact["source_count"] == 30
    assert compact["citation_count"] == 40
    assert len(compact["output_preview"]) <= settings.synthesis_max_output_chars_per_operation + 1
    # Per-root caps applied to what the model sees.
    assert len(evidence["citations_by_root"]["/vulns"]) == settings.synthesis_max_citations_per_root
    assert len(evidence["sources_by_root"]["/vulns"]) == settings.synthesis_max_sources_per_root
    # Exact citation strings are preserved for citation quality.
    assert evidence["citations_by_root"]["/vulns"][0]["citation"].startswith("/vulns/nvd/CVE-")
    # Whole payload stays under the configured total budget.
    assert len(serialized) <= settings.synthesis_max_total_chars


def test_build_synthesis_evidence_respects_total_cap(tmp_path: Path):
    settings = replace(_settings(tmp_path, tmp_path / "missing", key_present=True), synthesis_max_total_chars=6000)
    service = QueryService(settings, model_client=FakeModelClient())
    outputs = [_synthetic_output(text_len=40_000, n_sources=50, n_citations=60) for _ in range(4)]
    citations = [citation for output in outputs for citation in output.citations]
    sources = [source for output in outputs for source in output.sources]
    evidence, serialized = service._build_synthesis_evidence(
        "broad query", {"route_type": "general"}, [], outputs, citations, sources, 50
    )
    assert len(serialized) <= 6000
    assert evidence.get("evidence_truncated_for_synthesis") is True


def test_synthesis_failure_falls_back_to_deterministic_answer(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=True)

    class _SynthFailClient(FakeModelClient):
        def generate_text(self, *, model: str, system: str, user: str) -> str:
            raise RuntimeError("413 Request Entity Too Large")

    client = _SynthFailClient(
        json_responses=[
            {"route_type": "topic_research", "entities": {}, "likely_roots": [], "confidence": 0.5},
            [{"tool": "search", "query": "prompt injection", "limit": 5}],
        ]
    )
    result = QueryService(settings, model_client=client).run_query("Summarize prompt injection research")
    assert "## Summary" in result["answer_markdown"]
    assert any(detail["code"] == "synthesis_failed" for detail in result["error_details"])


def test_load_settings_defaults_to_50_results(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SECURITYCLIP_INDEX", str(tmp_path / "idx"))
    for var in (
        "SECURITYCLIP_MAX_RESULTS",
        "SECURITYCLIP_SYNTHESIS_MAX_SOURCES_PER_ROOT",
        "SECURITYCLIP_SYNTHESIS_MAX_CITATIONS_PER_ROOT",
        "SECURITYCLIP_SYNTHESIS_MAX_OUTPUT_CHARS_PER_OPERATION",
        "SECURITYCLIP_SYNTHESIS_MAX_TOTAL_CHARS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = load_settings()
    assert settings.max_results == 50
    assert settings.synthesis_max_sources_per_root == 8
    assert settings.synthesis_max_citations_per_root == 12
    assert settings.synthesis_max_output_chars_per_operation == 2000
    assert settings.synthesis_max_total_chars == 50_000


def test_group_by_root_caps_and_buckets():
    items = [{"path": f"/papers/doc{idx}/content.lines"} for idx in range(20)]
    items.append({"path": "/vulns/nvd/CVE-1/content.lines"})
    items.append({"path": "weird"})
    grouped = group_by_root(items, per_root_cap=15)
    assert len(grouped["/papers"]) == 15
    assert len(grouped["/vulns"]) == 1
    assert grouped["other"] == [{"path": "weird"}]


def test_run_query_with_roots_constrains_outputs(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=False)
    result = QueryService(settings, model_client=FakeModelClient()).run_query(
        "Give me sources that include CVE-2021-44228", roots=["/papers"]
    )
    assert result["requested_roots"] == ["/papers"]
    assert result["route"]["likely_roots"] == ["/papers"]
    assert all(source["path"].startswith("/papers/") for source in result["sources"])


def test_run_query_emits_ordered_events(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=False)
    events: list[dict] = []
    QueryService(settings, model_client=FakeModelClient()).run_query(
        "Give me sources that include CVE-2021-44228", on_event=events.append
    )
    stages = [event["stage"] for event in events]
    assert stages == ["routing", "planning", "validating", "executing", "synthesizing"]
    executing = next(event for event in events if event["stage"] == "executing")
    assert executing["operation_index"] == 1
    assert executing["operation_count"] == 1
    assert executing["command"].startswith("security-scope search")


def test_run_query_event_callback_errors_are_swallowed(securityclip_fixture_index: Path, tmp_path: Path):
    settings = _settings(tmp_path, securityclip_fixture_index, key_present=False)

    def boom(event: dict) -> None:
        raise RuntimeError("callback failure")

    result = QueryService(settings, model_client=FakeModelClient()).run_query(
        "Give me sources that include CVE-2021-44228", on_event=boom
    )
    assert result["answer_markdown"]


def test_history_migration_adds_columns(tmp_path: Path):
    db = tmp_path / "history.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE web_runs (
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
            latency_ms INTEGER NOT NULL
        )
        """
    )
    con.execute(
        "INSERT INTO web_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("run_old", "2026-01-01T00:00:00+00:00", "old query", "{}", "r", "p", "s", "[]", "[]", "[]", "old answer", "[]", "[]", 5),
    )
    con.commit()
    con.close()

    history = RunHistory(db)
    history.save(
        {
            "run_id": "run_new",
            "query": "new query",
            "answer_markdown": "new answer",
            "planner_repair": {"repair_succeeded": True},
            "error_details": [{"code": "no_results", "message": "m", "hint": "h"}],
        }
    )
    loaded = history.get("run_new")
    assert loaded["planner_repair"] == {"repair_succeeded": True}
    assert loaded["error_details"][0]["code"] == "no_results"
    old = history.get("run_old")
    assert old["planner_repair"] is None
    assert old["error_details"] == []
    assert old["answer_markdown"] == "old answer"
