from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from securityclip.cli import main
from securityclip.config import FINAL_DATA_SOURCE_SPECS, SourceSpec
from securityclip.indexer import build_index
from securityclip.store import SecurityClipStore


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _fixture_sources(tmp_path: Path) -> list[SourceSpec]:
    data = tmp_path / "data"
    _write_parquet(
        data / "arxiv.parquet",
        [
            {
                "source_id": "arxiv",
                "source_record_id": "2401.00001",
                "record_id": "arxiv:2401.00001",
                "title": "JNDI Exploit Detection",
                "content": "# JNDI Exploit Detection\n\nLog4j Log4Shell uses JNDI lookup paths.\n\n## Methods\n\nWe evaluate detection.",
                "content_length": 42,
                "content_hash": "hash-arxiv",
                "license": "CC-BY-4.0",
                "source_url": "https://arxiv.org/abs/2401.00001",
                "arxiv_id": "2401.00001",
                "categories": ["cs.CR"],
                "primary_category": "cs.CR",
                "qwen_should_keep": True,
                "qwen_reason": "Security-relevant paper.",
            }
        ],
    )
    _write_parquet(
        data / "nvd.parquet",
        [
            {
                "source_id": "nvd",
                "source_record_id": "CVE-2021-44228",
                "record_id": "nvd:CVE-2021-44228",
                "title": None,
                "content": "Apache Log4j JNDI injection vulnerability allows remote code execution.",
                "content_length": 12,
                "content_hash": "hash-nvd",
                "license": "Public Domain",
                "source_url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
                "cve_id": "CVE-2021-44228",
                "severity": "critical",
                "cvss_score": 10.0,
            }
        ],
    )
    _write_parquet(
        data / "qa.parquet",
        [
            {
                "source_id": "stackoverflow",
                "source_record_id": "question-1",
                "record_id": "stackoverflow:question-1",
                "title": "How do I audit Log4j usage?",
                "content": "# How do I audit Log4j usage?\n\nSearch classpaths and block JNDI lookups.",
                "content_length": 16,
                "content_hash": "hash-qa",
                "license": "CC-BY-SA-4.0",
                "source_url": "https://stackoverflow.com/questions/1",
                "score": 10,
                "answer_count": 2,
                "tags": ["java", "security"],
                "qwen_should_keep": True,
                "qwen_reason": "Practical security audit thread.",
            }
        ],
    )
    _write_parquet(
        data / "sigma.parquet",
        [
            {
                "source_id": "sigma",
                "source_record_id": "rule-1",
                "record_id": "sigma:rule-1",
                "title": "Log4Shell JNDI Probe",
                "content": "Detects Log4j JNDI exploit probes.\n```yaml\ndetection:\n  selection: jndi\n```",
                "content_length": 20,
                "content_hash": "hash-sigma",
                "license": "LGPL-2.1",
                "source_url": "https://example.com/rule.yml",
                "rule_id": "rule-1",
                "rule_level": "high",
                "rule_source": "title: Log4Shell JNDI Probe\nlevel: high\n",
            }
        ],
    )
    _write_parquet(
        data / "cloudtrail.parquet",
        [
            {
                "source_id": "cloudtrail-flaws",
                "source_record_id": "1.2.3.4:20200101T000000Z",
                "record_id": "cloudtrail-flaws:1.2.3.4:20200101T000000Z",
                "content": "# CloudTrail Session\n\n## Events\n\n{\"eventName\":\"RunInstances\"}",
                "content_length": 18,
                "content_hash": "hash-cloudtrail",
                "license": "Public Domain (flaws.cloud)",
                "event_count": 1,
                "session_duration_seconds": 1,
                "source_ip": "1.2.3.4",
            }
        ],
    )
    _write_parquet(
        data / "fineweb.parquet",
        [
            {
                "source_id": "fineweb-security",
                "source_record_id": "web-doc-1",
                "record_id": "fineweb-security:web-doc-1",
                "title": "Browser Sandbox Escape Notes",
                "content": "A web document about browser sandbox escapes, exploit chains, and mitigations.",
                "content_length": 14,
                "content_hash": "hash-fineweb",
                "license": "FineWeb Terms",
                "source_url": "https://example.com/browser-sandbox",
                "language": "en",
                "dsir_score": 1.5,
            }
        ],
    )
    return [
        SourceSpec("arxiv", str(data / "arxiv.parquet")),
        SourceSpec("nvd", str(data / "nvd.parquet")),
        SourceSpec("qa", str(data / "qa.parquet")),
        SourceSpec("sigma", str(data / "sigma.parquet")),
        SourceSpec("cloudtrail", str(data / "cloudtrail.parquet")),
        SourceSpec("fineweb", str(data / "fineweb.parquet")),
    ]


def _build_fixture_index(tmp_path: Path) -> Path:
    index_dir = tmp_path / "index"
    summary = build_index(index_dir, source_specs=_fixture_sources(tmp_path), fts_max_chars=0)
    assert summary.documents == 6
    return index_dir


def test_build_index_and_virtual_files(tmp_path: Path):
    index_dir = _build_fixture_index(tmp_path)
    store = SecurityClipStore(index_dir)
    try:
        assert "arx_2401.00001/" in store.ls("/papers/")
        assert "fineweb-security/" in store.ls("/web/")
        assert "meta.json" in store.ls("/papers/arx_2401.00001/")
        assert "content.lines" in store.ls("/papers/arx_2401.00001/")
        assert "rule.yml" in store.ls("/rules/sigma/rule-1/")

        meta = store.cat("/papers/arx_2401.00001/meta.json")
        assert '"qwen_reason": "Security-relevant paper."' in meta
        assert '"content"' not in meta

        web_meta = store.cat("/web/fineweb-security/web-doc-1/meta.json")
        assert '"dsir_score": 1.5' in web_meta

        head = store.head(2, "/papers/arx_2401.00001/content.lines")
        assert head.startswith("L1: # JNDI Exploit Detection")
        assert "L3:" not in head
    finally:
        store.close()


def test_search_grep_and_handles(tmp_path: Path):
    index_dir = _build_fixture_index(tmp_path)
    store = SecurityClipStore(index_dir)
    try:
        handle, results = store.search("Log4j JNDI", limit=10)
        paths = [result.doc.content_path for result in results]
        assert handle.startswith("s_")
        assert "/vulns/nvd/CVE-2021-44228/content.lines" in paths
        assert "/rules/sigma/rule-1/content.lines" in paths

        _, cve_results = store.search("CVE-2021-44228", limit=5)
        assert "/vulns/nvd/CVE-2021-44228/content.lines" in [result.doc.content_path for result in cve_results]

        grep_handle, matches = store.grep("RunInstances", "/logs/cloudtrail-flaws/", limit=5)
        assert grep_handle and grep_handle.startswith("s_")
        assert matches[0].doc.content_path == "/logs/cloudtrail-flaws/1.2.3.4_20200101T000000Z/content.lines"

        _, scoped = store.grep("JNDI", from_handle=handle, ignore_case=True, limit=20)
        assert {match.doc.source_id for match in scoped} >= {"arxiv", "nvd", "sigma"}
    finally:
        store.close()


def test_literal_directory_grep_uses_fts_candidates(tmp_path: Path, monkeypatch):
    index_dir = _build_fixture_index(tmp_path)
    store = SecurityClipStore(index_dir)
    try:
        def fail_full_scope(*args, **kwargs):
            raise AssertionError("literal grep should not enumerate the whole directory scope")

        monkeypatch.setattr(store, "_docs_for_scope", fail_full_scope)
        _, matches = store.grep("RunInstances", "/logs/cloudtrail-flaws/", limit=5)

        assert matches[0].doc.content_path == "/logs/cloudtrail-flaws/1.2.3.4_20200101T000000Z/content.lines"
    finally:
        store.close()


def test_cli_stdout_is_shell_friendly(tmp_path: Path, capsys):
    index_dir = _build_fixture_index(tmp_path)
    assert main(["--index", str(index_dir), "search", "Log4j", "-n", "5"]) == 0
    out = capsys.readouterr().out
    assert "Search results: s_" in out
    assert "/vulns/nvd/CVE-2021-44228/content.lines" in out

    assert main(["--index", str(index_dir), "head", "-1", "/vulns/nvd/CVE-2021-44228/content.lines"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("L1: Apache Log4j")


def test_cli_json_output_modes(tmp_path: Path, capsys):
    index_dir = _build_fixture_index(tmp_path)

    assert main(["--index", str(index_dir), "search", "Log4j", "-n", "5", "--json"]) == 0
    search_payload = json.loads(capsys.readouterr().out)
    assert search_payload["handle"].startswith("s_")
    assert search_payload["results"][0]["path"].startswith("/")

    assert main(["--index", str(index_dir), "grep", "-i", "RunInstances", "/logs/cloudtrail-flaws/", "--limit", "5", "--json"]) == 0
    grep_payload = json.loads(capsys.readouterr().out)
    assert grep_payload["matches"][0]["citation"].endswith(":L5")

    assert main(["--index", str(index_dir), "ls", "/", "--json"]) == 0
    ls_payload = json.loads(capsys.readouterr().out)
    assert "papers/" in ls_payload["children"]

    assert main(["--index", str(index_dir), "head", "-1", "/vulns/nvd/CVE-2021-44228/content.lines", "--json"]) == 0
    head_payload = json.loads(capsys.readouterr().out)
    assert head_payload["lines"][0]["line_number"] == 1


def test_cli_defaults_to_securityclip_index_env(tmp_path: Path, monkeypatch, capsys):
    index_dir = _build_fixture_index(tmp_path)
    monkeypatch.setenv("SECURITYCLIP_INDEX", str(index_dir))

    assert main(["search", "Log4j", "-n", "5"]) == 0

    out = capsys.readouterr().out
    assert "Search results: s_" in out
    assert "/vulns/nvd/CVE-2021-44228/content.lines" in out


def test_cli_map_uses_configured_command(tmp_path: Path, monkeypatch, capsys):
    index_dir = _build_fixture_index(tmp_path)
    store = SecurityClipStore(index_dir)
    try:
        handle, _ = store.search("Log4j", limit=1)
    finally:
        store.close()

    monkeypatch.setenv("SECURITYCLIP_MAP_COMMAND", "python -c \"import sys; print('mapped:' + sys.stdin.readline().strip())\"")
    assert main(["--index", str(index_dir), "map", "--from", handle, "What methods were used?"]) == 0
    out = capsys.readouterr().out
    assert "mapped:Path:" in out


def test_final_data_source_specs_match_marlowe_layout():
    patterns = {spec.name: spec.pattern for spec in FINAL_DATA_SOURCE_SPECS}

    assert patterns["arxiv-cs-cr-full"] == "final-data/academic_papers/arxiv_cs_cr_full.parquet"
    assert patterns["arxiv-citation-qwen-kept-full"] == "final-data/academic_papers/arxiv_citation_qwen_kept_full.parquet"
    assert patterns["sigma"] == "final-data/normalized/source_id=sigma/*.parquet"
    assert patterns["cloudtrail-flaws"] == "final-data/normalized/source_id=cloudtrail-flaws/*.parquet"
    assert patterns["fineweb-security"] == "final-data/normalized/source_id=fineweb-security/*.parquet"
    assert patterns["stackoverflow"] == "final-data/qwen_qa_kept/source_id=stackoverflow/*.parquet"
