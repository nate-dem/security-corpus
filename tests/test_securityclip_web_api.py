from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from conftest import build_securityclip_fixture_index
from securityclip.web.app import create_app


def test_web_health_reports_missing_index(monkeypatch, tmp_path):
    monkeypatch.setenv("SECURITYCLIP_INDEX", str(tmp_path / "missing"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(create_app())
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["ok"] is False


def test_doc_endpoint_requires_existing_index(monkeypatch, tmp_path):
    monkeypatch.setenv("SECURITYCLIP_INDEX", str(tmp_path / "missing"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(create_app())
    response = client.get("/api/doc", params={"path": "/papers/example/content.lines"})

    assert response.status_code == 404


def test_query_endpoint_accepts_camel_case_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("SECURITYCLIP_INDEX", str(tmp_path / "missing"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(create_app())
    response = client.post(
        "/api/query",
        json={"question": "Find CVE-2021-44228", "maxSteps": 2, "maxResults": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Find CVE-2021-44228"
    assert body["caps"] == {"max_steps": 2, "max_results": 5}


def test_query_endpoint_rejects_unknown_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SECURITYCLIP_INDEX", str(tmp_path / "missing"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(create_app())
    response = client.post("/api/query", json={"query": "find papers", "roots": ["/etc"]})

    assert response.status_code == 400
    assert "unknown corpus roots" in response.json()["detail"]


def test_query_endpoint_accepts_roots_camel_case(monkeypatch, tmp_path):
    monkeypatch.setenv("SECURITYCLIP_INDEX", str(tmp_path / "missing"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(create_app())
    response = client.post(
        "/api/query",
        json={"question": "find papers about fuzzing", "sourceRoots": ["papers"]},
    )

    assert response.status_code == 200
    assert response.json()["requested_roots"] == ["/papers"]


def test_query_stream_endpoint_emits_sse_frames(monkeypatch, tmp_path):
    index_dir = build_securityclip_fixture_index(tmp_path)
    monkeypatch.setenv("SECURITYCLIP_INDEX", str(index_dir))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(create_app())
    with client.stream(
        "POST", "/api/query/stream", json={"query": "Give me sources that include CVE-2021-44228"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: progress" in body
    assert body.count("event: result") == 1
    data_lines = [line for line in body.splitlines() if line.startswith("data: ")]
    final = json.loads(data_lines[-1][len("data: "):])
    assert final["answer_markdown"]
    assert final["status"] == "done"
