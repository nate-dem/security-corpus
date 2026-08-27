"""FastAPI app for the Security Scope web UI."""

from __future__ import annotations

import argparse
import json
import queue
import threading
from pathlib import Path
from typing import Any, Iterator, Sequence

from securityclip.store import SecurityClipStore
from securityclip.web.history import RunHistory
from securityclip.web.service import QueryService, normalize_requested_roots
from securityclip.web.settings import load_settings


def create_app():
    try:
        from fastapi import Body, FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError
    except ImportError as exc:
        raise RuntimeError("security-scope-web requires `pip install -e '.[web]'`") from exc

    settings = load_settings()
    service = QueryService(settings)
    history = RunHistory(settings.web_db)
    app = FastAPI(title="Security Scope", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8765", "http://localhost:8765"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class QueryRequest(BaseModel):
        model_config = ConfigDict(populate_by_name=True)

        query: str = Field(validation_alias=AliasChoices("query", "q", "prompt", "question"), min_length=1)
        max_steps: int | None = Field(default=None, validation_alias=AliasChoices("max_steps", "maxSteps"))
        max_results: int | None = Field(default=None, validation_alias=AliasChoices("max_results", "maxResults"))
        roots: list[str] | None = Field(default=None, validation_alias=AliasChoices("roots", "sourceRoots"))

    def _parse_request(payload: dict) -> tuple[QueryRequest, list[str] | None]:
        try:
            request = QueryRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        try:
            roots = normalize_requested_roots(request.roots)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return request, roots

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": settings.index_dir.joinpath("securityclip.sqlite").exists(),
            "index_dir": str(settings.index_dir),
            "openai_api_key_present": settings.openai_api_key_present,
            "router_model": settings.router_model,
            "planner_model": settings.planner_model,
            "synthesis_model": settings.synthesis_model,
        }

    @app.post("/api/query")
    def query_corpus(payload: dict = Body(...)) -> dict:
        request, roots = _parse_request(payload)
        return service.run_query(request.query, max_steps=request.max_steps, max_results=request.max_results, roots=roots)

    @app.post("/api/query/stream")
    def query_corpus_stream(payload: dict = Body(...)) -> StreamingResponse:
        request, roots = _parse_request(payload)
        return StreamingResponse(
            _stream_query_events(service, request.query, max_steps=request.max_steps, max_results=request.max_results, roots=roots),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=200)) -> dict:
        return {"runs": history.list_runs(limit=limit)}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = history.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.get("/api/doc")
    def get_doc(path: str, count: int = Query(default=100, ge=1, le=settings.max_head_count)) -> dict:
        try:
            store = SecurityClipStore(settings.index_dir)
            try:
                if path.endswith("/content.lines"):
                    text = store.head(count, path)
                else:
                    text = store.cat(path)
            finally:
                store.close()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"path": path, "text": text}

    dist_dir = _frontend_dist_dir()
    if dist_dir.joinpath("index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")

        @app.get("/{path:path}", response_class=HTMLResponse)
        def spa(path: str = ""):
            return dist_dir.joinpath("index.html").read_text(encoding="utf-8")

    else:
        @app.get("/", response_class=HTMLResponse)
        def missing_frontend():
            return """
            <html><body>
            <h1>Security Scope API</h1>
            <p>The React frontend has not been built yet. Run <code>npm install && npm run build</code> in <code>web/</code>.</p>
            <p>API health: <a href="/api/health">/api/health</a></p>
            </body></html>
            """

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="security-scope-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("security-scope-web requires `pip install -e '.[web]'`") from exc
    uvicorn.run("securityclip.web.app:create_app", host=args.host, port=args.port, factory=True)
    return 0


def _frontend_dist_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def _sse_event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _stream_query_events(
    service: QueryService,
    query: str,
    *,
    max_steps: int | None,
    max_results: int | None,
    roots: list[str] | None,
    keep_alive_seconds: float = 15.0,
) -> Iterator[str]:
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            result = service.run_query(
                query,
                max_steps=max_steps,
                max_results=max_results,
                roots=roots,
                on_event=lambda event: events.put(("progress", event)),
            )
            events.put(("result", result))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an error event
            events.put(("error", {"message": str(exc)}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            item = events.get(timeout=keep_alive_seconds)
        except queue.Empty:
            yield ": keep-alive\n\n"
            continue
        if item is None:
            break
        name, data = item
        yield _sse_event(name, data)
