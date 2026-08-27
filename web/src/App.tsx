import { FormEvent, useEffect, useState } from "react";

import DocGroups from "./components/DocGroups";
import DocPanel from "./components/DocPanel";
import ErrorCards from "./components/ErrorCards";
import HistoryDrawer from "./components/HistoryDrawer";
import Markdown from "./components/Markdown";
import Progress from "./components/Progress";
import { readSSE } from "./lib/sse";
import {
  ALL_ROOTS,
  DocPreview,
  ProgressEvent,
  ROOT_LABELS,
  RunResponse,
  RunStatus,
  RunSummary
} from "./lib/types";

const suggestions = [
  { query: "Give me a list of sources that include CVE-2021-44228.", topic: "Cross-source lookup" },
  { query: "Find all documents related to prompt injection.", topic: "Topic research" },
  { query: "Find papers about adversarial robustness and biosecurity.", topic: "Academic papers" },
  { query: "Find web documents about browser sandbox escapes.", topic: "Web corpus" },
  { query: "Find Sigma rules related to Log4Shell.", topic: "Detection rules" },
  { query: "Show NVD records involving prompt injection.", topic: "Vulnerabilities" },
  { query: "Find CloudTrail examples involving RunInstances privilege escalation.", topic: "Logs" }
];

const MAX_RESULTS = 50;

export default function App() {
  const [query, setQuery] = useState("");
  const [run, setRun] = useState<RunResponse | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [progressEvent, setProgressEvent] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<DocPreview | null>(null);
  const [selectedRoots, setSelectedRoots] = useState<string[]>([...ALL_ROOTS]);

  const showHero = !run && status === "idle" && !error;

  useEffect(() => {
    void loadHistory();
  }, []);

  async function loadHistory() {
    try {
      const response = await fetch("/api/runs?limit=30");
      if (response.ok) {
        const payload = await response.json();
        setHistory(payload.runs ?? []);
      }
    } catch {
      // history is best-effort; the query flow surfaces connectivity errors
    }
  }

  function toggleRoot(root: string) {
    setSelectedRoots((current) =>
      current.includes(root) ? current.filter((item) => item !== root) : [...current, root]
    );
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void submitQuery(query);
  }

  async function submitQuery(text: string) {
    const trimmed = text.trim();
    if (!trimmed || status === "running" || selectedRoots.length === 0) {
      return;
    }
    setQuery(trimmed);
    const payload: Record<string, unknown> = { query: trimmed, max_steps: 8, max_results: MAX_RESULTS };
    if (selectedRoots.length > 0 && selectedRoots.length < ALL_ROOTS.length) {
      payload.roots = selectedRoots;
    }
    const body = JSON.stringify(payload);
    setStatus("running");
    setError(null);
    setRun(null);
    setSelectedDoc(null);
    setProgressEvent({ stage: "routing" });
    try {
      const result = (await runStreaming(body)) ?? (await runBlocking(body));
      setRun(result);
      setStatus("done");
      await loadHistory();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setStatus("error");
    }
  }

  /** Stream progress over SSE; returns null when the endpoint is unavailable so the caller falls back. */
  async function runStreaming(body: string): Promise<RunResponse | null> {
    let response: Response;
    try {
      response = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body
      });
    } catch {
      return null;
    }
    if (response.status === 400) {
      throw new Error(await response.text());
    }
    if (!response.ok || !response.body) {
      return null;
    }
    let result: RunResponse | null = null;
    for await (const message of readSSE(response.body)) {
      if (message.event === "progress") {
        setProgressEvent(message.data as ProgressEvent);
      } else if (message.event === "result") {
        result = message.data as RunResponse;
      } else if (message.event === "error") {
        throw new Error((message.data as { message?: string }).message ?? "query failed");
      }
    }
    return result;
  }

  async function runBlocking(body: string): Promise<RunResponse> {
    let response: Response;
    try {
      response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body
      });
    } catch (exc) {
      throw new Error(`Request failed — is Security Scope running? (${exc instanceof Error ? exc.message : exc})`);
    }
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return (await response.json()) as RunResponse;
  }

  function goHome() {
    setRun(null);
    setError(null);
    setStatus("idle");
    setProgressEvent(null);
    setSelectedDoc(null);
    setHistoryOpen(false);
    setQuery("");
  }

  async function openDoc(path: string) {
    const response = await fetch(`/api/doc?path=${encodeURIComponent(path)}&count=100`);
    if (response.ok) {
      setSelectedDoc((await response.json()) as DocPreview);
    }
  }

  async function openRun(runId: string) {
    const response = await fetch(`/api/runs/${runId}`);
    if (response.ok) {
      setRun((await response.json()) as RunResponse);
      setStatus("idle");
      setError(null);
      setSelectedDoc(null);
      setHistoryOpen(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <button type="button" className="brand" onClick={goHome} aria-label="Back to home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="32" height="32" rx="9" fill="url(#ss-grad)" />
              <circle cx="16" cy="16" r="7.5" stroke="#ffffff" strokeWidth="2" opacity="0.95" />
              <path
                d="M16 3.5v5M16 23.5v5M3.5 16h5M23.5 16h5"
                stroke="#ffffff"
                strokeWidth="2"
                strokeLinecap="round"
              />
              <circle cx="16" cy="16" r="2.4" fill="#ffffff" />
              <defs>
                <linearGradient id="ss-grad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#2f6bff" />
                  <stop offset="1" stopColor="#6d5efc" />
                </linearGradient>
              </defs>
            </svg>
          </span>
          <span className="brand-name">Security Scope</span>
        </button>
        <button type="button" className="ghost-button" onClick={() => setHistoryOpen(true)}>
          History
        </button>
      </header>

      <main className={`page${showHero ? " page-hero" : ""}`}>
        {showHero && (
          <div className="hero">
            <h2>Find evidence for any security question.</h2>
            <p>
              Agentic retrieval across academic papers, vulnerabilities, detection rules, logs, web documents, video
              transcripts, and community knowledge.
            </p>
          </div>
        )}

        <form className="chatbox" onSubmit={submit}>
          <textarea
            aria-label="Ask Security Scope"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submitQuery(query);
              }
            }}
            rows={showHero ? 3 : 2}
            placeholder="Ask about CVEs, papers, detection rules, logs…"
          />
          <div className="chatbox-footer">
            <div className="chatbox-filters">
              <p className={`filters-label${selectedRoots.length === 0 ? " filters-label-warn" : ""}`}>
                {selectedRoots.length === 0 ? "Select at least one source to run" : "Specify the sources you want"}
              </p>
              <div className="root-filters" role="group" aria-label="Source type filters">
                {ALL_ROOTS.map((root) => (
                  <button
                    type="button"
                    key={root}
                    className={`root-chip${selectedRoots.includes(root) ? " selected" : ""}`}
                    aria-pressed={selectedRoots.includes(root)}
                    onClick={() => toggleRoot(root)}
                  >
                    {ROOT_LABELS[root] ?? root}
                  </button>
                ))}
              </div>
            </div>
            <button
              className="run-button"
              type="submit"
              disabled={status === "running" || !query.trim() || selectedRoots.length === 0}
            >
              {selectedRoots.length === 0 ? "" : status === "running" ? "Running…" : "Run query"}
            </button>
          </div>
        </form>

        {showHero && (
          <div className="suggestions-block">
            <p className="suggestions-label">Or begin with one of these</p>
            <ol className="suggestions">
              {suggestions.map((item, idx) => (
                <li key={item.query}>
                  <button type="button" className="suggestion" onClick={() => void submitQuery(item.query)}>
                    <span className="suggestion-index">{String(idx + 1).padStart(2, "0")}</span>
                    <span className="suggestion-query">{item.query}</span>
                    <span className="suggestion-topic">{item.topic}</span>
                  </button>
                </li>
              ))}
            </ol>
          </div>
        )}

        <Progress event={progressEvent} status={status} />

        {error && <section className="error-panel">{error}</section>}

        {run && (
          <div className="results">
            <section className="card answer-card">
              <header className="card-header">
                <div className="card-title">
                  <h2>Summary</h2>
                </div>
                <div className="badges">
                  {run.status && (
                    <span className={`badge ${run.status === "done" ? "badge-success" : "badge-warn"}`}>
                      {run.status === "done" ? "done" : "done with errors"}
                    </span>
                  )}
                  <span className="badge badge-muted">{run.latency_ms} ms</span>
                </div>
              </header>
              {(run.requested_roots ?? []).length > 0 && (
                <p className="filter-note">
                  Filtered to {(run.requested_roots ?? []).map((root) => ROOT_LABELS[root] ?? root).join(", ")}
                </p>
              )}
              <ErrorCards details={run.error_details} legacyErrors={run.errors} />
              {run.answer_markdown ? (
                <Markdown text={run.answer_markdown} />
              ) : (
                <p className="empty-state">No summary was generated for this run.</p>
              )}
            </section>

            <section className="card">
              <header className="card-header">
                <div className="card-title">
                  <h2>Matched documents</h2>
                  <span className="doc-groups-count">{(run.sources ?? []).length}</span>
                </div>
              </header>
              <DocGroups sources={run.sources ?? []} onOpen={(path) => void openDoc(path)} />
              {(run.sources ?? []).length === 0 && (
                <p className="empty-state">No documents matched this query. Try broader terms or fewer filters.</p>
              )}
            </section>

            <details className="card collapse-card">
              <summary>
                <span className="collapse-title">Evidence</span>
                <span className="collapse-count">{(run.citations ?? []).length} citations</span>
              </summary>
              <div className="collapse-body evidence-table">
                {(run.citations ?? []).slice(0, 50).map((citation) => (
                  <button
                    key={citation.citation}
                    type="button"
                    className="evidence-row"
                    onClick={() => void openDoc(citation.path)}
                  >
                    <code>{citation.citation}</code>
                    <span>{citation.snippet}</span>
                  </button>
                ))}
                {(run.citations ?? []).length === 0 && (
                  <p className="empty-state">No line-level citations for this run.</p>
                )}
              </div>
            </details>

            <details className="card collapse-card">
              <summary>
                <span className="collapse-title">Command trace</span>
                <span className="collapse-count">{(run.command_trace ?? []).length} operations</span>
              </summary>
              <div className="collapse-body">
                {run.planner_repair && (
                  <details className="repair-details">
                    <summary>
                      Planner repair {run.planner_repair.repair_succeeded ? "(succeeded)" : "(failed)"}
                    </summary>
                    {run.planner_repair.validation_error && (
                      <p className="repair-error">{run.planner_repair.validation_error}</p>
                    )}
                    <pre>{JSON.stringify(run.planner_repair, null, 2)}</pre>
                  </details>
                )}
                {(run.command_trace ?? []).map((command, idx) => (
                  <details key={`${command}-${idx}`} className="trace-entry">
                    <summary>
                      <code>{command}</code>
                    </summary>
                    <pre>{run.operation_outputs?.[idx]?.output_text || run.operation_outputs?.[idx]?.error}</pre>
                  </details>
                ))}
              </div>
            </details>
          </div>
        )}
      </main>

      <HistoryDrawer
        open={historyOpen}
        runs={history}
        onSelect={(runId) => void openRun(runId)}
        onClose={() => setHistoryOpen(false)}
      />
      <DocPanel doc={selectedDoc} onClose={() => setSelectedDoc(null)} />
    </div>
  );
}
