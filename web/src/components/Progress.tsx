import { ProgressEvent, RunStatus } from "../lib/types";

const STAGES = [
  { id: "routing", label: "Routing" },
  { id: "planning", label: "Planning" },
  { id: "validating", label: "Validating" },
  { id: "executing", label: "Retrieving" },
  { id: "synthesizing", label: "Synthesizing" },
  { id: "done", label: "Done" }
];

export default function Progress({ event, status }: { event: ProgressEvent | null; status: RunStatus }) {
  if (status === "idle") {
    return null;
  }
  const activeStage = status === "done" ? "done" : event?.stage ?? "routing";
  const activeIdx = STAGES.findIndex((stage) => stage.id === activeStage);
  return (
    <div className="progress" aria-live="polite">
      <ol className="progress-track">
        {STAGES.map((stage, idx) => {
          const state =
            status === "done" || idx < activeIdx ? "completed" : idx === activeIdx && status !== "error" ? "active" : "";
          return (
            <li key={stage.id} className={state}>
              <span className="progress-dot" />
              {stage.label}
            </li>
          );
        })}
      </ol>
      {status === "running" && event?.stage === "executing" && (
        <p className="progress-detail">
          Running operation {event.operation_index ?? "?"}/{event.operation_count ?? "?"}
          {event.command ? (
            <>
              {" — "}
              <code>{event.command}</code>
            </>
          ) : null}
        </p>
      )}
    </div>
  );
}
