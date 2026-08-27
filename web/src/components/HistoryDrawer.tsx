import { RunSummary } from "../lib/types";

export default function HistoryDrawer({
  open,
  runs,
  onSelect,
  onClose
}: {
  open: boolean;
  runs: RunSummary[];
  onSelect: (runId: string) => void;
  onClose: () => void;
}) {
  if (!open) {
    return null;
  }
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="history-drawer" role="dialog" aria-label="Run history">
        <div className="drawer-header">
          <h2>Recent runs</h2>
          <button type="button" className="ghost-button" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="history-list">
          {runs.map((item) => (
            <button key={item.run_id} type="button" className="history-item" onClick={() => onSelect(item.run_id)}>
              <span>{item.query}</span>
              <small>
                {new Date(item.created_at).toLocaleString()} · {item.latency_ms} ms
              </small>
            </button>
          ))}
          {runs.length === 0 && <p className="empty-state">No runs yet.</p>}
        </div>
      </aside>
    </>
  );
}
