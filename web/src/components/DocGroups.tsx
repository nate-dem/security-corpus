import { ROOT_LABELS, Source } from "../lib/types";

export default function DocGroups({ sources, onOpen }: { sources: Source[]; onOpen: (path: string) => void }) {
  if (sources.length === 0) {
    return null;
  }
  const groups = new Map<string, Source[]>();
  for (const source of sources) {
    const root = "/" + (source.path.split("/")[1] || "docs");
    groups.set(root, [...(groups.get(root) ?? []), source]);
  }
  return (
    <div className="doc-groups">
      {Array.from(groups.entries()).map(([root, items]) => (
        <section className="doc-group" key={root}>
          <header className="doc-group-header">
            <span className="root-dot" data-root={root} />
            <h3>{ROOT_LABELS[root] ?? root}</h3>
            <span className="doc-group-count">{items.length}</span>
          </header>
          <div className="doc-rows">
            {items.map((source) => (
              <button key={source.path} type="button" className="doc-row" onClick={() => onOpen(source.path)}>
                <strong>{source.title || source.record_id}</strong>
                <span className="doc-row-meta">
                  {source.source_id} · {source.tokens != null ? `${source.tokens.toLocaleString()} tokens` : "tokens unknown"}
                </span>
              </button>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
