import { DocPreview } from "../lib/types";

export default function DocPanel({ doc, onClose }: { doc: DocPreview | null; onClose: () => void }) {
  if (!doc) {
    return null;
  }
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="doc-drawer" role="dialog" aria-label="Document preview">
        <div className="drawer-header">
          <h2>Document</h2>
          <button type="button" className="ghost-button" onClick={onClose}>
            Close
          </button>
        </div>
        <code className="doc-path">{doc.path}</code>
        <pre className="doc-body">{doc.text}</pre>
      </aside>
    </>
  );
}
