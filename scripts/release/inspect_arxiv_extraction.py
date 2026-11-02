"""Read-only follow-up for literal LaTeX includes and abstract-only records.

These are diagnostic matches, not exclusions. A paper about LaTeX can quote an
include command legitimately. Inspect the saved context before changing data.
"""

import argparse
from collections import Counter
from pathlib import Path
import json
import re

import pyarrow.parquet as pq

from ingest.connectors.arxiv.latex_processing import INCLUDE_RE
from scripts.release.inspect_final_corpus import save, sha


def body_sections_are_only_includes(text):
    """Prove body sections contain only unresolved includes, not quoted examples.

    Abstract/front-matter may remain. This narrow structural finding uses no
    length threshold and does not classify papers with any remaining body text.
    Reject multi-line regex matches that could consume malformed source prose.
    """
    parts = re.split(r"(?m)^## (.+)\n", text)
    bodies = [parts[i + 1] for i in range(1, len(parts), 2)
              if parts[i].lower() not in {"abstract", "front matter"}]
    if not bodies:
        return False
    found = False
    for body in bodies:
        matches = list(INCLUDE_RE.finditer(body))
        if any("\n" in match.group() for match in matches):
            return False
        found = found or bool(matches)
        if INCLUDE_RE.sub("", body).strip():
            return False
    return found


def diagnose(text):
    includes = list(INCLUDE_RE.finditer(text))
    abstract = re.search(r"(?m)^## Abstract\s*$", text)
    # The export inserts this heading. No following section is a review lead,
    # not proof that the record contains no body: headings can be lost too.
    no_section_after_abstract = bool(abstract and not re.search(
        r"(?m)^## ", text[abstract.end():]
    ))
    return {
        "include_commands": [m.group(0) for m in includes],
        "include_contexts": [text[max(0, m.start()-100):m.end()+100] for m in includes[:8]],
        "no_section_after_abstract": no_section_after_abstract,
    }


def run(corpus, audit, output):
    if output.exists() or output.resolve().is_relative_to(corpus.resolve()):
        raise ValueError("Use a new output path outside the corpus")
    binding = json.loads((audit / "binding.json").read_text())
    manifest_path = corpus / "corpus-manifest.json"
    if sha(manifest_path) != binding["manifest_sha256"]:
        raise ValueError("Corpus differs from the exhaustive scan")
    manifest = json.loads(manifest_path.read_text())
    counts, candidates = Counter(), []
    for file_index, entry in enumerate(manifest["files"]):
        path = corpus / entry["relative_path"]
        with pq.ParquetFile(path) as pf:
            first = next(pf.iter_batches(batch_size=1, columns=["source_id"]))
            if first.column(0)[0].as_py() != "arxiv":
                continue
            if sha(path) != entry["sha256"]:
                raise ValueError(f"Input changed: {path}")
            row_index = 0
            for batch in pf.iter_batches(batch_size=128, columns=[
                "source_id", "record_id", "content_hash", "content", "title",
                "content_length", "source_format",
            ]):
                for row in batch.to_pylist():
                    if row["source_id"] != "arxiv":
                        raise ValueError("Mixed source shard requires explicit routing")
                    text = row.pop("content")
                    finding = diagnose(text)
                    counts["records_checked"] += 1
                    counts["literal_include_records"] += bool(finding["include_commands"])
                    counts["no_section_after_abstract_records"] += finding["no_section_after_abstract"]
                    if finding["include_commands"] or finding["no_section_after_abstract"]:
                        candidates.append(dict(row, file=file_index, row=row_index, **finding))
                    row_index += 1
            if sha(path) != entry["sha256"]:
                raise ValueError(f"Input changed during scan: {path}")
    expected = json.loads((audit / "summary.json").read_text())["by_source"]["arxiv"]["records"]
    if counts["records_checked"] != expected:
        raise ValueError("Incomplete arXiv coverage")
    save(output, dict(
        complete=True, manifest_sha256=binding["manifest_sha256"],
        code_sha256=sha(__file__), counts=counts, candidates=candidates,
        scope="Literal includes and absent post-abstract section headings are review leads, not proof of missing text or automatic exclusion decisions.",
    ))
    print(json.dumps(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.corpus, args.audit, args.output)
