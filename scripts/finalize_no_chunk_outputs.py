#!/usr/bin/env python3
"""Create final no-chunk Parquet artifacts for training/indexing.

This script is intentionally conservative: it refuses rows whose
``record_id`` or ``source_record_id`` contains ``:chunk-`` and writes arXiv
paper outputs in the ``AcademicPaperData`` column shape. It is meant for the
final corpus paths used by downstream builders that should see whole source
documents, not generated chunks.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingest.utils import compute_content_hash, compute_token_count as _base_compute_token_count  # noqa: E402


def compute_token_count(content: str) -> int:
    """Return cl100k token count while treating tokenizer sentinels as normal text."""
    try:
        return _base_compute_token_count(content)
    except ValueError as exc:
        if "disallowed special token" not in str(exc):
            raise
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(content, disallowed_special=()))


SECTION_RE = re.compile(
    r"\\(?P<cmd>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)"
    r"\*?(?:\[[^\]]*\])?\{(?P<title>[^{}]{1,300})\}",
    re.IGNORECASE,
)
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", re.IGNORECASE | re.DOTALL)
BEGIN_DOCUMENT_RE = re.compile(r"\\begin\{document\}", re.IGNORECASE)
END_DOCUMENT_RE = re.compile(r"\\end\{document\}", re.IGNORECASE)
THE_BIB_RE = re.compile(r"\\begin\{thebibliography\}.*", re.IGNORECASE | re.DOTALL)
BIBLIOGRAPHY_CMD_RE = re.compile(r"\\bibliography\{[^}]*\}.*", re.IGNORECASE | re.DOTALL)
TOKENIZER_SENTINEL_RE = re.compile(
    r"<\|(?:endoftext|endofprompt|fim_prefix|fim_middle|fim_suffix)\|>"
)
FIGURE_ENV_RE = re.compile(
    r"\\begin\{(?P<env>figure\*?|wrapfigure|sidewaysfigure)\}.*?\\end\{(?P=env)\}",
    re.IGNORECASE | re.DOTALL,
)
GRAPHICS_ENV_RE = re.compile(
    r"\\begin\{(?P<env>tikzpicture|axis|pgfpicture|pspicture|picture)\*?\}.*?"
    r"\\end\{(?P=env)\*?\}",
    re.IGNORECASE | re.DOTALL,
)
CAPTION_RE = re.compile(
    r"\\(?:caption|subcaption)(?:\[[^\]]*\])?\{(?P<body>(?:[^{}]|\{[^{}]*\})*)\}",
    re.IGNORECASE | re.DOTALL,
)
ADDPLOT_COORDINATES_RE = re.compile(
    r"\\addplot(?:\[[^\]]*\])?\s*coordinates\s*\{.*?\}\s*;",
    re.IGNORECASE | re.DOTALL,
)
COORDINATE_PAIR_RE = re.compile(r"\([-+]?\d+(?:\.\d+)?,\s*[-+]?\d+(?:\.\d+)?\)")
LOW_VALUE_SECTION_TITLES = {
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "bibliography",
    "references",
}


@dataclass
class Chunk:
    title: str
    text: str


ACADEMIC_FIELDS: list[tuple[str, pa.DataType]] = [
    ("source_id", pa.string()),
    ("source_record_id", pa.string()),
    ("record_id", pa.string()),
    ("content", pa.string()),
    ("title", pa.string()),
    ("content_length", pa.int64()),
    ("content_hash", pa.string()),
    ("ingested_at", pa.timestamp("us", tz="UTC")),
    ("published_at", pa.timestamp("us", tz="UTC")),
    ("source_url", pa.string()),
    ("license", pa.string()),
    ("raw", pa.null()),
    ("arxiv_id", pa.string()),
    ("source_format", pa.string()),
    ("authors", pa.list_(pa.string())),
    ("abstract", pa.string()),
    ("categories", pa.list_(pa.string())),
    ("primary_category", pa.string()),
    ("doi", pa.string()),
    ("journal_ref", pa.string()),
]

QWEN_FIELDS: list[tuple[str, pa.DataType]] = [
    ("qwen_security_relevance", pa.int16()),
    ("qwen_quality", pa.int16()),
    ("qwen_model_should_keep", pa.bool_()),
    ("qwen_should_keep", pa.bool_()),
    ("qwen_reason", pa.string()),
    ("qwen_parse_status", pa.string()),
    ("qwen_keep_policy", pa.string()),
    ("qwen_keep_policy_passed", pa.bool_()),
    ("qwen_keep_policy_reason", pa.string()),
    ("qwen_model", pa.string()),
    ("qwen_prompt_version", pa.string()),
    ("qwen_scored_at", pa.string()),
    ("qwen_shard_id", pa.string()),
    ("qwen_task", pa.string()),
    ("qwen_input_kind", pa.string()),
    ("qwen_raw_response", pa.string()),
]


def main() -> None:
    args = parse_args()
    if args.overwrite:
        _remove_path(args.output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "output_root": args.output_root.as_posix(),
        "outputs": {},
        "missing_optional_inputs": [],
    }

    cscr_input_files = _parquet_files(args.cscr_input)
    if cscr_input_files:
        cscr_out = args.output_root / "academic_papers" / "arxiv_cs_cr_full.parquet"
        manifest["outputs"]["arxiv_cs_cr_full"] = _write_academic_papers(
            input_files=cscr_input_files,
            output_path=cscr_out,
            batch_size=args.batch_size,
            require_cscr_category=True,
            include_qwen_fields=False,
            clean_content=not args.no_clean_academic_content,
        )
    elif args.require_cscr:
        raise FileNotFoundError(
            f"No cs.CR full-paper Parquet found at {args.cscr_input}. "
            "Copy the unchunked Marlowe file first or pass --cscr-input."
        )
    else:
        manifest["missing_optional_inputs"].append(args.cscr_input.as_posix())

    citation_input_files = _parquet_files(args.citation_input)
    if not citation_input_files:
        raise FileNotFoundError(f"No citation kept Parquet found at {args.citation_input}")
    citation_out = (
        args.output_root / "academic_papers" / "arxiv_citation_qwen_kept_full.parquet"
    )
    manifest["outputs"]["arxiv_citation_qwen_kept_full"] = _write_academic_papers(
        input_files=citation_input_files,
        output_path=citation_out,
        batch_size=args.batch_size,
        require_cscr_category=False,
        include_qwen_fields=True,
        clean_content=not args.no_clean_academic_content,
    )

    if args.materialize_supporting_sources:
        manifest["outputs"]["cloudtrail_flaws_no_chunk"] = _copy_no_chunk_parquet(
            input_path=args.cloudtrail_input,
            output_path=args.output_root
            / "normalized"
            / "source_id=cloudtrail-flaws"
            / "part-00000.parquet",
            label="cloudtrail-flaws",
        )
        manifest["outputs"]["sigma_no_chunk"] = _copy_no_chunk_parquet(
            input_path=args.sigma_input,
            output_path=args.output_root / "normalized" / "source_id=sigma" / "part-00000.parquet",
            label="sigma",
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cscr-input",
        type=Path,
        default=Path("data/training-clean-v1/normalized/source_id=arxiv"),
        help="Unchunked cs.CR AcademicPaperData Parquet file or source_id=arxiv directory.",
    )
    parser.add_argument(
        "--citation-input",
        type=Path,
        default=Path("data/filtering/v3/qwen_citation_abstract_kept_full.parquet"),
        help="Qwen-kept citation full-paper Parquet.",
    )
    parser.add_argument(
        "--cloudtrail-input",
        type=Path,
        default=Path("data/cloudtrail/normalized/source_id=cloudtrail-flaws/flaws.parquet"),
    )
    parser.add_argument(
        "--sigma-input",
        type=Path,
        default=Path("data/sigma/normalized/source_id=sigma/raw.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/final-no-chunk"),
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument(
        "--no-clean-academic-content",
        action="store_true",
        help="Preserve arXiv content as stored instead of writing v2-style cleaned full papers.",
    )
    parser.add_argument("--require-cscr", action="store_true")
    parser.add_argument(
        "--materialize-supporting-sources",
        action="store_true",
        help="Also write unchunked CloudTrail and Sigma Parquets from original normalized data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    return []


def _write_academic_papers(
    *,
    input_files: list[Path],
    output_path: Path,
    batch_size: int,
    require_cscr_category: bool,
    include_qwen_fields: bool,
    clean_content: bool,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ACADEMIC_FIELDS + (QWEN_FIELDS if include_qwen_fields else [])
    schema = pa.schema([pa.field(name, type_) for name, type_ in fields])
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    tokens = 0
    examples: list[str] = []
    try:
        for path in input_files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size):
                projected_rows = []
                for row in batch.to_pylist():
                    if _has_chunk_id(row):
                        raise ValueError(f"{path}: chunked academic row found: {row.get('record_id')}")
                    if require_cscr_category and "cs.CR" not in _string_list(row.get("categories")):
                        continue
                    projected = _project_academic_row(
                        row,
                        include_qwen_fields=include_qwen_fields,
                        clean_content=clean_content,
                    )
                    projected_rows.append(projected)
                    tokens += int(projected.get("content_length") or 0)
                    if len(examples) < 3:
                        examples.append(str(projected.get("record_id") or ""))
                if not projected_rows:
                    continue
                table = pa.Table.from_pydict(
                    {
                        field.name: pa.array(
                            [row.get(field.name) for row in projected_rows],
                            type=field.type,
                        )
                        for field in schema
                    },
                    schema=schema,
                )
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
                writer.write_table(table)
                rows_written += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if rows_written == 0:
        raise ValueError(f"{output_path}: wrote zero academic rows")
    return {
        "path": output_path.as_posix(),
        "rows": rows_written,
        "tokens": tokens,
        "example_record_ids": examples,
    }


def _project_academic_row(
    row: dict[str, Any],
    *,
    include_qwen_fields: bool,
    clean_content: bool,
) -> dict[str, Any]:
    out = {name: row.get(name) for name, _ in ACADEMIC_FIELDS}
    out["source_id"] = out.get("source_id") or "arxiv"
    out["source_record_id"] = out.get("source_record_id") or out.get("arxiv_id") or ""
    out["record_id"] = out.get("record_id") or f"arxiv:{out['source_record_id']}"
    out["raw"] = None
    out["authors"] = _string_list(row.get("authors"))
    out["categories"] = _string_list(row.get("categories"))
    source_format = row.get("source_format")
    out["source_format"] = source_format if source_format in {"latex", "pdf"} else None
    if clean_content:
        cleaned_content = _format_clean_full_arxiv_paper(row)
        out["content"] = cleaned_content
        out["content_length"] = compute_token_count(cleaned_content)
        out["content_hash"] = compute_content_hash(cleaned_content)
    if include_qwen_fields:
        for name, _ in QWEN_FIELDS:
            value = row.get(name)
            if name == "qwen_shard_id" and value is not None:
                value = str(value)
            out[name] = value
    return out


def _format_clean_full_arxiv_paper(row: dict[str, Any]) -> str:
    title = str(row.get("title") or row.get("arxiv_id") or "arXiv paper").strip()
    arxiv_id = str(row.get("arxiv_id") or row.get("source_record_id") or "").strip()
    cleaned, extracted_abstract = _clean_arxiv_content(str(row.get("content") or ""))
    abstract = extracted_abstract or row.get("abstract")
    blocks = _arxiv_blocks(cleaned, title=title, abstract=abstract)
    body_parts = [_format_block(block) for block in blocks if block.text.strip()]
    body = "\n\n".join(body_parts).strip() or cleaned.strip()
    header = [
        f"# {title}",
        "",
        f"arXiv ID: {arxiv_id}",
        "Document: Full paper",
    ]
    return "\n".join(header).strip() + "\n\n" + body


def _clean_arxiv_content(content: str) -> tuple[str, str | None]:
    text = _remove_tokenizer_sentinels(content).replace("\r\n", "\n").replace("\r", "\n")
    begin_match = BEGIN_DOCUMENT_RE.search(text)
    if begin_match:
        text = text[begin_match.end():]
    end_match = END_DOCUMENT_RE.search(text)
    if end_match:
        text = text[:end_match.start()]

    abstract = None
    abstract_match = ABSTRACT_RE.search(text)
    if abstract_match:
        abstract = _clean_latex_text(abstract_match.group("body"))
        text = text[:abstract_match.start()] + "\n" + text[abstract_match.end():]

    text = THE_BIB_RE.sub("", text)
    text = BIBLIOGRAPHY_CMD_RE.sub("", text)
    text = re.sub(
        r"\\appendix\b",
        lambda _match: "\n\\section{Appendix}\n",
        text,
        flags=re.IGNORECASE,
    )
    text = _remove_latex_rendering_artifacts(text)
    text = _remove_latex_noise_lines(text)
    text = _clean_latex_text(text)
    return text, abstract


def _remove_tokenizer_sentinels(text: str) -> str:
    return TOKENIZER_SENTINEL_RE.sub(" ", text)


def _remove_latex_rendering_artifacts(text: str) -> str:
    text = FIGURE_ENV_RE.sub(_figure_env_replacement, text)
    text = GRAPHICS_ENV_RE.sub(" ", text)
    text = ADDPLOT_COORDINATES_RE.sub(" ", text)
    return text


def _figure_env_replacement(match: re.Match[str]) -> str:
    captions = []
    for caption_match in CAPTION_RE.finditer(match.group(0)):
        caption = _clean_latex_text(caption_match.group("body"))
        caption = re.sub(r"[{}]", "", caption).strip()
        if caption:
            captions.append(f"Figure caption: {caption}")
    return "\n\n".join(captions)


def _remove_latex_noise_lines(text: str) -> str:
    drop_line_re = re.compile(
        r"^\s*\\("
        r"documentclass|usepackage|RequirePackage|newcommand|renewcommand|providecommand|"
        r"DeclareMathOperator|DeclareRobustCommand|DeclareUnicodeCharacter|newtheorem|"
        r"theoremstyle|setlength|addtolength|graphicspath|hypersetup|bibliographystyle|"
        r"title|author|date|maketitle|tableofcontents|pagestyle|thispagestyle"
        r")\b",
        re.IGNORECASE,
    )
    drop_only_re = re.compile(
        r"^\s*\\(label|vspace|hspace|smallskip|medskip|bigskip|clearpage|newpage|pagebreak)\b.*$",
        re.IGNORECASE,
    )
    render_command_re = re.compile(
        r"^\s*\\(?:"
        r"begin\{(?:tikzpicture|axis|pgfpicture|pspicture|picture)\*?\}|"
        r"end\{(?:tikzpicture|axis|pgfpicture|pspicture|picture)\*?\}|"
        r"(?:pgf[a-zA-Z]*|addplot|addlegendentry|draw|path|coordinate|fill|shade)\b"
        r")",
        re.IGNORECASE,
    )
    kept_lines = []
    for line in text.splitlines():
        stripped = re.sub(r"(?<!\\)%.*$", "", line).strip()
        if not stripped:
            kept_lines.append("")
            continue
        coordinate_pairs = len(COORDINATE_PAIR_RE.findall(stripped))
        if (
            drop_line_re.match(stripped)
            or drop_only_re.match(stripped)
            or render_command_re.match(stripped)
            or coordinate_pairs >= 20
            or stripped.count(r"\pgfqpoint") >= 3
        ):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _clean_latex_text(text: str) -> str:
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\(emph|textbf|textit|texttt|mathrm|mathbf)\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _arxiv_blocks(cleaned: str, *, title: str, abstract: str | None) -> list[Chunk]:
    blocks: list[Chunk] = []
    if abstract and abstract.strip():
        blocks.append(Chunk(title="Abstract", text=abstract.strip()))

    matches = list(SECTION_RE.finditer(cleaned))
    if not matches:
        body = cleaned.strip()
        if body:
            blocks.append(Chunk(title="Body", text=body))
        return blocks

    front_matter = cleaned[:matches[0].start()].strip()
    front_matter = _strip_empty_latex_commands(front_matter)
    if front_matter:
        blocks.append(Chunk(title="Front Matter", text=front_matter))

    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        heading = _clean_heading(match.group("title"))
        body = cleaned[match.end():next_start].strip()
        body = _strip_empty_latex_commands(body)
        if not body:
            continue
        if _is_low_value_section(heading):
            continue
        blocks.append(Chunk(title=heading or title, text=body))
    return blocks


def _strip_empty_latex_commands(text: str) -> str:
    text = re.sub(r"^\s*\\(begin|end)\{document\}\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_heading(title: str) -> str:
    title = _clean_latex_text(title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def _is_low_value_section(title: str) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", title.lower()).strip()
    return normalized in LOW_VALUE_SECTION_TITLES


def _format_block(block: Chunk) -> str:
    if block.title:
        return f"## {block.title}\n\n{block.text.strip()}"
    return block.text.strip()


def _copy_no_chunk_parquet(*, input_path: Path, output_path: Path, label: str) -> dict[str, Any]:
    files = _parquet_files(input_path)
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one {label} Parquet at {input_path}, found {len(files)}")
    stats = _no_chunk_stats(files[0])
    if stats["chunked_rows"]:
        raise ValueError(f"{files[0]} contains {stats['chunked_rows']:,} chunked rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(files[0], output_path)
    return {"path": output_path.as_posix(), **stats}


def _no_chunk_stats(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    rows = 0
    chunked = 0
    tokens = 0
    columns = set(parquet.schema_arrow.names)
    needed = [name for name in ("record_id", "source_record_id", "content_length") if name in columns]
    for batch in parquet.iter_batches(columns=needed, batch_size=65_536):
        for row in batch.to_pylist():
            rows += 1
            if _has_chunk_id(row):
                chunked += 1
            tokens += int(row.get("content_length") or 0)
    return {"rows": rows, "chunked_rows": chunked, "tokens": tokens}


def _has_chunk_id(row: dict[str, Any]) -> bool:
    return any(":chunk-" in str(row.get(name) or "") for name in ("record_id", "source_record_id"))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
