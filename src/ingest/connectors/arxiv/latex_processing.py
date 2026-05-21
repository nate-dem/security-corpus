"""LaTeX normalization helpers for arXiv paper sources.

Adapted from the Arxiv-Scraper project (~/Desktop/Arxiv-Scraper/src/ingest/normalize_latex.py).
Finds the main TeX file, recursively inlines \\input/\\include directives,
strips comments, and collapses blank lines to produce a single cleaned document.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


INPUT_RE = re.compile(r'''
    (?<!%)                            # skip commented-out lines
    \\(?P<cmd>input|include)\*?      # match \input or \include (optionally starred)
    (?:\[[^\]]*\])*                   # optional [key=val] arguments
    (?:
      \{(?P<file_braced>[^}]+)\}      # braced filename
      |
      \s+(?P<file_unbraced>[^{}\s%]+) # whitespace + unbraced filename
    )
''', re.VERBOSE)

# Regex to remove comments
COMMENT_LINE_RE = re.compile(r'(?m)^%.*$')
INLINE_COMMENT_RE = re.compile(r'(?m)(?<!\\)%.*$')
COMMENT_SECTION_RE = re.compile(
    r'\\begin\{comment\}.*?\\end\{comment\}',
    flags=re.DOTALL,
)
BLANK_LINES_RE = re.compile(r"(\n\s*){3,}")


def find_main_file(tex_dir: Path) -> Path:
    """Return the file containing ``\\documentclass`` and ``\\begin{document}``."""
    candidates = []
    tex_files = (
        list(tex_dir.rglob("*.tex"))
        + list(tex_dir.rglob("*.TEX"))
        + list(tex_dir.rglob("*.pdflatex"))
    )
    for tex in tex_files:
        text = tex.read_text(encoding="utf-8", errors="ignore")
        if r"\documentclass" in text or r"\documentstyle" in text:
            candidates.append(tex)
    if not candidates:
        raise FileNotFoundError(f"No main file found in {tex_dir}.")
    # First check for common main file names
    main_names = ["main.tex", "paper.tex", "article.tex", "manuscript.tex"]
    for name in main_names:
        for candidate in candidates:
            if candidate.name.lower() == name.lower():
                return candidate
    # Otherwise choose the one where \documentclass appears earliest
    return min(
        candidates,
        key=lambda t: t.read_text(encoding="utf-8", errors="ignore")
                       .find(r"\documentclass"),
    )


def inline_file(tex_path: Path, base_dir: Path, seen: set[Path] | None = None) -> str:
    if seen is None:
        seen = set()
    content = tex_path.read_text(encoding="utf-8", errors="ignore")
    def _repl(m: re.Match) -> str:
        # extract the relative path from whichever group matched
        rel = m.group("file_braced") or m.group("file_unbraced")
        # ensure .tex extension
        fname = rel if rel.endswith(".tex") else f"{rel}.tex"
        fpath = (base_dir / fname).resolve()
        # skip if missing or circular
        if fpath in seen or not fpath.exists():
            return f"% Skipped missing or circular include: {fname}\n"
        seen.add(fpath)
        # recurse to inline nested inputs
        return inline_file(fpath, base_dir, seen)
    return INPUT_RE.sub(_repl, content)


def clean_latex(text: str) -> str:
    """Strip comments and collapse excessive blank lines."""
    # 1) Remove whole-line comments
    text = COMMENT_LINE_RE.sub("", text)
    # 2) Remove inline comments
    text = INLINE_COMMENT_RE.sub("", text)
    # 3) Remove any comment-environment blocks
    text = COMMENT_SECTION_RE.sub("", text)
    # 4) Collapse runs of 3+ blank lines into just 2
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text


def merge_project(source_dir: Path, out_file: Path) -> bool:
    """Produce a clean, single ``main.tex`` from a multi-file LaTeX project."""
    main = find_main_file(source_dir)
    inlined = inline_file(main, source_dir)
    cleaned = clean_latex(inlined)
    out_file.write_text(cleaned, encoding="utf-8")
    return True


def check_auto_ignore(project_dir: Path, project_id: str) -> bool:
    """
    Check if this is an auto-ignore case: a single file named projectid.tex 
    with content '%auto-ignore'
    
    Args:
        project_dir: Directory to check
        project_id: The project ID
        
    Returns:
        True if this is an auto-ignore case, False otherwise
    """
    # Check if there's exactly one .tex file
    tex_files = list(project_dir.glob("*.tex"))
    if len(tex_files) != 1:
        return False
    
    # Check if the file is named after the project ID
    if tex_files[0].name != f"{project_id}.tex":
        return False
    
    # Check file content
    content = tex_files[0].read_text(encoding="utf-8", errors="ignore").strip()
    return content == "%auto-ignore"


def write_status_json(target_dir: Path, status: dict[str, Any]) -> None:
    """Write processing status to ``status.json``."""
    with open(target_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def process_project(args: tuple) -> tuple[int, int, int]:
    """Normalize a single LaTeX project.

    ``args`` is ``(source_root, yymm, arxiv_id, target_root)``.
    Returns ``(processed, skipped, failed)`` counts (each 0 or 1).
    """
    source_root, yymm, arxiv_id, target_root = args
    src = source_root / yymm / arxiv_id
    tgt = target_root / yymm / arxiv_id
    tgt.mkdir(parents=True, exist_ok=True)

    # Already completed?
    status_file = tgt / "status.json"
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status = json.load(f)
                if status.get("completed", False):
                    return (0, 1, 0)
        except Exception:
            pass

    if check_auto_ignore(src, arxiv_id):
        write_status_json(tgt, {
            "aid": arxiv_id, "completed": True, "auto_ignore": True,
            "timestamp": datetime.now().isoformat(),
        })
        return (0, 1, 0)

    status: dict[str, Any] = {
        "aid": arxiv_id,
        "timestamp": datetime.now().isoformat(),
        "completed": False,
        "tex_merged": False,
        "errors": [],
    }
    write_status_json(tgt, status)

    try:
        ok = merge_project(src, tgt / "main.tex")
        status["tex_merged"] = ok
        status["completed"] = True
        write_status_json(tgt, status)
        return (1, 0, 0)
    except Exception as e:
        status["errors"].append(str(e))
        write_status_json(tgt, status)
        return (0, 0, 1)
