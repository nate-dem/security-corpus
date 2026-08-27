"""Safely assemble an arXiv LaTeX project into one deterministic document."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


SIMPLE_INCLUDE_RE = re.compile(
    r"\\(?P<simple_command>input|include|subfile)\*?"
    r"(?:\[[^\]]*\])*"
    r"(?:\{(?P<braced>[^}]+)\}|\s+(?P<unbraced>[^{}\s%]+))"
)
IMPORT_RE = re.compile(
    r"\\(?P<import_command>import|subimport|inputfrom|subinputfrom)\*?"
    r"\{(?P<directory>[^}]*)\}\{(?P<filename>[^}]*)\}"
)
INCLUDE_RE = re.compile(f"(?:{IMPORT_RE.pattern})|(?:{SIMPLE_INCLUDE_RE.pattern})")
COMMENT_ENVIRONMENT_RE = re.compile(
    r"\\begin\{comment\}.*?\\end\{comment\}",
    flags=re.DOTALL,
)
BLANK_LINES_RE = re.compile(r"(\n\s*){3,}")
VERBATIM_BEGIN_RE = re.compile(
    r"\\begin\{(?P<name>verbatim\*?|Verbatim|lstlisting|minted)\}"
)
VERBATIM_END_TEMPLATE = r"\\end\{%s\}"
MAIN_FILENAMES = ("main.tex", "paper.tex", "article.tex", "manuscript.tex")


@dataclass
class MergeDiagnostics:
    """Auditable include-resolution results for one paper."""

    main_file: str
    files_read: list[str] = field(default_factory=list)
    includes_found: int = 0
    includes_inlined: int = 0
    missing_includes: list[str] = field(default_factory=list)
    circular_includes: list[str] = field(default_factory=list)
    outside_project_includes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_main_file(tex_dir: Path) -> Path:
    """Select the most plausible top-level TeX file deterministically."""
    project_root = tex_dir.resolve()
    candidates: list[tuple[tuple[int, int, int, str], Path]] = []
    for path in sorted(project_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".tex", ".pdflatex"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        has_class = r"\documentclass" in text or r"\documentstyle" in text
        has_document = r"\begin{document}" in text
        if not has_class:
            continue
        relative = path.relative_to(project_root).as_posix()
        common_name = int(path.name.lower() in MAIN_FILENAMES)
        root_level = int(path.parent == project_root)
        candidates.append(
            ((-int(has_document), -common_name, -root_level, relative), path)
        )
    if not candidates:
        raise FileNotFoundError(f"No LaTeX main file found in {project_root}")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def inline_file(
    tex_path: Path,
    project_root: Path,
    seen: set[Path] | None = None,
    diagnostics: MergeDiagnostics | None = None,
) -> str:
    """Recursively inline supported include commands without leaving the project."""
    root = project_root.resolve()
    current = tex_path.resolve()
    _require_inside_project(current, root)
    if seen is None:
        seen = set()
    if diagnostics is None:
        diagnostics = MergeDiagnostics(main_file=current.relative_to(root).as_posix())
    seen.add(current)
    relative_current = current.relative_to(root).as_posix()
    if relative_current not in diagnostics.files_read:
        diagnostics.files_read.append(relative_current)
    content = current.read_text(encoding="utf-8", errors="ignore")

    def replace(match: re.Match[str]) -> str:
        diagnostics.includes_found += 1
        command = match.group("import_command") or match.group("simple_command")
        if command in {"import", "subimport", "inputfrom", "subinputfrom"}:
            directory = match.group("directory") or ""
            filename = match.group("filename") or ""
            requested = filename
            bases = [current.parent / directory]
        else:
            requested = match.group("braced") or match.group("unbraced") or ""
            # Nested files commonly use paths relative to themselves. Some
            # projects assume compilation from the root, so keep that fallback.
            bases = [current.parent, root]

        target, outside = _resolve_include(requested, bases, root)
        label = f"{relative_current}: {match.group(0)}"
        if outside:
            diagnostics.outside_project_includes.append(label)
            return f"% SECURITY_CORPUS_SKIPPED_OUTSIDE_PROJECT: {requested}\n"
        if target is None:
            diagnostics.missing_includes.append(label)
            return f"% SECURITY_CORPUS_MISSING_INCLUDE: {requested}\n"
        if target in seen:
            diagnostics.circular_includes.append(label)
            return f"% SECURITY_CORPUS_SKIPPED_CIRCULAR_INCLUDE: {requested}\n"

        diagnostics.includes_inlined += 1
        return inline_file(target, root, seen, diagnostics)

    return INCLUDE_RE.sub(replace, content)


def clean_latex(text: str) -> str:
    """Remove comments outside verbatim-like environments and normalize whitespace."""
    text = COMMENT_ENVIRONMENT_RE.sub("", text)
    output: list[str] = []
    offset = 0
    while offset < len(text):
        verbatim = VERBATIM_BEGIN_RE.search(text, offset)
        plain_end = verbatim.start() if verbatim else len(text)
        output.append(_strip_tex_comments(text[offset:plain_end]))
        if verbatim is None:
            break
        environment = verbatim.group("name")
        end_match = re.search(
            VERBATIM_END_TEMPLATE % re.escape(environment),
            text[verbatim.end() :],
        )
        if end_match is None:
            output.append(text[verbatim.start() :])
            break
        absolute_end = verbatim.end() + end_match.end()
        output.append(text[verbatim.start() : absolute_end])
        offset = absolute_end
    cleaned = "".join(output).replace("\r\n", "\n").replace("\r", "\n")
    return BLANK_LINES_RE.sub("\n\n", cleaned).strip() + "\n"


def merge_project(source_dir: Path, out_file: Path) -> MergeDiagnostics:
    """Write one cleaned document and return include-resolution diagnostics."""
    root = source_dir.resolve()
    main = find_main_file(root)
    diagnostics = MergeDiagnostics(main_file=main.relative_to(root).as_posix())
    inlined = inline_file(main, root, diagnostics=diagnostics)
    cleaned = clean_latex(inlined)
    if not cleaned.strip():
        raise ValueError(f"LaTeX assembly produced empty content for {root}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_file.with_suffix(out_file.suffix + ".tmp")
    temporary.write_text(cleaned, encoding="utf-8")
    temporary.replace(out_file)
    return diagnostics


def check_auto_ignore(project_dir: Path, project_id: str) -> bool:
    """Return whether arXiv supplied a single ``%auto-ignore`` placeholder."""
    tex_files = list(project_dir.glob("*.tex"))
    if len(tex_files) != 1 or tex_files[0].name != f"{project_id}.tex":
        return False
    return tex_files[0].read_text(encoding="utf-8", errors="ignore").strip() == "%auto-ignore"


def write_status_json(target_dir: Path, status: dict[str, Any]) -> None:
    """Atomically write one paper's normalization status."""
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / "status.json"
    temporary = target_dir / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def process_project(args: tuple[Path, str, str, Path]) -> tuple[int, int, int]:
    """Normalize an extracted project; retained for existing batch callers."""
    source_root, yymm, arxiv_id, target_root = args
    source = source_root / yymm / arxiv_id
    target = target_root / yymm / arxiv_id
    status_file = target / "status.json"
    if status_file.exists():
        try:
            if json.loads(status_file.read_text(encoding="utf-8")).get("completed"):
                return (0, 1, 0)
        except (json.JSONDecodeError, OSError):
            pass

    timestamp = datetime.now(timezone.utc).isoformat()
    if check_auto_ignore(source, arxiv_id):
        write_status_json(
            target,
            {
                "aid": arxiv_id,
                "auto_ignore": True,
                "completed": True,
                "normalizer_version": "latex-v2",
                "timestamp": timestamp,
            },
        )
        return (0, 1, 0)

    status: dict[str, Any] = {
        "aid": arxiv_id,
        "completed": False,
        "errors": [],
        "normalizer_version": "latex-v2",
        "source_format": "latex",
        "tex_merged": False,
        "timestamp": timestamp,
    }
    try:
        diagnostics = merge_project(source, target / "main.tex")
        status["include_diagnostics"] = diagnostics.to_dict()
        status["tex_merged"] = True
        status["completed"] = True
        write_status_json(target, status)
        return (1, 0, 0)
    except Exception as error:
        status["errors"].append(f"{type(error).__name__}: {error}")
        write_status_json(target, status)
        return (0, 0, 1)


def _resolve_include(
    requested: str,
    bases: list[Path],
    project_root: Path,
) -> tuple[Path | None, bool]:
    rendered = requested.strip()
    if not rendered:
        return None, False
    candidates = [rendered]
    if Path(rendered).suffix == "":
        candidates.append(f"{rendered}.tex")
    outside = False
    visited: set[Path] = set()
    for base in bases:
        for candidate in candidates:
            resolved = (base / candidate).resolve()
            if resolved in visited:
                continue
            visited.add(resolved)
            try:
                resolved.relative_to(project_root)
            except ValueError:
                outside = True
                continue
            if resolved.is_file():
                return resolved, outside
    return None, outside


def _require_inside_project(path: Path, project_root: Path) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"LaTeX path escapes project root: {path}") from error


def _strip_tex_comments(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        comment_at = None
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            lines.append(line)
        elif line.endswith("\n"):
            lines.append(line[:comment_at] + "\n")
        else:
            lines.append(line[:comment_at])
    return "".join(lines)
