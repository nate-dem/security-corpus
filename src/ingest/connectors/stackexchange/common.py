"""Shared helpers for Stack Exchange connectors.

Used by StackExchangeSiteConnector (small-site in-memory connector) and
will be reused by a future StackOverflowConnector (streaming multi-pass).
"""

import re
from datetime import datetime, timezone

from markdownify import markdownify


_TAG_RE = re.compile(r"<([^>]+)>")
_COLLAPSE_NEWLINES_RE = re.compile(r"\n{3,}")


def html_to_markdown(html: str) -> str:
    """Convert SE HTML body to Markdown suitable for training data.

    Uses markdownify with ATX headings. Preserves fenced code blocks
    (```...```) and inline code (`...`). Collapses runs of 3+ newlines
    to 2 (paragraph break).

    Returns empty string for empty/None input.
    """
    if not html:
        return ""
    md = markdownify(html, heading_style="ATX")
    md = _COLLAPSE_NEWLINES_RE.sub("\n\n", md)
    return md.strip()


def parse_tag_string(tag_str: str | None) -> list[str]:
    """Parse Stack Exchange tag strings into a list.

    Handles both formats:
      - angle brackets: "<cryptography><ssl>" → ["cryptography", "ssl"]
      - pipe-delimited: "|cryptography|ssl|" → ["cryptography", "ssl"]
    """
    if not tag_str:
        return []
    # angle-bracket format
    matches = _TAG_RE.findall(tag_str)
    if matches:
        return matches
    # pipe-delimited format
    return [t for t in tag_str.split("|") if t]


def parse_se_datetime(value: str | None) -> datetime | None:
    """Parse SE timestamp (ISO-ish without timezone) as UTC datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def assemble_qa_content(question: dict, answers: list[dict]) -> str:
    """Build a training document from a question and its answers.

    Structure:
        # {title}

        {question body markdown}

        ---

        ## Answer (Accepted) (Score: N)

        {answer body markdown}

        ---

        ## Answer (Score: M)

        {answer body markdown}

    Answers are expected to already be sorted by score descending.
    """
    parts = [f"# {question['title']}\n\n{question['body_md']}"]
    for ans in answers:
        header = "## Answer"
        if ans.get("is_accepted"):
            header = "## Answer (Accepted)"
        header += f" (Score: {ans['score']})"
        parts.append(f"{header}\n\n{ans['body_md']}")
    return "\n\n---\n\n".join(parts)


def detect_code_in_html(html: str) -> bool:
    """Detect whether HTML body contains code blocks (<pre><code> elements).

    Returns True if at least one <pre><code> or standalone <code> block is found.
    """
    if not html:
        return False
    # Fast string check before parsing — <code is present in virtually all
    # posts with code, and absent in those without.
    return "<code" in html


def extract_closure(attrs: dict) -> bool:
    """Extract closure status from SE post attributes.

    Returns closed where closed is True if ClosedDate is present in the post attributes.
    """
    closed = "ClosedDate" in attrs
    return closed
