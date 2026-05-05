"""Shared helpers for Reddit connectors.

Used by RedditSubredditConnector (in-memory two-pass for individual subreddit dumps).
"""

import re
from datetime import datetime, timezone

# depth of 7 collects ~97-98% of comments on average for all subreddits; can be tuned
MAX_COMMENT_DEPTH = 7

_COLLAPSE_NEWLINES_RE = re.compile(r"\n{3,}")

_DELETED_MARKERS = frozenset({"[deleted]", "[removed]"})


def parse_reddit_timestamp(utc_timestamp) -> datetime | None:
    """Convert a Unix UTC timestamp (int or float) to a timezone-aware datetime."""
    if utc_timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(float(utc_timestamp), tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None


def is_deleted_or_removed(text: str | None) -> bool:
    """Check if text is a Reddit deleted/removed marker or empty."""
    if not text:
        return True
    return text.strip() in _DELETED_MARKERS


def clean_selftext(text: str | None) -> str:
    """Clean Reddit selftext for training data.

    Collapses excessive newlines. Returns empty string for empty/None input.
    """
    if not text:
        return ""
    text = _COLLAPSE_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def build_comment_tree(
    comments: list[dict],
    submission_id: str,
) -> list[dict]:
    """Build a nested comment tree from a flat list of comments.

    Each comment dict must have 'id', 'parent_id', 'body', 'score'.
    parent_id is either 't3_xxx' (top-level, parent is submission) or
    't1_xxx' (reply to another comment).

    Returns a list of top-level comment dicts, each with a 'children' key
    containing nested replies. Sorted by score descending at each level.
    Orphan comments (parent missing from the set) are dropped.
    """
    by_id: dict[str, dict] = {}
    for c in comments:
        c_copy = dict(c)
        c_copy["children"] = []
        by_id[f"t1_{c_copy['id']}"] = c_copy

    top_level: list[dict] = []
    submission_fullname = f"t3_{submission_id}"

    for c in by_id.values():
        parent = c.get("parent_id", "")
        if parent == submission_fullname:
            top_level.append(c)
        elif parent in by_id:
            by_id[parent]["children"].append(c)
        # if neither of these conditionals execute then parent was deleted/missing - drop the comment

    _sort_tree(top_level)
    return top_level


def _sort_tree(comments: list[dict]) -> None:
    """Recursively sort comment tree by score descending."""
    comments.sort(key=lambda c: c.get("score", 0), reverse=True)
    for c in comments:
        _sort_tree(c.get("children", []))


def assemble_thread_content(
    title: str,
    selftext: str,
    comment_tree: list[dict],
    max_depth: int = MAX_COMMENT_DEPTH,
) -> str:
    """Build a training document from a Reddit submission and its comment tree.

    Structure:
        # {title}

        {selftext}

        ---

        **Comment** (Score: N)

        comment body

        > **Reply** (Score: M)
        >
        > reply body
        >
        > > **Reply** (Score: K)
        > >
        > > nested reply body

    Uses blockquote indentation (>) for nesting depth.
    """
    parts = [f"# {title}"]
    if selftext:
        parts.append(selftext)

    if comment_tree:
        parts.append("---")
        for comment in comment_tree:
            rendered = _render_comment(comment, depth=0, max_depth=max_depth)
            if rendered:
                parts.append(rendered)

    return "\n\n".join(parts)


def _render_comment(comment: dict, depth: int, max_depth: int) -> str:
    """Render a single comment and its children as markdown.

    depth=0: no blockquote prefix (top-level comments)
    depth=1: '> ' prefix
    depth=2: '> > ' prefix
    etc.
    """
    if depth >= max_depth:
        return ""

    prefix = "> " * depth
    score = comment.get("score", 0)
    body = clean_selftext(comment.get("body", ""))

    if depth == 0:
        header = f"**Comment** (Score: {score})"
    else:
        header = f"**Reply** (Score: {score})"

    lines = [f"{prefix}{header}", f"{prefix}"]
    for body_line in body.split("\n"):
        lines.append(f"{prefix}{body_line}")

    result = "\n".join(lines)

    children_parts = []
    for child in comment.get("children", []):
        rendered = _render_comment(child, depth + 1, max_depth)
        if rendered:
            children_parts.append(rendered)

    if children_parts:
        result += "\n" + prefix + "\n" + ("\n" + prefix + "\n").join(children_parts)

    return result
