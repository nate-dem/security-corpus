"""Reddit subreddit connector for Arctic Shift JSONL dumps.

Handles individual subreddit dumps where all comments fit in memory.

In-memory two-pass approach:
  Pass 1: Read {Subreddit}_comments.zst, index by link_id (submission fullname)
  Pass 2: Read {Subreddit}_submissions.zst, filter, join comments, assemble threads

Scope filters applied during Pass 2:
  - Selftext posts only (is_self=True)
  - Skip deleted/removed submissions ([deleted], [removed], empty selftext)
  - Skip NSFW submissions (over_18=True)
  - Skip mod-removed submissions (removed_by_category set)

Deleted/removed comments are filtered during Pass 1 indexing.

Quality features computed in normalize():
  - content_length (tokens), content_hash (SHA-256)
  - score (submission score), answer_count (top-level comment count)
  - tags (subreddit + link flair)
"""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import QAThreadData
from ingest.connectors.reddit.common import (
    assemble_thread_content,
    build_comment_tree,
    clean_selftext,
    is_deleted_or_removed,
    parse_reddit_timestamp,
)
from ingest.readers import read
from ingest.utils import REDDIT_TERMS, compute_content_hash, compute_token_count


class RedditSubredditConnector:
    """Handles individual subreddit dumps from Arctic Shift.

    Each subreddit has two .zst files in the data directory:
      - {subreddit}_submissions.zst
      - {subreddit}_comments.zst

    This connector indexes all comments in memory and streams submissions.
    Works for subreddits under ~5M comments.
    """

    def __init__(self, subreddit: str):
        self.subreddit = subreddit
        self.source_id = f"reddit-{subreddit.lower()}"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Yield one assembled thread record per valid submission.

        ``path`` is the directory containing the .zst files for this subreddit.
        """
        # Pass 1: index comments by submission (link_id)
        comments_index = _build_comments_index(path, self.subreddit)

        # Pass 2: stream submissions, filter, join comments, yield assembled records
        submissions_path = path / f"{self.subreddit}_submissions.zst"
        for record in read(submissions_path):
            if not record.get("is_self", False):
                continue

            selftext = record.get("selftext", "")
            if is_deleted_or_removed(selftext):
                continue

            if record.get("over_18", False):
                continue

            if record.get("removed_by_category"):
                continue

            yield _assemble_thread_record(record, comments_index)

    def normalize(self, record: dict) -> QAThreadData:
        """Convert an assembled thread record into QAThreadData."""
        submission = record["submission"]
        sid = submission["id"]

        content = assemble_thread_content(
            title=submission["title"],
            selftext=submission["selftext_clean"],
            comment_tree=record["comment_tree"],
        )

        tags = [self.subreddit.lower()]
        if submission.get("link_flair_text"):
            tags.append(submission["link_flair_text"])

        return QAThreadData(
            record_id=f"{self.source_id}:submission-{sid}",
            source_id=self.source_id,
            source_record_id=f"submission-{sid}",
            content=content,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            title=submission["title"],
            ingested_at=datetime.now(timezone.utc),
            license=REDDIT_TERMS,
            published_at=parse_reddit_timestamp(submission.get("created_utc")),
            source_url=f"https://www.reddit.com{submission.get('permalink', '')}",
            score=submission.get("score"),
            answer_count=record["top_level_comment_count"],
            has_accepted_answer=None,
            closed=None,
            tags=tags,
        )


def _build_comments_index(data_dir: Path, subreddit: str) -> dict[str, list[dict]]:
    """Parse comments.zst, index by submission ID.

    Each comment dict has: id, parent_id, body, score.
    Deleted/removed comments are filtered out.
    """
    comments: dict[str, list[dict]] = defaultdict(list)
    comments_path = data_dir / f"{subreddit}_comments.zst"

    if not comments_path.exists():
        return comments

    for record in read(comments_path):
        body = record.get("body", "")
        if is_deleted_or_removed(body):
            continue

        link_id = record.get("link_id", "")
        if not link_id.startswith("t3_"):
            continue

        submission_id = link_id[3:]

        comments[submission_id].append({
            "id": record.get("id", ""),
            "parent_id": record.get("parent_id", ""),
            "body": body,
            "score": int(record.get("score", 0)),
        })

    return comments


def _assemble_thread_record(
    submission: dict,
    comments_index: dict[str, list[dict]],
) -> dict:
    """Build the full thread record with comment tree."""
    sid = submission.get("id", "")

    raw_comments = comments_index.get(sid, [])
    comment_tree = build_comment_tree(raw_comments, sid)

    top_level_count = len(comment_tree)

    assembled_submission = {
        "id": sid,
        "title": submission.get("title", ""),
        "selftext_clean": clean_selftext(submission.get("selftext", "")),
        "created_utc": submission.get("created_utc"),
        "score": int(submission.get("score", 0)),
        "permalink": submission.get("permalink", ""),
        "link_flair_text": submission.get("link_flair_text"),
    }

    return {
        "submission": assembled_submission,
        "comment_tree": comment_tree,
        "top_level_comment_count": top_level_count,
    }
