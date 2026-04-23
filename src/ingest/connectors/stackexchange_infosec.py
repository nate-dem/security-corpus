import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup

from ingest.connectors.base import NormalizedData
from ingest.readers import read


_TAG_RE = re.compile(r"<([^>]+)>")


def _strip_html(html: str) -> str:
    """Convert HTML body text to plain text, preserving paragraph breaks."""
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    # collapse runs of 3+ newlines to 2 (paragraph break)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _parse_tag_string(tag_str: str) -> list[str]:
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


def _parse_datetime(value: str | None) -> datetime | None:
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


def _row_to_dict(elem) -> dict:
    """Convert an lxml element with attributes to a plain dict."""
    return dict(elem.attrib)


def _build_users_index(extracted_dir: Path) -> dict[int, dict]:
    """Parse Users.xml into a lookup by user ID."""
    users: dict[int, dict] = {}
    users_path = extracted_dir / "Users.xml"
    if not users_path.exists():
        return users
    for elem in read(users_path, xml_tag="row"):
        attrs = _row_to_dict(elem)
        uid = attrs.get("Id")
        if uid is None:
            continue
        users[int(uid)] = {
            "display_name": attrs.get("DisplayName", ""),
            "reputation": int(attrs.get("Reputation", 0)),
        }
    return users


def _build_comments_index(extracted_dir: Path) -> dict[int, list[dict]]:
    """Parse Comments.xml into a lookup by post ID, sorted by creation date."""
    comments: dict[int, list[dict]] = defaultdict(list)
    comments_path = extracted_dir / "Comments.xml"
    if not comments_path.exists():
        return comments
    for elem in read(comments_path, xml_tag="row"):
        attrs = _row_to_dict(elem)
        post_id = attrs.get("PostId")
        if post_id is None:
            continue
        user_id = attrs.get("UserId")
        comments[int(post_id)].append({
            "id": int(attrs["Id"]),
            "text": attrs.get("Text", ""),
            "score": int(attrs.get("Score", 0)),
            "creation_date": attrs.get("CreationDate"),
            "user_id": int(user_id) if user_id else None,
        })
    # sort each post's comments by creation date
    for post_id in comments:
        comments[post_id].sort(key=lambda c: c.get("creation_date") or "")
    return comments


def _build_answers_index(extracted_dir: Path) -> dict[int, list[dict]]:
    """Parse Posts.xml collecting answers (PostTypeId=2), keyed by parent question ID.

    Sorted by score descending.
    """
    answers: dict[int, list[dict]] = defaultdict(list)
    posts_path = extracted_dir / "Posts.xml"
    for elem in read(posts_path, xml_tag="row"):
        attrs = _row_to_dict(elem)
        if attrs.get("PostTypeId") != "2":
            continue
        parent_id = attrs.get("ParentId")
        if parent_id is None:
            continue
        answers[int(parent_id)].append({
            "id": int(attrs["Id"]),
            "body_html": attrs.get("Body", ""),
            "creation_date": attrs.get("CreationDate"),
            "last_activity_date": attrs.get("LastActivityDate"),
            "score": int(attrs.get("Score", 0)),
            "owner_user_id": int(attrs["OwnerUserId"]) if attrs.get("OwnerUserId") else None,
            "comment_count": int(attrs.get("CommentCount", 0)),
        })
    # sort by score descending
    for qid in answers:
        answers[qid].sort(key=lambda a: a["score"], reverse=True)
    return answers


def _resolve_owner(owner_user_id: int | None, users: dict[int, dict]) -> dict | None:
    """Look up owner info, returning None if user ID is missing or not found."""
    if owner_user_id is None:
        return None
    user = users.get(owner_user_id)
    if user is None:
        return None
    return {"id": owner_user_id, **user}


def _assemble_question_record(
    attrs: dict,
    answers_index: dict[int, list[dict]],
    comments_index: dict[int, list[dict]],
    users: dict[int, dict],
) -> dict:
    """Build the full question record with answers, comments, and owner info."""
    qid = int(attrs["Id"])
    owner_user_id = int(attrs["OwnerUserId"]) if attrs.get("OwnerUserId") else None
    accepted_answer_id = int(attrs["AcceptedAnswerId"]) if attrs.get("AcceptedAnswerId") else None

    question = {
        "id": qid,
        "title": attrs.get("Title", ""),
        "body_html": attrs.get("Body", ""),
        "body_text": _strip_html(attrs.get("Body", "")),
        "creation_date": attrs.get("CreationDate"),
        "last_activity_date": attrs.get("LastActivityDate"),
        "score": int(attrs.get("Score", 0)),
        "view_count": int(attrs.get("ViewCount", 0)),
        "answer_count": int(attrs.get("AnswerCount", 0)),
        "accepted_answer_id": accepted_answer_id,
        "tags": _parse_tag_string(attrs.get("Tags", "")),
        "owner": _resolve_owner(owner_user_id, users),
        "comments": comments_index.get(qid, []),
    }

    raw_answers = answers_index.get(qid, [])
    assembled_answers = []
    for ans in raw_answers:
        assembled_answers.append({
            "id": ans["id"],
            "body_html": ans["body_html"],
            "body_text": _strip_html(ans["body_html"]),
            "creation_date": ans["creation_date"],
            "score": ans["score"],
            "is_accepted": ans["id"] == accepted_answer_id if accepted_answer_id else False,
            "owner": _resolve_owner(ans.get("owner_user_id"), users),
            "comments": comments_index.get(ans["id"], []),
        })

    return {"question": question, "answers": assembled_answers}


class StackExchangeInfosecConnector:
    source_id = "stackexchange-infosec"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Yield one assembled question record per question in the SE data dump.

        ``path`` is the extracted/ directory containing Posts.xml, Users.xml,
        and Comments.xml.
        """
        # Pass 1: lookup tables
        users = _build_users_index(path)
        comments = _build_comments_index(path)

        # Pass 2: index answers
        answers = _build_answers_index(path)

        # Pass 3: emit questions
        posts_path = path / "Posts.xml"
        for elem in read(posts_path, xml_tag="row"):
            attrs = _row_to_dict(elem)
            if attrs.get("PostTypeId") != "1":
                continue
            # skip deleted questions
            if attrs.get("DeletionDate") or attrs.get("Deleted"):
                continue
            yield _assemble_question_record(attrs, answers, comments, users)

    def normalize(self, record: dict) -> NormalizedData:
        """Convert an assembled question record into the canonical schema."""
        q = record["question"]
        qid = q["id"]

        # Build content: question text + all answers in score order
        parts = [f"# {q['title']}\n\n{q['body_text']}"]
        for ans in record["answers"]:
            header = "## Answer"
            if ans.get("is_accepted"):
                header = "## Answer (Accepted)"
            header += f" (Score: {ans['score']})"
            parts.append(f"{header}\n\n{ans['body_text']}")
        content = "\n\n---\n\n".join(parts)

        published_at = _parse_datetime(q.get("creation_date"))
        modified_at = _parse_datetime(q.get("last_activity_date")) or published_at

        return NormalizedData(
            record_id=f"stackexchange-infosec:question-{qid}",
            source_id=self.source_id,
            source_record_id=f"question-{qid}",
            content=content,
            title=q["title"],
            raw=record,
            ingested_at=datetime.now(timezone.utc),
            published_at=published_at,
            modified_at=modified_at,
            source_url=f"https://security.stackexchange.com/questions/{qid}",
            language="en",
        )
