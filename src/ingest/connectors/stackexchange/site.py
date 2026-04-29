"""Stack Exchange site connector for small-to-medium site dumps.

Handles sites where the full answer index fits in memory (e.g., Infosec, ReverseEngineering, etc)

For Stack Overflow, use a separate streaming connector (not implemented here).

In-memory two-pass approach:
  Pass 1: Index all answers (PostTypeId=2) from Posts.xml
  Pass 2: Stream questions (PostTypeId=1) from Posts.xml, emitting assembled records
"""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import QAThreadData
from ingest.connectors.stackexchange.common import (
    assemble_qa_content,
    detect_code_in_html,
    extract_closure,
    html_to_markdown,
    parse_se_datetime,
    parse_tag_string,
)
from ingest.readers import read
from ingest.utils import CC_BY_SA_4_0, compute_content_hash, compute_token_count


class StackExchangeSiteConnector:
    """Handles small Stack Exchange site dumps (InfoSec, RE, Crypto, Tor).

    Each site dump is one .7z extracted to a directory containing Posts.xml.
    This connector indexes all answers in memory and streams questions —
    works for sites under ~1M questions.
    """

    def __init__(self, site_slug: str, site_domain: str):
        self.site_slug = site_slug
        self.site_domain = site_domain
        self.source_id = f"stackexchange-{site_slug}"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Yield one assembled question record per question in the SE data dump.

        ``path`` is the extracted directory containing Posts.xml
        """
        # Pass 1: index answers (PostTypeId=2)
        answers = _build_answers_index(path)

        # Pass 2: stream questions (PostTypeId=1), emit assembled records
        posts_path = path / "Posts.xml"
        for elem in read(posts_path, xml_tag="row"):
            attrs = dict(elem.attrib)
            if attrs.get("PostTypeId") != "1":
                continue
            yield _assemble_question_record(attrs, answers)

    def normalize(self, record: dict) -> QAThreadData:
        """Convert an assembled question record into QAThreadData."""
        q = record["question"]
        qid = q["id"]

        content = assemble_qa_content(q, record["answers"])

        return QAThreadData(
            record_id=f"{self.source_id}:question-{qid}",
            source_id=self.source_id,
            source_record_id=f"question-{qid}",
            content=content,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            title=q["title"],
            ingested_at=datetime.now(timezone.utc),
            license=CC_BY_SA_4_0,
            published_at=parse_se_datetime(q.get("creation_date")),
            source_url=f"https://{self.site_domain}/questions/{qid}",
            # QA-specific fields
            score=q.get("score"),
            answer_count=len(record["answers"]),
            has_accepted_answer=q.get("accepted_answer_id") is not None,
            closed=record.get("closed", False),
            tags=q.get("tags", []),
        )


def _build_answers_index(extracted_dir: Path) -> dict[int, list[dict]]:
    """Parse Posts.xml collecting answers (PostTypeId=2), keyed by parent question ID.

    Sorted by score descending.
    """
    answers: dict[int, list[dict]] = defaultdict(list)
    posts_path = extracted_dir / "Posts.xml"
    for elem in read(posts_path, xml_tag="row"):
        attrs = dict(elem.attrib)
        if attrs.get("PostTypeId") != "2":
            continue
        parent_id = attrs.get("ParentId")
        if parent_id is None:
            continue
        answers[int(parent_id)].append({
            "id": int(attrs["Id"]),
            "body_html": attrs.get("Body", ""),
            "score": int(attrs.get("Score", 0)),
        })
    # sort by score descending
    for qid in answers:
        answers[qid].sort(key=lambda a: a["score"], reverse=True)
    return answers


def _assemble_question_record(
    attrs: dict,
    answers_index: dict[int, list[dict]],
) -> dict:
    """Build the full question record with answers."""
    qid = int(attrs["Id"])
    accepted_answer_id = int(attrs["AcceptedAnswerId"]) if attrs.get("AcceptedAnswerId") else None

    body_html = attrs.get("Body", "")
    closed = extract_closure(attrs)

    question = {
        "id": qid,
        "title": attrs.get("Title", ""),
        "body_html": body_html,
        "body_md": html_to_markdown(body_html),
        "creation_date": attrs.get("CreationDate"),
        "score": int(attrs.get("Score", 0)),
        "answer_count": int(attrs.get("AnswerCount", 0)),
        "accepted_answer_id": accepted_answer_id,
        "tags": parse_tag_string(attrs.get("Tags", "")),
        "has_code": detect_code_in_html(body_html),
    }

    raw_answers = answers_index.get(qid, [])
    assembled_answers = []
    for ans in raw_answers:
        ans_html = ans["body_html"]
        assembled_answers.append({
            "id": ans["id"],
            "body_html": ans_html,
            "body_md": html_to_markdown(ans_html),
            "score": ans["score"],
            "is_accepted": ans["id"] == accepted_answer_id if accepted_answer_id else False,
            "has_code": detect_code_in_html(ans_html),
        })

    return {
        "question": question,
        "answers": assembled_answers,
        "closed": closed,
    }
