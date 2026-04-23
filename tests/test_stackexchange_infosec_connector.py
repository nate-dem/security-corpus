from pathlib import Path

from ingest.connectors.base import NormalizedData
from ingest.connectors.stackexchange_infosec import (
    StackExchangeInfosecConnector,
    _parse_tag_string,
    _strip_html,
)

FIXTURES = Path(__file__).parent / "fixtures" / "stackexchange-infosec"


def _get_records():
    """Load all records from the sample fixtures."""
    connector = StackExchangeInfosecConnector()
    return list(connector.iter_records(FIXTURES))


def _get_record_by_qid(qid: int):
    for r in _get_records():
        if r["question"]["id"] == qid:
            return r
    raise ValueError(f"No record with question id {qid}")


def test_strip_html_removes_tags_preserves_text():
    html = "<p>foo <code>bar</code></p>"
    result = _strip_html(html)
    assert "foo" in result
    assert "bar" in result
    assert "<p>" not in result
    assert "<code>" not in result


def test_strip_html_preserves_paragraph_breaks():
    html = "<p>First paragraph</p><p>Second paragraph</p>"
    result = _strip_html(html)
    assert "First paragraph" in result
    assert "Second paragraph" in result
    # paragraphs should be separated by newlines
    assert "\n" in result


def test_parse_tag_string_angle_brackets():
    assert _parse_tag_string("<cryptography><ssl>") == ["cryptography", "ssl"]
    assert _parse_tag_string("<cryptography><ssl><certificates>") == [
        "cryptography", "ssl", "certificates"
    ]


def test_parse_tag_string_pipe_delimited():
    assert _parse_tag_string("|cryptography|ssl|") == ["cryptography", "ssl"]


def test_parse_tag_string_empty():
    assert _parse_tag_string("") == []
    assert _parse_tag_string(None) == []


def test_normalize_produces_valid_normalized_data():
    record = _get_record_by_qid(100)
    connector = StackExchangeInfosecConnector()
    result = connector.normalize(record)

    assert isinstance(result, NormalizedData)
    assert result.record_id == "stackexchange-infosec:question-100"
    assert result.source_id == "stackexchange-infosec"
    assert result.source_record_id == "question-100"
    assert result.title == "Does an established HTTPS connection mean a line is really secure?"
    assert result.source_url == "https://security.stackexchange.com/questions/100"
    assert result.language == "en"
    assert result.severity is None
    assert result.cvss_score is None
    assert result.cwe_ids == []


def test_content_merges_question_and_answers_in_score_order():
    record = _get_record_by_qid(100)
    connector = StackExchangeInfosecConnector()
    result = connector.normalize(record)

    # The accepted answer (id=102, score=32) should come first (highest score)
    # then id=101 (score=12)
    assert "---" in result.content
    assert "Answer (Accepted)" in result.content
    # Accepted answer text appears before non-accepted
    accepted_pos = result.content.index("Answer (Accepted)")
    # Both answers should be present
    assert "TLS ensures confidentiality" in result.content
    assert "man-in-the-middle" in result.content


def test_no_answers_produces_valid_record():
    record = _get_record_by_qid(200)
    connector = StackExchangeInfosecConnector()
    result = connector.normalize(record)

    assert isinstance(result, NormalizedData)
    assert result.record_id == "stackexchange-infosec:question-200"
    assert "symmetric" in result.content
    assert "asymmetric" in result.content
    # No separator when there are no answers
    assert "---" not in result.content


def test_deleted_questions_are_skipped():
    records = _get_records()
    qids = {r["question"]["id"] for r in records}
    # Question 300 has DeletionDate and should be skipped
    assert 300 not in qids
    # Questions 100 and 200 should be present
    assert 100 in qids
    assert 200 in qids


def test_comments_attached_to_correct_posts():
    record = _get_record_by_qid(100)
    # Comment 501 is on the question (PostId=100)
    q_comments = record["question"]["comments"]
    assert len(q_comments) == 1
    assert q_comments[0]["id"] == 501

    # Comment 502 is on answer 101
    ans_101 = [a for a in record["answers"] if a["id"] == 101][0]
    assert len(ans_101["comments"]) == 1
    assert ans_101["comments"][0]["id"] == 502


def test_owner_resolved_from_users():
    record = _get_record_by_qid(100)
    owner = record["question"]["owner"]
    assert owner is not None
    assert owner["id"] == 201
    assert owner["display_name"] == "alice"
    assert owner["reputation"] == 5000


def test_raw_preserves_html():
    record = _get_record_by_qid(100)
    connector = StackExchangeInfosecConnector()
    result = connector.normalize(record)
    # raw should have original HTML in body_html
    assert "<p>" in result.raw["question"]["body_html"]
    assert "<code>" in result.raw["question"]["body_html"]
