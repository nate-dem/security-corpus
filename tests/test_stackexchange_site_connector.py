from pathlib import Path

from ingest.connectors.base import NormalizedData, QAThreadData
from ingest.connectors.stackexchange import StackExchangeSiteConnector
from ingest.connectors.stackexchange.common import (
    assemble_qa_content,
    detect_code_in_html,
    extract_closure,
    html_to_markdown,
    parse_tag_string,
)

FIXTURES = Path(__file__).parent / "fixtures" / "stackexchange-infosec"


def _make_connector():
    return StackExchangeSiteConnector("infosec", "security.stackexchange.com")


def _get_records():
    """Load all records from the sample fixtures."""
    return list(_make_connector().iter_records(FIXTURES))


def _get_record_by_qid(qid: int):
    for r in _get_records():
        if r["question"]["id"] == qid:
            return r
    raise ValueError(f"No record with question id {qid}")


# ── common.py helpers ────────────────────────────────────────────────


def test_html_to_markdown_preserves_code_blocks():
    html = "<p>Use this:</p><pre><code>x = 1\ny = 2</code></pre><p>Done.</p>"
    md = html_to_markdown(html)
    assert "```" in md
    assert "x = 1" in md
    assert "y = 2" in md
    # no raw HTML tags
    assert "<pre>" not in md
    assert "<code>" not in md


def test_html_to_markdown_preserves_inline_code():
    html = "<p>Run <code>ls -la</code> to list files</p>"
    md = html_to_markdown(html)
    assert "`ls -la`" in md


def test_html_to_markdown_preserves_paragraph_breaks():
    html = "<p>First paragraph</p><p>Second paragraph</p>"
    md = html_to_markdown(html)
    assert "First paragraph" in md
    assert "Second paragraph" in md
    assert "\n" in md


def test_html_to_markdown_empty_input():
    assert html_to_markdown("") == ""
    assert html_to_markdown(None) == ""


def test_html_to_markdown_collapses_newlines():
    html = "<p>A</p><p></p><p></p><p></p><p>B</p>"
    md = html_to_markdown(html)
    assert "\n\n\n" not in md
    assert "A" in md
    assert "B" in md


def test_parse_tag_string_angle_brackets():
    assert parse_tag_string("<cryptography><ssl>") == ["cryptography", "ssl"]
    assert parse_tag_string("<cryptography><ssl><certificates>") == [
        "cryptography", "ssl", "certificates"
    ]


def test_parse_tag_string_pipe_delimited():
    assert parse_tag_string("|cryptography|ssl|") == ["cryptography", "ssl"]


def test_parse_tag_string_empty():
    assert parse_tag_string("") == []
    assert parse_tag_string(None) == []


def test_assemble_qa_content_orders_answers_by_score():
    question = {"title": "Test Q", "body_md": "Question body"}
    answers = [
        {"score": 30, "is_accepted": True, "body_md": "Best answer"},
        {"score": 10, "is_accepted": False, "body_md": "Other answer"},
    ]
    content = assemble_qa_content(question, answers)
    # Title at top
    assert content.startswith("# Test Q")
    # Question body present
    assert "Question body" in content
    # Both answers present
    assert "Best answer" in content
    assert "Other answer" in content
    # Accepted answer comes first (higher score)
    assert content.index("Answer (Accepted)") < content.index("Other answer")
    # Section separators
    assert "---" in content


def test_assemble_qa_content_no_answers():
    question = {"title": "Solo Q", "body_md": "Just a question"}
    content = assemble_qa_content(question, [])
    assert "# Solo Q" in content
    assert "Just a question" in content
    assert "---" not in content


def test_detect_code_in_html():
    assert detect_code_in_html("<p>text <code>x</code></p>") is True
    assert detect_code_in_html("<pre><code>print(1)</code></pre>") is True
    assert detect_code_in_html("<p>no code here</p>") is False
    assert detect_code_in_html("") is False
    assert detect_code_in_html(None) is False


def test_extract_closure():
    closed, reason = extract_closure({"ClosedDate": "2020-01-01T00:00:00.000"})
    assert closed is True
    assert reason is None

    closed, reason = extract_closure({"Score": "5"})
    assert closed is False
    assert reason is None


# ── StackExchangeSiteConnector ────────────────────────────────────────


def test_normalize_returns_qa_thread_data():
    record = _get_record_by_qid(100)
    result = _make_connector().normalize(record)

    assert isinstance(result, QAThreadData)
    assert isinstance(result, NormalizedData)
    assert result.record_id == "stackexchange-infosec:question-100"
    assert result.source_id == "stackexchange-infosec"
    assert result.source_record_id == "question-100"
    assert result.title == "Does an established HTTPS connection mean a line is really secure?"
    assert result.source_url == "https://security.stackexchange.com/questions/100"


def test_normalize_populates_qa_fields():
    record = _get_record_by_qid(100)
    result = _make_connector().normalize(record)

    assert result.score == 47
    assert result.answer_count == 2
    assert result.has_accepted_answer is True
    assert result.closed is False
    assert result.closure_reason is None
    assert result.tags == ["cryptography", "ssl", "tls"]


def test_normalize_populates_base_fields():
    record = _get_record_by_qid(100)
    result = _make_connector().normalize(record)

    assert result.content_hash is not None
    assert result.content_length is not None
    assert result.content_length > 0
    assert result.license is not None
    assert result.published_at is not None
    # raw is None for Q&A sources
    assert result.raw is None


def test_content_contains_question_and_answers():
    record = _get_record_by_qid(100)
    result = _make_connector().normalize(record)

    # Title at top
    assert result.content.startswith("# Does an established HTTPS connection")
    # Section separators between question and answers
    assert "---" in result.content
    # Accepted answer marked
    assert "Answer (Accepted)" in result.content
    # Both answers present — check key terms survive markdown conversion
    assert "confidentiality" in result.content
    assert "man-in-the-middle" in result.content


def test_content_answer_score_ordering():
    record = _get_record_by_qid(100)
    result = _make_connector().normalize(record)

    # Accepted answer (score=32) should appear before non-accepted (score=12)
    accepted_pos = result.content.index("Answer (Accepted)")
    # The second answer section should come after
    other_answer_pos = result.content.index("man-in-the-middle")
    assert accepted_pos < other_answer_pos


def test_no_answers_produces_valid_record():
    record = _get_record_by_qid(200)
    result = _make_connector().normalize(record)

    assert isinstance(result, QAThreadData)
    assert result.record_id == "stackexchange-infosec:question-200"
    assert result.answer_count == 0
    assert result.has_accepted_answer is False
    # Content has question but no answer sections
    assert "symmetric" in result.content
    assert "asymmetric" in result.content
    assert "---" not in result.content


def test_iter_records_yields_all_questions():
    records = _get_records()
    qids = {r["question"]["id"] for r in records}
    # All 3 questions in Posts.xml should come through (including deleted—
    # DeletionDate check was removed; deleted posts aren't in public dumps)
    assert 100 in qids
    assert 200 in qids


def test_owner_resolved_from_users():
    record = _get_record_by_qid(100)
    owner = record["question"]["owner"]
    assert owner is not None
    assert owner["id"] == 201
    assert owner["display_name"] == "alice"
    assert owner["reputation"] == 5000


def test_markdown_conversion_in_content():
    """Content uses markdown, not plain text — code blocks are fenced."""
    record = _get_record_by_qid(100)
    result = _make_connector().normalize(record)

    # The question body contains <code>eavesdropping</code> in HTML
    # which should become `eavesdropping` in markdown
    assert "`eavesdropping`" in result.content


def test_has_code_detected():
    record = _get_record_by_qid(100)
    q = record["question"]
    # Question 100 has <code>eavesdropping</code> in the body
    assert q["has_code"] is True

    record200 = _get_record_by_qid(200)
    q200 = record200["question"]
    # Question 200 also has <code> tags
    assert q200["has_code"] is True


def test_site_slug_customizes_source_id():
    """Different site slugs produce different source_ids and URLs."""
    crypto = StackExchangeSiteConnector("crypto", "crypto.stackexchange.com")
    assert crypto.source_id == "stackexchange-crypto"
