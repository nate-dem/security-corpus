from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ingest.connectors.stackexchange import stackoverflow
from ingest.connectors.stackexchange.stackoverflow import (
    StackOverflowConnector,
    _iter_post_elements,
)


FIXTURES = Path(__file__).parent / "fixtures" / "stackexchange-infosec"
POSTS_XML = FIXTURES / "Posts.xml"

# Dummy archive path — never actually used due to monkeypatching
DUMMY_ARCHIVE = Path("/dev/null")

TEST_TAGS = {"cryptography", "ssl"}


@contextmanager
def _mock_posts_stream(archive_path):
    """Substitute for _open_posts_stream that reads from the fixture file."""
    yield _iter_post_elements(POSTS_XML)


@pytest.fixture
def so_env(monkeypatch, tmp_path):
    """Set up monkeypatched environment for SO connector tests."""
    monkeypatch.setattr(stackoverflow, "_open_posts_stream", _mock_posts_stream)
    monkeypatch.setattr(stackoverflow, "SECURITY_TAGS", TEST_TAGS)
    return tmp_path


# ── _iter_post_elements ───────────────────────────────────────────


def test_iter_post_elements_yields_all_rows():
    elements = list(_iter_post_elements(POSTS_XML))
    # Fixture has 5 rows: questions 100, 200, 300 and answers 101, 102
    assert len(elements) == 5


def test_iter_post_elements_yields_attr_dicts():
    first = next(iter(_iter_post_elements(POSTS_XML)))
    assert isinstance(first, dict)
    assert "Id" in first
    assert "PostTypeId" in first


# ── Pass 1: _collect_question_ids ─────────────────────────────────


def test_collect_question_ids_returns_tagged_only(so_env):
    connector = StackOverflowConnector()
    ids = connector._collect_question_ids(DUMMY_ARCHIVE, so_env)
    # Questions 100 (cryptography, ssl, tls) and 200 (cryptography, encryption)
    # both have "cryptography" which is in TEST_TAGS
    assert 100 in ids
    assert 200 in ids
    # Question 300 (tags: spam) should not be included
    assert 300 not in ids


def test_collect_question_ids_persists_to_parquet(so_env):
    connector = StackOverflowConnector()
    connector._collect_question_ids(DUMMY_ARCHIVE, so_env)

    ids_path = so_env / "question_ids.parquet"
    assert ids_path.exists()
    table = pq.read_table(ids_path)
    stored_ids = set(table.column("question_id").to_pylist())
    assert stored_ids == {100, 200}


def test_collect_question_ids_writes_done_marker(so_env):
    connector = StackOverflowConnector()
    connector._collect_question_ids(DUMMY_ARCHIVE, so_env)
    assert (so_env / "_question_ids.DONE").exists()


def test_collect_question_ids_resumes_from_cache(so_env):
    connector = StackOverflowConnector()
    ids1 = connector._collect_question_ids(DUMMY_ARCHIVE, so_env)
    ids2 = connector._collect_question_ids(DUMMY_ARCHIVE, so_env)
    assert ids1 == ids2


# ── Pass 2: _write_answer_index ───────────────────────────────────


def test_write_answer_index_creates_parquet(so_env):
    connector = StackOverflowConnector()
    question_ids = {100, 200}

    count = connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, so_env, batch_size=10
    )

    # Answers 101 and 102 are for question 100
    assert count == 2

    answers_dir = so_env / "answers"
    parquet_files = list(answers_dir.glob("batch_*.parquet"))
    assert len(parquet_files) >= 1


def test_write_answer_index_contains_expected_data(so_env):
    connector = StackOverflowConnector()
    question_ids = {100, 200}

    connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, so_env, batch_size=10
    )

    # Read the intermediate parquet and verify contents
    answers_dir = so_env / "answers"
    table = pq.read_table(answers_dir, filters=None)
    rows = table.to_pydict()

    ids = rows["id"]
    parent_ids = rows["parent_id"]
    scores = rows["score"]

    assert sorted(ids) == [101, 102]
    assert all(pid == 100 for pid in parent_ids)
    assert set(scores) == {12, 32}


def test_write_answer_index_writes_done_marker(so_env):
    connector = StackOverflowConnector()
    connector._write_answer_index(
        DUMMY_ARCHIVE, {100}, so_env, batch_size=10
    )
    assert (so_env / "answers" / "_DONE").exists()


def test_write_answer_index_restarts_on_interrupted(so_env):
    connector = StackOverflowConnector()

    # Create a partial answers dir without _DONE marker
    answers_dir = so_env / "answers"
    answers_dir.mkdir(parents=True)
    (answers_dir / "batch_000000.parquet").write_text("garbage")

    # Should clean up and restart
    count = connector._write_answer_index(
        DUMMY_ARCHIVE, {100}, so_env, batch_size=10
    )
    assert count == 2
    assert (answers_dir / "_DONE").exists()


def test_write_answer_index_skips_when_done(so_env):
    connector = StackOverflowConnector()
    question_ids = {100}

    # First run
    count1 = connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, so_env, batch_size=10
    )
    # Second run should skip
    count2 = connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, so_env, batch_size=10
    )
    assert count1 == count2


# ── Pass 3: _assemble_and_write ──────────────────────────────────


def test_assemble_and_write_produces_records(so_env):
    connector = StackOverflowConnector()
    question_ids = {100, 200}
    intermediate = so_env / "intermediate"
    output = so_env / "output"

    connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, intermediate, batch_size=10
    )
    count = connector._assemble_and_write(
        DUMMY_ARCHIVE, question_ids, intermediate, output, batch_size=10
    )

    assert count == 2

    output_dir = output / "source_id=stackoverflow"
    parquet_files = list(output_dir.glob("batch_*.parquet"))
    assert len(parquet_files) >= 1


def test_assemble_produces_valid_qa_fields(so_env):
    connector = StackOverflowConnector()
    question_ids = {100}
    intermediate = so_env / "intermediate"
    output = so_env / "output"

    connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, intermediate, batch_size=10
    )
    connector._assemble_and_write(
        DUMMY_ARCHIVE, question_ids, intermediate, output, batch_size=10
    )

    table = pq.read_table(output / "source_id=stackoverflow")
    assert table.num_rows == 1
    row = table.to_pydict()

    assert row["record_id"][0] == "stackoverflow:question-100"
    assert row["source_id"][0] == "stackoverflow"
    assert row["source_record_id"][0] == "question-100"
    assert row["title"][0] == "Does an established HTTPS connection mean a line is really secure?"
    assert row["source_url"][0] == "https://stackoverflow.com/questions/100"
    assert row["license"][0] == "CC-BY-SA-4.0"
    assert row["content_hash"][0] is not None
    assert row["content_length"][0] > 0
    assert row["score"][0] == 47
    assert row["answer_count"][0] == 2
    assert row["has_accepted_answer"][0] is True
    assert row["closed"][0] is False
    assert row["raw"][0] is None


def test_assemble_content_includes_answers(so_env):
    connector = StackOverflowConnector()
    question_ids = {100}
    intermediate = so_env / "intermediate"
    output = so_env / "output"

    connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, intermediate, batch_size=10
    )
    connector._assemble_and_write(
        DUMMY_ARCHIVE, question_ids, intermediate, output, batch_size=10
    )

    table = pq.read_table(output / "source_id=stackoverflow")
    content = table.column("content")[0].as_py()

    assert content.startswith("# Does an established HTTPS connection")
    assert "---" in content
    assert "Answer (Accepted)" in content
    # From accepted answer 102 (score=32)
    assert "confidentiality" in content
    # From answer 101 (score=12)
    assert "man-in-the-middle" in content


def test_assemble_answer_ordering(so_env):
    connector = StackOverflowConnector()
    question_ids = {100}
    intermediate = so_env / "intermediate"
    output = so_env / "output"

    connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, intermediate, batch_size=10
    )
    connector._assemble_and_write(
        DUMMY_ARCHIVE, question_ids, intermediate, output, batch_size=10
    )

    table = pq.read_table(output / "source_id=stackoverflow")
    content = table.column("content")[0].as_py()

    # Accepted answer (score=32) should appear before other answer (score=12)
    accepted_pos = content.index("Answer (Accepted)")
    other_pos = content.index("man-in-the-middle")
    assert accepted_pos < other_pos


def test_assemble_handles_question_without_answers(so_env):
    connector = StackOverflowConnector()
    question_ids = {200}  # Question 200 has no answers
    intermediate = so_env / "intermediate"
    output = so_env / "output"

    connector._write_answer_index(
        DUMMY_ARCHIVE, question_ids, intermediate, batch_size=10
    )
    count = connector._assemble_and_write(
        DUMMY_ARCHIVE, question_ids, intermediate, output, batch_size=10
    )

    assert count == 1
    table = pq.read_table(output / "source_id=stackoverflow")
    row = table.to_pydict()

    assert row["record_id"][0] == "stackoverflow:question-200"
    assert row["answer_count"][0] == 0
    assert row["has_accepted_answer"][0] is False
    content = row["content"][0]
    assert "symmetric" in content
    assert "---" not in content


# ── Full ingest ──────────────────────────────────────────────────


def test_ingest_raises_on_empty_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(stackoverflow, "SECURITY_TAGS", set())
    connector = StackOverflowConnector()

    with pytest.raises(ValueError, match="SECURITY_TAGS is empty"):
        connector.ingest(DUMMY_ARCHIVE, tmp_path / "out", tmp_path / "inter")


def test_full_ingest_end_to_end(so_env):
    connector = StackOverflowConnector()

    count = connector.ingest(
        archive_path=DUMMY_ARCHIVE,
        output_dir=so_env / "output",
        intermediate_dir=so_env / "intermediate",
        batch_size=10,
    )

    # Questions 100 and 200 match TEST_TAGS; 300 does not
    assert count == 2

    # Verify output files exist
    output_dir = so_env / "output" / "source_id=stackoverflow"
    assert any(output_dir.glob("batch_*.parquet"))

    # Verify intermediate markers
    intermediate = so_env / "intermediate"
    assert (intermediate / "_question_ids.DONE").exists()
    assert (intermediate / "answers" / "_DONE").exists()
