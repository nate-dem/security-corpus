import json
from datetime import timezone
from pathlib import Path

import zstandard as zstd

from ingest.connectors.base import NormalizedData, QAThreadData
from ingest.connectors.reddit import RedditSubredditConnector
from ingest.connectors.reddit.common import (
    assemble_thread_content,
    build_comment_tree,
    clean_selftext,
    is_deleted_or_removed,
    parse_reddit_timestamp,
)


# ── fixture helpers ─────────────────────────────────────────────────


def _write_zst_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as zstandard-compressed JSONL."""
    cctx = zstd.ZstdCompressor()
    lines = "\n".join(json.dumps(r) for r in records) + "\n"
    path.write_bytes(cctx.compress(lines.encode("utf-8")))


SUBMISSIONS = [
    {
        "id": "abc123",
        "title": "How to configure iptables for a home network?",
        "selftext": "I want to set up firewall rules.\n\n\n\n\nAny tips?",
        "score": 50,
        "num_comments": 3,
        "subreddit": "netsec",
        "created_utc": 1700000000,
        "permalink": "/r/netsec/comments/abc123/how_to_configure_iptables/",
        "is_self": True,
        "over_18": False,
        "link_flair_text": "Question",
    },
    {
        "id": "def456",
        "title": "Post with no comments",
        "selftext": "Just a question with no replies yet.",
        "score": 5,
        "num_comments": 0,
        "subreddit": "netsec",
        "created_utc": 1700100000,
        "permalink": "/r/netsec/comments/def456/post_with_no_comments/",
        "is_self": True,
        "over_18": False,
    },
    {
        "id": "del789",
        "title": "Deleted post",
        "selftext": "[deleted]",
        "score": 0,
        "subreddit": "netsec",
        "created_utc": 1700200000,
        "permalink": "/r/netsec/comments/del789/deleted_post/",
        "is_self": True,
    },
    {
        "id": "rem012",
        "title": "Removed post",
        "selftext": "[removed]",
        "score": 0,
        "subreddit": "netsec",
        "created_utc": 1700300000,
        "permalink": "/r/netsec/comments/rem012/removed_post/",
        "is_self": True,
    },
    {
        "id": "link345",
        "title": "Check out this article",
        "selftext": "",
        "score": 20,
        "subreddit": "netsec",
        "created_utc": 1700400000,
        "permalink": "/r/netsec/comments/link345/check_out_this_article/",
        "url": "https://example.com/article",
        "is_self": False,
    },
    {
        "id": "nsfw678",
        "title": "NSFW post",
        "selftext": "Should be filtered out.",
        "score": 10,
        "subreddit": "netsec",
        "created_utc": 1700500000,
        "permalink": "/r/netsec/comments/nsfw678/nsfw_post/",
        "is_self": True,
        "over_18": True,
    },
    {
        "id": "modrem9",
        "title": "Mod-removed post",
        "selftext": "This was removed by mods.",
        "score": 10,
        "subreddit": "netsec",
        "created_utc": 1700600000,
        "permalink": "/r/netsec/comments/modrem9/mod_removed_post/",
        "is_self": True,
        "removed_by_category": "moderator",
    },
]

COMMENTS = [
    # Top-level comment on abc123
    {
        "id": "c001",
        "body": "Use `ufw` instead, it's much simpler.",
        "score": 30,
        "parent_id": "t3_abc123",
        "link_id": "t3_abc123",
        "subreddit": "netsec",
        "created_utc": 1700001000,
    },
    # Reply to c001
    {
        "id": "c002",
        "body": "Agreed, ufw is great for beginners.",
        "score": 15,
        "parent_id": "t1_c001",
        "link_id": "t3_abc123",
        "subreddit": "netsec",
        "created_utc": 1700002000,
    },
    # Another top-level comment on abc123 (lower score)
    {
        "id": "c003",
        "body": "Check out the Arch Wiki page on iptables.",
        "score": 10,
        "parent_id": "t3_abc123",
        "link_id": "t3_abc123",
        "subreddit": "netsec",
        "created_utc": 1700003000,
    },
    # Reply to c002 (depth=2 from top)
    {
        "id": "c004",
        "body": "Specifically `ufw allow ssh` is a good start.",
        "score": 8,
        "parent_id": "t1_c002",
        "link_id": "t3_abc123",
        "subreddit": "netsec",
        "created_utc": 1700004000,
    },
    # Deleted comment — should be filtered
    {
        "id": "c005",
        "body": "[deleted]",
        "score": 0,
        "parent_id": "t3_abc123",
        "link_id": "t3_abc123",
        "subreddit": "netsec",
    },
    # Removed comment — should be filtered
    {
        "id": "c006",
        "body": "[removed]",
        "score": 0,
        "parent_id": "t3_abc123",
        "link_id": "t3_abc123",
        "subreddit": "netsec",
    },
]


def _make_fixtures(tmp_path: Path) -> Path:
    """Create .zst fixture files in tmp_path, return the directory."""
    _write_zst_jsonl(tmp_path / "netsec_submissions.zst", SUBMISSIONS)
    _write_zst_jsonl(tmp_path / "netsec_comments.zst", COMMENTS)
    return tmp_path


def _make_connector():
    return RedditSubredditConnector("netsec")


def _get_records(tmp_path: Path):
    fixtures = _make_fixtures(tmp_path)
    return list(_make_connector().iter_records(fixtures))


def _get_record_by_sid(records: list[dict], sid: str):
    for r in records:
        if r["submission"]["id"] == sid:
            return r
    raise ValueError(f"No record with submission id {sid}")


# ── common.py helpers ───────────────────────────────────────────────


def test_parse_reddit_timestamp_returns_utc_datetime():
    dt = parse_reddit_timestamp(1700000000)
    assert dt is not None
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2023


def test_parse_reddit_timestamp_handles_float():
    dt = parse_reddit_timestamp(1700000000.5)
    assert dt is not None


def test_parse_reddit_timestamp_handles_none():
    assert parse_reddit_timestamp(None) is None


def test_parse_reddit_timestamp_handles_invalid():
    assert parse_reddit_timestamp("not_a_number") is None


def test_is_deleted_or_removed_markers():
    assert is_deleted_or_removed("[deleted]") is True
    assert is_deleted_or_removed("[removed]") is True
    assert is_deleted_or_removed(" [deleted] ") is True


def test_is_deleted_or_removed_empty():
    assert is_deleted_or_removed("") is True
    assert is_deleted_or_removed(None) is True


def test_is_deleted_or_removed_normal_text():
    assert is_deleted_or_removed("This is a real comment.") is False


def test_clean_selftext_collapses_newlines():
    result = clean_selftext("First line.\n\n\n\n\nSecond line.")
    assert "\n\n\n" not in result
    assert "First line." in result
    assert "Second line." in result


def test_clean_selftext_empty():
    assert clean_selftext("") == ""
    assert clean_selftext(None) == ""


def test_build_comment_tree_flat_comments():
    comments = [
        {"id": "a", "parent_id": "t3_post1", "body": "First", "score": 10},
        {"id": "b", "parent_id": "t3_post1", "body": "Second", "score": 20},
    ]
    tree = build_comment_tree(comments, "post1")
    assert len(tree) == 2
    # Sorted by score descending
    assert tree[0]["body"] == "Second"
    assert tree[1]["body"] == "First"
    assert tree[0]["children"] == []


def test_build_comment_tree_nested_replies():
    comments = [
        {"id": "a", "parent_id": "t3_post1", "body": "Top", "score": 10},
        {"id": "b", "parent_id": "t1_a", "body": "Reply to top", "score": 5},
        {"id": "c", "parent_id": "t1_b", "body": "Nested reply", "score": 3},
    ]
    tree = build_comment_tree(comments, "post1")
    assert len(tree) == 1
    assert tree[0]["body"] == "Top"
    assert len(tree[0]["children"]) == 1
    assert tree[0]["children"][0]["body"] == "Reply to top"
    assert len(tree[0]["children"][0]["children"]) == 1
    assert tree[0]["children"][0]["children"][0]["body"] == "Nested reply"


def test_build_comment_tree_sorts_at_each_level():
    comments = [
        {"id": "a", "parent_id": "t3_post1", "body": "Top", "score": 10},
        {"id": "b", "parent_id": "t1_a", "body": "Low reply", "score": 1},
        {"id": "c", "parent_id": "t1_a", "body": "High reply", "score": 20},
    ]
    tree = build_comment_tree(comments, "post1")
    children = tree[0]["children"]
    assert children[0]["body"] == "High reply"
    assert children[1]["body"] == "Low reply"


def test_build_comment_tree_drops_orphans():
    comments = [
        {"id": "a", "parent_id": "t3_post1", "body": "Valid", "score": 10},
        {"id": "b", "parent_id": "t1_missing", "body": "Orphan", "score": 5},
    ]
    tree = build_comment_tree(comments, "post1")
    assert len(tree) == 1
    assert tree[0]["body"] == "Valid"


def test_build_comment_tree_empty_input():
    tree = build_comment_tree([], "post1")
    assert tree == []


def test_assemble_thread_content_basic():
    tree = [
        {"body": "A comment.", "score": 10, "children": []},
    ]
    content = assemble_thread_content("Test Title", "Body text.", tree)
    assert content.startswith("# Test Title")
    assert "Body text." in content
    assert "---" in content
    assert "**Comment** (Score: 10)" in content
    assert "A comment." in content


def test_assemble_thread_content_no_comments():
    content = assemble_thread_content("Solo Post", "Just the body.", [])
    assert "# Solo Post" in content
    assert "Just the body." in content
    assert "---" not in content


def test_assemble_thread_content_nested_blockquotes():
    tree = [
        {
            "body": "Top comment.",
            "score": 10,
            "children": [
                {
                    "body": "A reply.",
                    "score": 5,
                    "children": [],
                },
            ],
        },
    ]
    content = assemble_thread_content("Post", "Body.", tree)
    assert "**Comment** (Score: 10)" in content
    assert "> **Reply** (Score: 5)" in content


def test_assemble_thread_content_depth_cap():
    # Build a chain 5 levels deep, cap at 2
    tree = [
        {
            "body": "Depth 0",
            "score": 10,
            "children": [
                {
                    "body": "Depth 1",
                    "score": 5,
                    "children": [
                        {
                            "body": "Depth 2 — should not appear",
                            "score": 3,
                            "children": [],
                        },
                    ],
                },
            ],
        },
    ]
    content = assemble_thread_content("Post", "Body.", tree, max_depth=2)
    assert "Depth 0" in content
    assert "Depth 1" in content
    assert "Depth 2" not in content


# ── RedditSubredditConnector ────────────────────────────────────────


def test_iter_records_filters_correctly(tmp_path):
    records = _get_records(tmp_path)
    sids = {r["submission"]["id"] for r in records}
    # Only abc123 and def456 should pass all filters
    assert "abc123" in sids
    assert "def456" in sids
    # These should all be filtered
    assert "del789" not in sids   # deleted
    assert "rem012" not in sids   # removed
    assert "link345" not in sids  # link post
    assert "nsfw678" not in sids  # NSFW
    assert "modrem9" not in sids  # mod-removed


def test_iter_records_count(tmp_path):
    records = _get_records(tmp_path)
    assert len(records) == 2


def test_iter_records_joins_comments(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    # 2 top-level comments (c001 score=30, c003 score=10)
    # c005 and c006 are deleted/removed and filtered
    assert r["top_level_comment_count"] == 2


def test_iter_records_no_comments_record(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "def456")
    assert r["top_level_comment_count"] == 0
    assert r["comment_tree"] == []


def test_iter_records_deleted_comments_excluded(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    # Flatten all comments in tree to check none are deleted
    all_bodies = _collect_bodies(r["comment_tree"])
    assert "[deleted]" not in all_bodies
    assert "[removed]" not in all_bodies


def test_iter_records_comment_tree_structure(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    tree = r["comment_tree"]
    # Top-level sorted by score: c001 (30) then c003 (10)
    assert tree[0]["body"] == "Use `ufw` instead, it's much simpler."
    assert tree[1]["body"] == "Check out the Arch Wiki page on iptables."
    # c001 has reply c002
    assert len(tree[0]["children"]) == 1
    assert tree[0]["children"][0]["body"] == "Agreed, ufw is great for beginners."
    # c002 has reply c004
    assert len(tree[0]["children"][0]["children"]) == 1
    assert tree[0]["children"][0]["children"][0]["body"] == "Specifically `ufw allow ssh` is a good start."


def test_iter_records_handles_missing_comments_file(tmp_path):
    # Only write submissions, no comments file
    _write_zst_jsonl(tmp_path / "netsec_submissions.zst", SUBMISSIONS)
    connector = _make_connector()
    records = list(connector.iter_records(tmp_path))
    # Should still yield valid submissions, just with no comments
    assert len(records) == 2
    for r in records:
        assert r["top_level_comment_count"] == 0


def test_normalize_returns_qa_thread_data(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert isinstance(result, QAThreadData)
    assert isinstance(result, NormalizedData)


def test_normalize_record_id_format(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert result.record_id == "reddit-netsec:submission-abc123"
    assert result.source_id == "reddit-netsec"
    assert result.source_record_id == "submission-abc123"


def test_normalize_populates_base_fields(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert result.content_hash is not None
    assert result.content_length is not None
    assert result.content_length > 0
    assert result.license is not None
    assert result.published_at is not None
    assert result.published_at.tzinfo == timezone.utc
    assert result.raw is None


def test_normalize_source_url_format(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert result.source_url == "https://www.reddit.com/r/netsec/comments/abc123/how_to_configure_iptables/"


def test_normalize_score_mapping(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert result.score == 50


def test_normalize_answer_count_is_top_level_only(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    # 2 top-level comments, not counting nested replies
    assert result.answer_count == 2


def test_normalize_has_accepted_answer_is_none(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert result.has_accepted_answer is None


def test_normalize_closed_is_none(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert result.closed is None


def test_normalize_tags_includes_subreddit(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert "netsec" in result.tags


def test_normalize_tags_includes_flair(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert "Question" in result.tags


def test_normalize_tags_no_flair(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "def456")
    result = _make_connector().normalize(r)

    assert result.tags == ["netsec"]


def test_normalize_content_contains_title_and_body(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert result.content.startswith("# How to configure iptables")
    assert "firewall rules" in result.content


def test_normalize_content_contains_comments(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert "ufw" in result.content
    assert "Arch Wiki" in result.content


def test_normalize_content_has_nested_replies(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    assert "> **Reply**" in result.content


def test_normalize_selftext_newlines_collapsed(tmp_path):
    records = _get_records(tmp_path)
    r = _get_record_by_sid(records, "abc123")
    result = _make_connector().normalize(r)

    # Original selftext had 5 consecutive newlines, should be collapsed
    assert "\n\n\n" not in result.content


def test_subreddit_name_customizes_source_id():
    c = RedditSubredditConnector("cybersecurity")
    assert c.source_id == "reddit-cybersecurity"
    assert c.subreddit == "cybersecurity"


def test_all_records_normalize_without_error(tmp_path):
    records = _get_records(tmp_path)
    connector = _make_connector()
    for r in records:
        result = connector.normalize(r)
        assert result.content
        assert result.content_hash
        assert result.content_length > 0


def test_all_record_ids_are_unique(tmp_path):
    records = _get_records(tmp_path)
    connector = _make_connector()
    ids = [connector.normalize(r).record_id for r in records]
    assert len(ids) == len(set(ids))


# ── helpers ─────────────────────────────────────────────────────────


def _collect_bodies(tree: list[dict]) -> list[str]:
    """Recursively collect all comment bodies from a tree."""
    bodies = []
    for node in tree:
        bodies.append(node.get("body", ""))
        bodies.extend(_collect_bodies(node.get("children", [])))
    return bodies
