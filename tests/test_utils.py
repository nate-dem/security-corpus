import hashlib

from ingest.utils import compute_content_hash, compute_token_count


def test_compute_content_hash_returns_sha256_hex():
    content = "The debug command in Sendmail is enabled."
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert compute_content_hash(content) == expected


def test_compute_content_hash_empty_string():
    expected = hashlib.sha256(b"").hexdigest()
    assert compute_content_hash("") == expected


def test_compute_content_hash_unicode():
    content = "Schwachstelle in der Authentifizierung — Überprüfung"
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert compute_content_hash(content) == expected


def test_compute_token_count_returns_positive_int():
    content = "The debug command in Sendmail is enabled, allowing attackers to execute commands as root."
    count = compute_token_count(content)
    assert isinstance(count, int)
    assert count > 0


def test_compute_token_count_empty_string():
    assert compute_token_count("") == 0


def test_compute_token_count_scales_with_length():
    short = "hello"
    long = "hello " * 100
    assert compute_token_count(long) > compute_token_count(short)


def test_compute_token_count_treats_special_token_text_as_ordinary_text():
    count = compute_token_count("literal <|endoftext|> marker")
    assert isinstance(count, int)
    assert count > 0

