from datetime import timezone
from pathlib import Path

from ingest.connectors.base import AcademicPaperData, NormalizedData
from ingest.connectors.arxiv.connector import ArxivConnector, _paper_dir_to_arxiv_id
from ingest.connectors.arxiv.metadata import (
    _map_license,
    _parse_arxiv_id,
    _parse_authors,
    _parse_datestamp,
    _parse_oai_record,
    build_metadata_index,
)
from ingest.utils import (
    ARXIV_PERPETUAL_NON_EXCLUSIVE,
    CC_BY_4_0,
    CC_BY_SA_4_0,
    CC_BY_NC_SA_4_0,
    PUBLIC_DOMAIN,
)

FIXTURES = Path(__file__).parent / "fixtures" / "arxiv"


# ── Metadata parsing ───────────────────────────────────────────────────────

def test_parse_arxiv_id_new_format():
    assert _parse_arxiv_id("oai:arXiv.org:2401.12345") == "2401.12345"


def test_parse_arxiv_id_old_format():
    assert _parse_arxiv_id("oai:arXiv.org:cs/0601001") == "cs/0601001"


def test_parse_arxiv_id_no_prefix():
    assert _parse_arxiv_id("2401.12345") == "2401.12345"


def test_paper_dir_to_arxiv_id_new_format():
    assert _paper_dir_to_arxiv_id("2401.12345") == "2401.12345"


def test_paper_dir_to_arxiv_id_old_format():
    assert _paper_dir_to_arxiv_id("cs-0601001") == "cs/0601001"


def test_paper_dir_to_arxiv_id_old_hyphenated_category():
    assert _paper_dir_to_arxiv_id("quant-ph-0408108") == "quant-ph/0408108"


def test_parse_authors_multiple():
    field = {"author": [
        {"keyname": "Smith", "forenames": "Alice"},
        {"keyname": "Jones", "forenames": "Bob"},
    ]}
    assert _parse_authors(field) == ["Alice Smith", "Bob Jones"]


def test_parse_authors_single():
    field = {"author": {"keyname": "Chen", "forenames": "Wei"}}
    assert _parse_authors(field) == ["Wei Chen"]


def test_parse_authors_no_forenames():
    field = {"author": {"keyname": "Consortium"}}
    assert _parse_authors(field) == ["Consortium"]


def test_parse_authors_none():
    assert _parse_authors(None) == []


def test_parse_datestamp():
    dt = _parse_datestamp("2024-01-15")
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 15
    assert dt.tzinfo == timezone.utc


def test_map_license_known_urls():
    assert _map_license("http://creativecommons.org/licenses/by/4.0/") == CC_BY_4_0
    assert _map_license("http://creativecommons.org/licenses/by-sa/4.0/") == CC_BY_SA_4_0
    assert _map_license("http://creativecommons.org/licenses/by-nc-sa/4.0/") == CC_BY_NC_SA_4_0
    assert _map_license("http://creativecommons.org/publicdomain/zero/1.0/") == PUBLIC_DOMAIN
    assert _map_license("http://arxiv.org/licenses/nonexclusive-distrib/1.0/") == ARXIV_PERPETUAL_NON_EXCLUSIVE


def test_map_license_unknown_url():
    result = _map_license("http://example.com/unknown-license")
    # Unknown URLs are passed through as-is
    assert result == "http://example.com/unknown-license"


def test_map_license_none():
    assert _map_license(None) == ARXIV_PERPETUAL_NON_EXCLUSIVE


def test_parse_oai_record_extracts_fields():
    record = {
        "record": {
            "header": {
                "identifier": "oai:arXiv.org:2401.00001",
                "datestamp": "2024-01-02",
            },
            "metadata": {
                "arXiv": {
                    "id": "2401.00001",
                    "title": "Test Paper",
                    "authors": {"author": [
                        {"keyname": "Smith", "forenames": "Alice"},
                    ]},
                    "abstract": "  Test abstract  ",
                    "categories": "cs.CR cs.AI",
                    "license": "http://creativecommons.org/licenses/by/4.0/",
                    "doi": "10.1234/test",
                    "journal-ref": "Test Journal 2024",
                }
            }
        }
    }
    parsed = _parse_oai_record(record)
    assert parsed is not None
    assert parsed["arxiv_id"] == "2401.00001"
    assert parsed["title"] == "Test Paper"
    assert parsed["authors"] == ["Alice Smith"]
    assert parsed["abstract"] == "Test abstract"
    assert parsed["categories"] == ["cs.CR", "cs.AI"]
    assert parsed["primary_category"] == "cs.CR"
    assert parsed["doi"] == "10.1234/test"
    assert parsed["journal_ref"] == "Test Journal 2024"
    assert parsed["license_url"] == "http://creativecommons.org/licenses/by/4.0/"
    assert parsed["datestamp"] == "2024-01-02"


def test_parse_oai_record_handles_missing_optional_fields():
    record = {
        "record": {
            "header": {
                "identifier": "oai:arXiv.org:2401.99999",
                "datestamp": "2024-01-01",
            },
            "metadata": {
                "arXiv": {
                    "id": "2401.99999",
                    "title": "Minimal Paper",
                    "authors": {"author": {"keyname": "Solo"}},
                    "abstract": "Minimal.",
                    "categories": "cs.CR",
                    "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                }
            }
        }
    }
    parsed = _parse_oai_record(record)
    assert parsed is not None
    assert parsed["doi"] is None
    assert parsed["journal_ref"] is None


def test_parse_oai_record_deleted():
    record = {
        "record": {
            "header": {
                "@status": "deleted",
                "identifier": "oai:arXiv.org:2401.00099",
                "datestamp": "2024-01-01",
            },
            "metadata": {}
        }
    }
    assert _parse_oai_record(record) is None


# ── Metadata index ─────────────────────────────────────────────────────────

def test_build_metadata_index():
    index = build_metadata_index(FIXTURES / "metadata" / "cs_CR")
    # Fixture has 4 lines: 2 valid papers, 1 deleted, 1 orphan metadata
    # Deleted record (2401.00003) has @status=deleted in the outer header
    # but the fixture structure has a duplicate header key — xmltodict would
    # overwrite, so it depends on parse order.  We have 3 or 4 in the index.
    assert "2401.00001" in index
    assert "2401.00002" in index
    assert "2401.00004" in index
    assert index["2401.00001"]["title"] == "A Survey of Buffer Overflow Attacks and Defenses"
    assert index["2401.00002"]["authors"] == ["Wei Chen"]


# ── iter_records ───────────────────────────────────────────────────────────

def _get_records():
    connector = ArxivConnector()
    return list(connector.iter_records(FIXTURES))


def test_iter_records_joins_metadata_with_content():
    records = _get_records()
    # 2401.00001: has metadata + completed normalization → emitted
    # 2401.00002: has metadata + completed normalization → emitted
    # 2401.00003: completed normalization but deleted/no metadata → skipped
    # 2401.00004: has metadata but incomplete normalization → skipped
    ids = [r["arxiv_id"] for r in records]
    assert "2401.00001" in ids
    assert "2401.00002" in ids


def test_iter_records_skips_incomplete_normalization():
    records = _get_records()
    ids = [r["arxiv_id"] for r in records]
    # 2401.00004 has completed=false in status.json
    assert "2401.00004" not in ids


def test_iter_records_skips_orphan_content():
    records = _get_records()
    ids = [r["arxiv_id"] for r in records]
    # 2401.00003 has no matching metadata (deleted in metadata JSONL)
    assert "2401.00003" not in ids


def test_iter_records_record_has_content():
    records = _get_records()
    for r in records:
        assert "content" in r
        assert len(r["content"]) > 0
        assert "\\documentclass" in r["content"] or "\\section" in r["content"]


def test_iter_records_record_has_metadata():
    records = _get_records()
    for r in records:
        assert "title" in r
        assert "authors" in r
        assert "abstract" in r
        assert "categories" in r
        assert r["source_format"] in {"latex", "pdf"}


# ── normalize ──────────────────────────────────────────────────────────────

def _get_normalized():
    connector = ArxivConnector()
    records = list(connector.iter_records(FIXTURES))
    return [connector.normalize(r) for r in records]


def test_normalize_returns_academic_paper_data():
    for result in _get_normalized():
        assert isinstance(result, AcademicPaperData)
        assert isinstance(result, NormalizedData)


def test_normalize_record_id_format():
    for result in _get_normalized():
        assert result.record_id.startswith("arxiv:")
        assert result.source_id == "arxiv"
        assert result.record_id == f"arxiv:{result.arxiv_id}"


def test_normalize_source_url():
    for result in _get_normalized():
        assert result.source_url == f"https://arxiv.org/abs/{result.arxiv_id}"


def test_normalize_content_is_latex_body():
    results = _get_normalized()
    paper1 = [r for r in results if r.arxiv_id == "2401.00001"][0]
    assert "Buffer overflow" in paper1.content or "buffer overflow" in paper1.content


def test_normalize_abstract_separate():
    results = _get_normalized()
    paper1 = [r for r in results if r.arxiv_id == "2401.00001"][0]
    assert paper1.abstract is not None
    assert "buffer overflow" in paper1.abstract.lower()
    # Abstract comes from metadata, not from LaTeX body
    assert paper1.abstract != paper1.content


def test_normalize_authors_list():
    results = _get_normalized()
    paper1 = [r for r in results if r.arxiv_id == "2401.00001"][0]
    assert paper1.authors == ["Alice Smith", "Bob Jones"]

    paper2 = [r for r in results if r.arxiv_id == "2401.00002"][0]
    assert paper2.authors == ["Wei Chen"]


def test_normalize_categories():
    results = _get_normalized()
    paper1 = [r for r in results if r.arxiv_id == "2401.00001"][0]
    assert paper1.categories == ["cs.CR", "cs.SE"]
    assert paper1.primary_category == "cs.CR"


def test_normalize_license_mapping():
    results = _get_normalized()
    paper1 = [r for r in results if r.arxiv_id == "2401.00001"][0]
    assert paper1.license == CC_BY_4_0

    paper2 = [r for r in results if r.arxiv_id == "2401.00002"][0]
    assert paper2.license == ARXIV_PERPETUAL_NON_EXCLUSIVE


def test_normalize_doi_and_journal_ref():
    results = _get_normalized()
    paper1 = [r for r in results if r.arxiv_id == "2401.00001"][0]
    assert paper1.doi == "10.1234/example.2024.001"
    assert paper1.journal_ref == "Journal of Security 2024"

    paper2 = [r for r in results if r.arxiv_id == "2401.00002"][0]
    assert paper2.doi is None
    assert paper2.journal_ref is None


def test_normalize_source_format_defaults_to_latex():
    results = _get_normalized()
    for result in results:
        assert result.source_format == "latex"


def test_iter_records_reads_pdf_text(tmp_path):
    raw_dir = tmp_path / "raw"
    metadata_dir = raw_dir / "metadata" / "cs_CR"
    paper_dir = raw_dir / "source" / "normalized" / "2401" / "2401.00006"
    metadata_dir.mkdir(parents=True)
    paper_dir.mkdir(parents=True)

    metadata_dir.joinpath("2401.jsonl").write_text(
        '{"record": {"header": {"identifier": "oai:arXiv.org:2401.00006", '
        '"datestamp": "2024-01-18"}, "metadata": {"arXiv": {'
        '"id": "2401.00006", "title": "PDF Security Paper", '
        '"authors": {"author": {"keyname": "Reader"}}, '
        '"abstract": "PDF extracted abstract.", "categories": "cs.CR", '
        '"license": "http://creativecommons.org/licenses/by/4.0/"}}}}\n',
        encoding="utf-8",
    )
    paper_dir.joinpath("status.json").write_text(
        '{"aid": "2401.00006", "completed": true, '
        '"source_format": "pdf", "pdf_extracted": true, "errors": []}',
        encoding="utf-8",
    )
    paper_dir.joinpath("main.txt").write_text(
        "Extracted PDF text about side-channel attacks.",
        encoding="utf-8",
    )

    connector = ArxivConnector()
    records = list(connector.iter_records(raw_dir))

    assert len(records) == 1
    assert records[0]["source_format"] == "pdf"
    assert records[0]["content"] == "Extracted PDF text about side-channel attacks."

    normalized = connector.normalize(records[0])
    assert normalized.source_format == "pdf"


def test_normalize_populates_base_fields():
    for result in _get_normalized():
        assert result.content_hash is not None
        assert result.content_length is not None
        assert result.content_length > 0
        assert result.published_at is not None
        assert result.published_at.tzinfo == timezone.utc
        assert result.ingested_at.tzinfo == timezone.utc
        assert result.license is not None


def test_normalize_no_raw_field():
    for result in _get_normalized():
        assert result.raw is None


def test_all_record_ids_are_unique():
    results = _get_normalized()
    ids = [r.record_id for r in results]
    assert len(ids) == len(set(ids))


# ── Multi-directory metadata (citation expansion) ─────────────────────────

def test_build_metadata_index_multiple_dirs():
    dirs = [
        FIXTURES / "metadata" / "cs_CR",
        FIXTURES / "metadata" / "citations",
    ]
    index = build_metadata_index(dirs)
    # Should contain papers from both directories
    assert "2401.00001" in index  # from cs_CR
    assert "2401.00005" in index  # from citations


def test_build_metadata_index_single_path_still_works():
    index = build_metadata_index(FIXTURES / "metadata" / "cs_CR")
    assert "2401.00001" in index


def test_iter_records_includes_citation_papers():
    connector = ArxivConnector()
    records = list(connector.iter_records(FIXTURES))
    ids = [r["arxiv_id"] for r in records]
    # Citation paper with both metadata and normalized content
    assert "2401.00005" in ids


def test_normalize_citation_paper():
    connector = ArxivConnector()
    records = list(connector.iter_records(FIXTURES))
    citation_rec = [r for r in records if r["arxiv_id"] == "2401.00005"][0]
    result = connector.normalize(citation_rec)
    assert isinstance(result, AcademicPaperData)
    assert result.primary_category == "cs.LG"
    assert result.categories == ["cs.LG", "cs.AI"]
    assert result.authors == ["Li Zhang", "Min Park"]
