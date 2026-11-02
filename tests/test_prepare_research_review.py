import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.release import prepare_research_review as review


def test_packet_binds_exact_full_documents_and_marks_unreviewed(tmp_path, monkeypatch):
    root, out = tmp_path / "corpus", tmp_path / "review"
    root.mkdir()
    text = "Security example. " * 1000
    records = [dict(source_id="qa", record_id=str(i), content=text+str(i),
                    content_hash=hashlib.sha256((text+str(i)).encode()).hexdigest(),
                    content_length=2000+i, license="CC-BY-SA", score=i)
               for i in range(30)]
    payload = root / "part.parquet"
    pq.write_table(pa.Table.from_pylist(records), payload)
    manifest = dict(files=[dict(relative_path="part.parquet", records=30,
                               sha256=review.final.sha(payload))], totals=dict(records=30))
    monkeypatch.setattr(review, "verify_parent", lambda _: ({"manifest_sha256":"bound"}, manifest))
    report = review.build(root, out)
    assert report["draws"] == 16 and not report["review_complete"]
    assert report["populations"]["qa"]["records"] == 30
    notes = [json.loads(s) for s in (out / "review.jsonl").read_text().splitlines()]
    assert len(notes) == report["unique_documents"]
    for note in notes:
        original = records[note["row"]]
        assert note["record_id"] == original["record_id"]
        assert (out / note["document"]).read_text() == original["content"]
        assert note["document_sha256"] == original["content_hash"]
        assert note["metadata"]["score"] == original["score"]
        assert not note["full_text_in_excerpts"]
        assert note["review_status"] == "unreviewed" and note["proposed_action"] is None
    assert review.select(manifest, root, "bound")[0] == review.select(manifest, root, "bound")[0]
    assert review.final.sha(payload) == manifest["files"][0]["sha256"]
