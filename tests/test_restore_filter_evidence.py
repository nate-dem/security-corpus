import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.release import restore_filter_evidence as restore
from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write
from scripts.release.restore_filter_evidence import check_web_score, check_youtube_spans, selected_table
from scripts.curation_single import rubric


def test_web_evidence_enforces_original_model_selection_and_relevance():
    plan = dict(web_plan_sha256="p", model_sha256="m")
    row = dict(source="redsage-cfw", plan_sha256="p", model_sha256="m", route="scored",
               domain_score=.9, substance_score=.8, joint_score=.8, upstream_relevant="true")
    check_web_score(row, plan)
    with pytest.raises(ValueError, match="relevance flag"):
        check_web_score(dict(row, upstream_relevant="false"), plan)
    with pytest.raises(ValueError, match="provenance"):
        check_web_score(dict(row, model_sha256="wrong"), plan)
    with pytest.raises(ValueError, match="selection"):
        check_web_score(dict(row, substance_score=.7, joint_score=.7), plan)


def test_youtube_evidence_rejects_span_gaps_and_ineligible_labels():
    plan = dict(youtube_plan_sha256="p", youtube_model=dict(model="m", revision="r"))
    decision = dict(content_hash="h", route="eligible", scoring_complete=True,
                    plan_sha256="p", run_config_sha256="c")
    span = dict(content_hash="h", start=0, end=10, model="m", model_revision="r",
                run_config_sha256="c", parse_status="ok", domain="computing",
                quality="useful", language="english", prompt_version=rubric.VERSION)
    check_youtube_spans([span], decision, 10, plan, "c")
    with pytest.raises(ValueError, match="coverage"):
        check_youtube_spans([span], decision, 11, plan, "c")
    with pytest.raises(ValueError, match="gap/overlap"):
        check_youtube_spans([dict(span, end=4), dict(span, start=5)], decision, 10, plan, "c")
    with pytest.raises(ValueError, match="not eligible"):
        check_youtube_spans([dict(span, quality="unusable")], decision, 10, plan, "c")


def test_evidence_selection_preserves_original_values_and_adds_row_binding():
    table = pa.table(dict(content_hash=["drop", "keep"], selection_decision=pa.array([None,None],pa.bool_()), score=[.1,.9]))
    out = selected_table(table, [1], [dict(record_id="s:1", final_file=3, final_row=4)])
    assert out.select(table.column_names).equals(table.slice(1,1))
    assert out["record_id"].to_pylist() == ["s:1"]


def test_package_joins_only_retained_rows_and_keeps_span_provenance(tmp_path, monkeypatch):
    corpus, fetched, snapshot, output = (tmp_path/n for n in ("corpus", "fetched", "snapshot", "output"))
    (corpus/"records").mkdir(parents=True)
    (snapshot/"dedup").mkdir(parents=True)
    webpath = fetched/"web-semantic-v1/parts/00000/scores.parquet"
    webpath.parent.mkdir(parents=True)
    folder = fetched/"single-pass-v1/parts/00000"
    (folder/"work/score").mkdir(parents=True)
    files = []
    for i, source in enumerate(("primus-fineweb", "youtube-commons")):
        row = dict(source_id=source, dataset_revision="revision", source_shard="shard", source_row=4,
                   content_hash="hash"+str(i), record_id=source+":id", content_length=1, content="example")
        path=corpus/f"records/{i:06d}.parquet"
        pq.write_table(pa.Table.from_pylist([row]),path)
        files.append(dict(relative_path=f"records/{i:06d}.parquet", sha256=final.sha(path),
                          original_remote_path="/dedup/records/youtube-commons/00000.parquet"))
    # Prepared-file validation is independently covered by finalizer tests.
    prepared=dict(manifest_sha256="manifest")
    monkeypatch.setattr(final,"verify_prepared",lambda root:(prepared,dict(files=files)))
    write(snapshot/"dedup/plan.json",dict(files=[dict(slot=0)]))
    score=dict(source="primus-fineweb",dataset_revision="revision",source_shard="shard",source_row=4,
               content_hash="hash0",content_length=1,plan_sha256="p",model_sha256="m",route="scored",
               domain_score=.9,substance_score=.8,joint_score=.8,selection_decision=None)
    pq.write_table(pa.Table.from_pylist([score,dict(score,source_row=5,content_hash="not-retained")]),webpath)
    write(folder/"work/score/run-config.json",{})
    config_sha=final.sha(folder/"work/score/run-config.json")
    decision=dict(source="youtube-commons",content_hash="hash1",content_length=1,route="eligible",
                  scoring_complete=True,plan_sha256="yp",run_config_sha256=config_sha,selection_decision=None)
    span=dict(content_hash="hash1",start=0,end=7,model="ym",model_revision="yr",
              run_config_sha256=config_sha,parse_status="ok",domain="computing",quality="useful",
              language="english",prompt_version=rubric.VERSION,attempts_json='[{"raw":"preserve"}]')
    pq.write_table(pa.Table.from_pylist([decision]),folder/"decisions.parquet")
    pq.write_table(pa.Table.from_pylist([span]),folder/"spans.parquet")
    write(folder/"summary.json",dict(complete=True,workflow_finished=True,plan_sha256="yp",
          files={n:final.sha(folder/n) for n in ("decisions.parquet","spans.parquet")}))
    plan=dict(web_plan_sha256="p",model_sha256="m",youtube_plan_sha256="yp",
              youtube_model=dict(model="ym",revision="yr"),snapshot=str(snapshot),
              frozen=dict(parts=[dict(slot=0,summary_sha256=final.sha(folder/"summary.json"))]),
              files={str(p.relative_to(fetched)):final.sha(p) for p in fetched.rglob("*") if p.is_file()})
    monkeypatch.setattr(restore,"evidence_plan",lambda project:plan)
    result=restore.package(tmp_path,corpus,fetched,output)
    assert result["record_coverage"] == 2 and result["evidence_rows"] == {
        "web-records":1,"youtube-records":1,"youtube-spans":1}
    web=pq.read_table(output/"web-records.parquet").to_pylist()
    assert web[0]["selection_decision"] is None and web[0]["source_row"]==4
    assert web[0]["record_id"]=="primus-fineweb:id"
    assert pq.read_table(output/"youtube-spans.parquet")["attempts_json"].to_pylist()==[span["attempts_json"]]
