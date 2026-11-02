import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.youtube import prepare_english as english, profile
from test_youtube_profile import make_download


def setup(root):
    text = "Virtual memory maps addresses to pages.\n" * 100
    a = pa.table({'text':[text, 'bonjour', '', None, 'same video different translation'],
        'transcription_language':['en','fr','en','en','en'],
        'original_language':['en','en','es','en','de'],
        'video_id':['v','v','blank','missing','v'],
        'license':['CC BY',None,None,None,'CC BY']})
    b = pa.table({'text':[text,'short'], 'transcription_language':['en','en'],
        'original_language':['es','en'], 'video_id':['v2','short'], 'extra_metadata':['preserve','x']})
    make_download(root,[a,b])
    assert profile.main(['--data-dir',str(root),'--workers','1']) == 0
    return text


def run(root, workers=1):
    return english.main(['--data-dir',str(root),'--workers',str(workers)])


def test_complete_english_frame_exact_hash_tokens_and_all_variant_lineage(tmp_path):
    text = setup(tmp_path)
    before = {p:profile._sha256(p) for p in (tmp_path/'raw').rglob('*.parquet')}
    assert run(tmp_path,2) == 0
    output=tmp_path/'english-candidates-v1'
    summary=json.loads((output/'summary.json').read_text())
    assert summary['complete'] and summary['english_rows']==6 and summary['candidate_rows']==4
    assert summary['exact_unique_texts']==3 and summary['extra_exact_copies']==1
    assert summary['candidate_tokens']-summary['exact_unique_tokens']==compute_token_count(text)
    with duckdb.connect() as con:
        rows=con.read_parquet(str(output/'rows/*.parquet'),hive_partitioning=False).to_arrow_table().to_pylist()
    assert {r['text_state'] for r in rows}=={'candidate','blank','missing'}
    originals={r['source_shard']:pq.read_table(tmp_path/'raw'/r['source_shard']).to_pylist() for r in rows}
    for r in rows:
        original=originals[r['source_shard']][r['source_row']]
        assert r['text']==original['text']
        assert json.loads(r['metadata_json'])=={k:v for k,v in original.items() if k!='text'}
        if r['text_state']=='candidate':
            assert r['content_hash']==compute_content_hash(r['text'])
            assert r['content_length']==compute_token_count(r['text'])
    assert sum(r['video_id']=='v' for r in rows)==2  # Different texts not collapsed by video ID.
    index=pq.read_table(output/'unique_index.parquet').to_pylist()
    duplicate=next(r for r in index if r['content_hash']==compute_content_hash(text))
    assert duplicate['source_copies']==2
    assert duplicate['source_shard']=='nested/shard_0.parquet' and duplicate['source_row']==0
    assert before=={p:profile._sha256(p) for p in before}


def test_resume_reuses_outputs_and_repairs_corruption(tmp_path,monkeypatch):
    setup(tmp_path)
    assert run(tmp_path)==0
    output=tmp_path/'english-candidates-v1'
    paths=sorted((output/'rows').glob('*.parquet'))
    times={p:p.stat().st_mtime_ns for p in paths}
    old=english._row
    monkeypatch.setattr(english,'_row',lambda *a:pytest.fail('Must reuse completed shards'))
    assert run(tmp_path)==0
    assert times=={p:p.stat().st_mtime_ns for p in paths}
    monkeypatch.setattr(english,'_row',old)
    paths[0].write_bytes(b'corrupt')
    assert run(tmp_path)==0
    assert pq.read_table(paths[0]).num_rows==4
    assert paths[1].stat().st_mtime_ns==times[paths[1]]


def test_source_corruption_and_config_change_fail_closed(tmp_path):
    setup(tmp_path)
    assert run(tmp_path)==0
    root=tmp_path/'english-candidates-v1'
    old=(root/'summary.json').read_bytes()
    config=json.loads((root/'run-config.json').read_text())
    config['language']='fr'
    (root/'run-config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='configuration changed'):
        run(tmp_path)
    assert (root/'summary.json').read_bytes()==old
    config['language']='en'
    (root/'run-config.json').write_text(json.dumps(config))
    raw=next((tmp_path/'raw').rglob('*.parquet'))
    data=bytearray(raw.read_bytes())
    data[30]^=1
    raw.write_bytes(data)
    with pytest.raises(ValueError):
        run(tmp_path)
    assert json.loads((root/'summary.json').read_text())['complete'] is False


def test_empty_english_frame_and_output_protection(tmp_path):
    make_download(tmp_path,[pa.table({'text':['bonjour'],'transcription_language':['fr']})])
    assert profile.main(['--data-dir',str(tmp_path),'--workers','1'])==0
    assert run(tmp_path)==0
    report=json.loads((tmp_path/'english-candidates-v1/summary.json').read_text())
    assert report['english_rows']==report['exact_unique_texts']==report['exact_unique_tokens']==0
    for output in [tmp_path,tmp_path/'raw',tmp_path/'profile-v1',tmp_path.parent]:
        with pytest.raises(SystemExit):
            english.main(['--data-dir',str(tmp_path),'--output-dir',str(output)])
    assert Path(english.__file__).is_file()
