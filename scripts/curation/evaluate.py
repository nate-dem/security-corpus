"""Compare raw classifier responses with explicitly assistant-authored references.

Agreement is a development diagnostic, never human accuracy or a release gate.
Reparse bound raw checkpoints; do not trust a cached parsed-label summary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter import score as engine
from scripts.youtube_filter.rubric import LABELS
from . import policy, review_packet, rubric, score


def reference(packet, path):
    data=json.loads(path.read_text())
    if (data.get('purpose')!='assistant_development_review' or data.get('reviewer_kind')!='assistant'
            or data.get('human_validated') is not False or data.get('independent_ground_truth') is not False):
        raise ValueError('Reference must disclose assistant origin and limitations')
    if review_packet.validate_labels(packet,data)['remaining']:
        raise ValueError('Incomplete assistant reference')
    cases={c['case_id']:c for c in packet['cases']}
    for row in data['annotations']:
        if row.get('reviewer_kind')!='assistant':
            raise ValueError('Reference reviewer origin mismatch')
        excerpt=row['supporting_excerpt']
        text=cases[row['case_id']]['text']
        if (type(excerpt['start']) is not int or type(excerpt['end']) is not int
                or not 0<=excerpt['start']<excerpt['end']<=len(text)
                or text[excerpt['start']:excerpt['end']]!=excerpt['text']):
            raise ValueError('Reference excerpt does not match source text')
        concerns=row['quality_concerns']
        if (not isinstance(concerns,list) or any(c not in rubric.QUALITY_CONCERNS for c in concerns)
                or len(set(concerns))!=len(concerns)):
            raise ValueError('Invalid reference quality concerns')
    return {r['case_id']:r for r in data['annotations']}


def reference_route(row):
    return policy.route({**row['labels'],'quality_concerns':row['quality_concerns']})


def evaluate(packet_path, reference_path, score_dir=None):
    packet=score.load_packet(packet_path)
    refs=reference(packet,reference_path)
    cases={c['case_id']:c for c in packet['cases']}
    result={'packet_sha256':packet['packet_sha256'],'reference_sha256':_sha256(reference_path),
        'policy_version':policy.VERSION,'policy_sha256':_sha256(Path(policy.__file__)),
        'evaluator_sha256':_sha256(Path(__file__)),
        'reference_kind':'assistant','human_validated':False,'independent_ground_truth':False,
        'production_accuracy_established':False,'cases':len(cases),
        'reference_routes':dict(Counter(reference_route(r) for r in refs.values())),
        'reference_by_source':{s:dict(Counter(reference_route(refs[k]) for k,c in cases.items() if c['source']==s))
                               for s in sorted({c['source'] for c in cases.values()})},
        'scope':'Development agreement only. Cases are not a representative yield sample or independent ground truth. '
                'Eligible is a quality candidate, subject to deduplication and publication terms.'}
    if score_dir is None:
        return result
    summary=json.loads((score_dir/'summary.json').read_text())
    config=json.loads((score_dir/'run-config.json').read_text())
    config_sha=_sha256(score_dir/'run-config.json')
    if (summary.get('dry_run') is not False or config.get('dry_run') is not False
            or config['packet_sha256']!=packet['packet_sha256'] or summary['packet_sha256']!=packet['packet_sha256']
            or config['prompt_version']!=rubric.VERSION or summary['run_config_sha256']!=config_sha
            or config['code']['scripts.curation.rubric']!=_sha256(Path(rubric.__file__))
            or summary['requests_sha256']!=_sha256(score_dir/'requests.jsonl')):
        raise ValueError('Scoring run/configuration does not match the reviewed packet and rubric')
    by_text=defaultdict(list)
    for key,c in cases.items():
        kind='youtube' if c['source']=='youtube-commons' else 'web'
        by_text[(kind,c['text_sha256'])].append(key)
    by_case=defaultdict(list)
    seen=set()
    provenance={k:config[k] for k in ('model','model_revision','prompt_version')}
    provenance['run_config_sha256']=config_sha
    with (score_dir/'requests.jsonl').open() as handle:
        for line in handle:
            task=json.loads(line)
            group=by_text.get((task['kind'],task['content_hash']))
            if not group:
                raise ValueError('Unknown scoring request text')
            text=cases[group[0]]['text']
            start,end=task['start'],task['end']
            if (type(start) is not int or type(end) is not int or not 0<=start<end<=len(text)
                    or task['key']!=f"{task['kind']}-{task['content_hash']}-{start}-{end}" or task['key'] in seen):
                raise ValueError('Invalid or repeated scoring span')
            seen.add(task['key'])
            cached=engine._cached(score_dir/'decisions'/f"{task['key']}.json",task,text[start:end],rubric.parse_response,provenance)
            state=cached or {'parse_status':'unprocessed','labels':None}
            for key in group:
                by_case[key].append({'start':start,'end':end,**state})
    if len(seen)!=summary['spans'] or set(by_case)!=set(cases):
        raise ValueError('Missing or extra scoring requests')
    comparisons=[]
    confusion=Counter()
    dimensions={k:Counter() for k in LABELS}
    for key,c in cases.items():
        spans=sorted(by_case[key],key=lambda r:r['start'])
        if (spans[0]['start']!=0 or spans[-1]['end']!=len(c['text'])
                or any(a['end']!=b['start'] for a,b in zip(spans,spans[1:]))):
            raise ValueError('Scoring requests omit or overlap source characters')
        routes=[policy.route(r['labels'],r['parse_status']) for r in spans]
        predicted=routes[0] if len(set(routes))==1 else 'review_required'
        expected=reference_route(refs[key])
        confusion[(expected,predicted)]+=1
        if len(spans)==1 and spans[0]['parse_status']=='ok':
            for dimension in LABELS:
                dimensions[dimension]['compared']+=1
                dimensions[dimension]['agreements']+=spans[0]['labels'][dimension]==refs[key]['labels'][dimension]
        comparisons.append({'case_id':key,'source':c['source'],'content_length':c['content_length'],
            'assistant_route':expected,'model_route':predicted,'span_routes':routes,
            'parse_statuses':[r['parse_status'] for r in spans],
            'assistant_quality_concerns':refs[key]['quality_concerns'],
            'model_quality_concerns':sorted({v['kind'] for r in spans for v in (r.get('labels') or {}).get('quality_concerns',[])}),
            'assistant_notes':refs[key]['notes']})
    result.update(model=config['model'],model_revision=config['model_revision'],run_config_sha256=config_sha,
        score_summary_sha256=_sha256(score_dir/'summary.json'),
        scoring_coverage_complete=all(r['parse_status']=='ok' for spans in by_case.values() for r in spans),
        model_routes=dict(Counter(r['model_route'] for r in comparisons)),
        action_agreements=sum(r['model_route']==r['assistant_route'] for r in comparisons),
        action_confusion=[{'assistant':a,'model':b,'cases':n} for (a,b),n in sorted(confusion.items())],
        dimension_agreement_single_span={k:dict(v) for k,v in dimensions.items()},
        model_eligible_reference_not_eligible=[r['case_id'] for r in comparisons if r['model_route']=='eligible' and r['assistant_route']!='eligible'],
        reference_eligible_model_not_eligible=[r['case_id'] for r in comparisons if r['assistant_route']=='eligible' and r['model_route']!='eligible'],
        generation=summary['generation_this_invocation'],comparisons=comparisons)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--score-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.resolve() in {args.packet.resolve(),args.reference.resolve()} or (args.score_dir and args.output.resolve().is_relative_to(args.score_dir.resolve())):
        p.error('Report output must be separate from input artifacts')
    result=evaluate(args.packet,args.reference,args.score_dir)
    _write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
