"""Resumable development evaluation on a bound review packet; never selects data.

Uses the existing vLLM batch/checkpoint mechanics with the new source-specific
rubric. Old YouTube runners and decisions remain reproducible and unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
import time

from ingest.utils import compute_content_hash, compute_token_count
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter import score as engine, rubric as base, rubric_segments
from . import review_packet, rubric, rubric_v3, rubric_v4, critic, critic_v2, runtime


def load_packet(path):
    packet=json.loads(path.read_text())
    digest=hashlib.sha256(json.dumps({k:v for k,v in packet.items() if k!='packet_sha256'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if packet.get('packet_sha256')!=digest or packet.get('purpose') not in {'development_only','candidate_scoring'}:
        raise ValueError('Invalid development packet binding')
    seen=set()
    for case in packet['cases']:
        if (case['case_id'] in seen or not case['text'].strip()
                or compute_content_hash(case['text'])!=case['text_sha256']
                or compute_token_count(case['text'])!=case['content_length']):
            raise ValueError('Duplicate or inconsistent review case')
        seen.add(case['case_id'])
    if not seen:
        raise ValueError('Empty review packet')
    return packet


def main(argv=None, model_factory=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--rubric-version',choices=('v2','v3','critic-v1','v4','critic-v2'),default='v2')
    parser.add_argument('--model',default='Qwen/Qwen3-32B')
    parser.add_argument('--model-revision',default='9216db5781bf21249d130ec9da846c4624c16137')
    parser.add_argument('--tensor-parallel-size',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--focal-tokens',type=int,default=4096)
    parser.add_argument('--max-model-len',type=int,default=8192)
    parser.add_argument('--max-output-tokens',type=int,default=1024)
    parser.add_argument('--language-model-only',action='store_true',help='Skip multimodal encoders on a compatible vLLM runtime')
    parser.add_argument('--enable-cuda-graphs',action='store_true',help='Use vLLM graph execution; changes the bound runtime configuration')
    parser.add_argument('--compact-json',action='store_true',help='Pin xgrammar and disallow inter-field JSON whitespace; requires a new run directory')
    args=parser.parse_args(argv)
    active_rubric = {'v2':rubric,'v3':rubric_v3,'critic-v1':critic,'v4':rubric_v4,'critic-v2':critic_v2}[args.rubric_version]
    if not re.fullmatch('[0-9a-f]{40}',args.model_revision) or args.batch_size<1 or args.tensor_parallel_size<1:
        parser.error('Immutable model revision and positive parallelism required')
    packet=load_packet(args.packet)
    output=args.output_dir.resolve()
    if output==args.packet.parent.resolve() or args.packet.resolve().is_relative_to(output):
        parser.error('Keep score output separate from the review packet')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    output.mkdir(parents=True,exist_ok=True)
    with _directory_lock(output):
        tokenizer,snapshot=engine.load_tokenizer(args.model,args.model_revision,args.local_files_only)
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        config.update(packet_sha256=packet['packet_sha256'],prompt_version=active_rubric.VERSION,
            versions=engine._versions(args.dry_run),temperature=0,enable_thinking=False,
            code={m.__name__:_sha256(Path(m.__file__)) for m in (active_rubric,rubric,rubric_v3,review_packet,engine,base,rubric_segments)},
            runner_sha256=_sha256(Path(__file__)),
            resident_runtime_sha256=_sha256(Path(runtime.__file__)) if model_factory else None,
            tokenizer_files={n:_sha256(snapshot/n) for n in engine.TOKENIZER_FILES if (snapshot/n).is_file()})
        if active_rubric is critic_v2:
            config['code'][rubric_v4.__name__]=_sha256(Path(rubric_v4.__file__))
        config_path=output/'run-config.json'
        if config_path.exists() and json.loads(config_path.read_text())!=config:
            raise ValueError('Configuration changed; choose a new output directory')
        if not config_path.exists() and any((output/'decisions').glob('*.json')):
            raise ValueError('Decisions without run configuration')
        _write_json(config_path,config)
        config_sha=_sha256(config_path)
        _write_json(output/'summary.json',{'complete':False,'status':'planning','dry_run':args.dry_run})
        tasks,documents,by_kind={}, {}, {}
        for case in packet['cases']:
            if case['source'] not in {'youtube-commons','redsage-cfw','primus-fineweb'}:
                raise ValueError('No source-specific rubric for case')
            kind='youtube' if case['source']=='youtube-commons' else 'web'
            digest=case['text_sha256']
            documents[digest]={'text':case['text']}
            group=by_kind.setdefault((kind,digest),[])
            group.append(case['case_id'])
            if len(group)>1:
                continue
            for span in active_rubric.make_spans(case['text'],tokenizer,kind,focal_tokens=args.focal_tokens,
                    max_model_len=args.max_model_len,max_output_tokens=args.max_output_tokens):
                key=f"{kind}-{digest}-{span['start']}-{span['end']}"
                task={**span,'kind':kind,'content_hash':digest,'key':key}
                task['task_sha256']=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
                tasks[key]=task
        tasks=list(tasks.values())
        request_path=output/'requests.jsonl'
        request_path.write_text(''.join(json.dumps({k:v for k,v in t.items() if k!='prompt_token_ids'},ensure_ascii=False)+'\n' for t in tasks))
        metrics={'attempted_spans':0,'input_tokens':0,'output_tokens':0,'generation_seconds':0.0}
        common={'packet_sha256':packet['packet_sha256'],'run_config_sha256':config_sha,
            'requests_sha256':_sha256(request_path),'cases':len(packet['cases']),'distinct_text_kind_pairs':len(by_kind),
            'spans':len(tasks),'max_prompt_tokens':max(len(t['prompt_token_ids']) for t in tasks),
            'prompt_tokens':sum(len(t['prompt_token_ids']) for t in tasks),
            'classification_accuracy_established':False,'selection_decision':None,'dry_run':args.dry_run}
        if args.dry_run:
            _write_json(output/'summary.json',{**common,'complete':True,'inference_performed':False})
            print(json.dumps(common,indent=2))
            return 0
        provenance={k:config[k] for k in ('model','model_revision','prompt_version')}
        provenance['run_config_sha256']=config_sha
        (output/'decisions').mkdir(exist_ok=True)
        states={t['key']:engine._cached(output/'decisions'/f"{t['key']}.json",t,
            documents[t['content_hash']]['text'][t['start']:t['end']],active_rubric.parse_response,provenance) for t in tasks}
        pending=[t for t in tasks if not states[t['key']] or states[t['key']]['parse_status']!='ok']
        model_load=0.0
        if pending:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            started=time.monotonic()
            llm=(model_factory or LLM)(model=args.model,revision=args.model_revision,tokenizer=str(snapshot),tokenizer_revision=args.model_revision,
                trust_remote_code=False,dtype='bfloat16',max_model_len=args.max_model_len,max_num_seqs=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,gpu_memory_utilization=.85,
                enable_prefix_caching=True,enforce_eager=not args.enable_cuda_graphs,seed=0,
                **({'structured_outputs_config':{'backend':'xgrammar','disable_any_whitespace':True}} if args.compact_json else {}),
                **({'language_model_only':True} if args.language_model_only else {}))
            model_load=time.monotonic()-started
            for offset in range(0,len(pending),args.batch_size):
                batch=pending[offset:offset+args.batch_size]
                sampling=[SamplingParams(temperature=0,max_tokens=args.max_output_tokens,seed=0,
                    structured_outputs=StructuredOutputsParams(json=t['response_schema'])) for t in batch]
                results,timing=engine.score_batch(llm,sampling,batch,documents,output,args.model,args.model_revision,
                    active_rubric.parse_response,active_rubric.VERSION,config_sha)
                states.update({t['key']:r for t,r in zip(batch,results)})
                metrics['attempted_spans']+=len(batch)
                metrics['input_tokens']+=timing['input_tokens']
                metrics['output_tokens']+=timing['output_tokens']
                metrics['generation_seconds']+=timing['seconds']
                logging.info('Processed %d/%d pending spans',min(offset+len(batch),len(pending)),len(pending))
        rows=[]
        for task in tasks:
            rows.append({**{k:task[k] for k in ('key','kind','content_hash','start','end')},
                'case_ids':by_kind[(task['kind'],task['content_hash'])],
                **{k:states[task['key']].get(k) for k in ('parse_status','labels','evidence_offsets','error')},
                'selection_decision':None})
        errors=Counter(r['error'] for r in rows if r['parse_status']!='ok')
        resolved=output/'resolved_spans.jsonl'
        resolved.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        _write_json(output/'summary.json',{**common,'complete':not errors,'inference_performed':bool(pending),
            'unresolved_spans':sum(errors.values()),'validation_errors':dict(errors),
            'generation_this_invocation':metrics,'model_load_seconds_this_invocation':model_load,
            'resolved_spans_sha256':_sha256(resolved),
            'scope':'Development labels only; evidence coverage is not classification accuracy. No final selection.'})
        return 2 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
