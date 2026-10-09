"""Resumable candidate-coverage diagnostic for the offline ByT5 reference.

Decode without references. Report hindsight candidate coverage separately from
actual top-1 accuracy; an oracle ceiling is not an achievable deployment claim.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
import torch
from tools.corpus.provenance import digest_json,sha256_file,write_json
from tools.correction.byte_data import PREFIX,encode,decode
from tools.correction.byte_train import load_model
from tools.autosuggest.compare_proofreaders import normalize,protected


def package(source, sequences, scores):
    candidates=[];seen=set()
    for rank,(tokens,score) in enumerate(zip(sequences,scores),1):
        if not math.isfinite(score):raise ValueError('Nonfinite beam score')
        raw,reason=decode(tokens)
        if raw is not None and protected(raw)!=protected(source):reason='protected_span_changed'
        answer=source if reason else raw
        if answer in seen:continue
        seen.add(answer)
        candidates.append(dict(rank=rank,raw_answer=raw,answer=answer,fallback_reason=reason,beam_score=score))
    if source not in seen:candidates.append(dict(rank=None,raw_answer=source,answer=source,fallback_reason=None,beam_score=None))
    return candidates


def summarize(rows):
    groups={}
    for name in ['all']+sorted({r['kind'] for r in rows}):
        c=Counter()
        for row in rows:
            if name!='all' and row['kind']!=name:continue
            error=row['source']!=row['target'];top=row['candidates'][0]['answer']
            c['count']+=1;c['errors']+=error;c['clean']+=not error
            c['top1_repairs']+=error and top==row['target']
            c['oracle_repairs']+=error and any(p['answer']==row['target'] for p in row['candidates'])
            c['top1_clean_preserved']+=not error and top==row['source']
        groups[name]=dict(c)
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','pretrained','bank','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--beams',type=int,default=4);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--limit-count',type=int);p.add_argument('--stop-epoch',type=float,required=True)
    a=p.parse_args()
    if not 1<=a.beams<=8 or a.batch<1 or a.limit_count is not None and a.limit_count<1:p.error('Invalid candidate budget')
    torch.set_num_threads(8)
    bank=list(map(json.loads,a.bank.read_text().splitlines()))
    if a.limit_count:bank=bank[:a.limit_count]
    contract=dict(checkpoint_sha256=sha256_file(a.checkpoint),bank_sha256=digest_json(bank),beams=a.beams,batch=a.batch,
                  length_penalty=1.,script_sha256=sha256_file(Path(__file__)),decoder_sha256=sha256_file(Path(__file__).with_name('byte_data.py')))
    a.output.mkdir(parents=True,exist_ok=True);cp=a.output/'contract.json';rows_path=a.output/'rows.jsonl'
    if cp.exists() and json.loads(cp.read_text())!=contract:raise ValueError('Resume contract mismatch')
    if not cp.exists():
        if any(a.output.iterdir()):raise ValueError('Unrecognized output contents')
        write_json(cp,contract)
    rows=list(map(json.loads,rows_path.read_text().splitlines())) if rows_path.exists() else []
    if len(rows)>len(bank) or any(any(row[k]!=(normalize(expected[k]) if k in ('source','target') else expected[k]) for k in ('id','kind','source','target','mode')) for row,expected in zip(rows,bank)):
        raise ValueError('Invalid resumed candidate prefix')
    report_path=a.output/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report.get('completed'):
            if len(rows)!=len(bank) or report['rows_sha256']!=sha256_file(rows_path):raise ValueError('Invalid completed candidate receipt')
            print(json.dumps(dict(event='candidate_audit_complete',rows=len(rows),reloaded=True)),flush=True);return
    (a.output/'.building').touch()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);limit=state['contract']['byte_limit']
    model=load_model(a.pretrained);model.load_state_dict(state['state_dict'],strict=True);del state
    model.to('cuda').eval()
    with rows_path.open('a') as output,torch.inference_mode():
        for start in range(len(rows),len(bank),a.batch):
            if time.time()>=a.stop_epoch:raise RuntimeError('Time limit; completed batches can resume')
            batch=[dict(r,source=normalize(r['source']),target=normalize(r['target'])) for r in bank[start:start+a.batch]]
            encoded=[encode(PREFIX[r['mode']]+r['source']) for r in batch]
            if any(len(s)>limit for s in encoded):raise ValueError('Candidate bank exceeds model capacity')
            longest=max(map(len,encoded));source=torch.tensor([s+[0]*(longest-len(s)) for s in encoded],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model.generate(input_ids=source,attention_mask=source.ne(0),max_new_tokens=limit,
                    do_sample=False,num_beams=a.beams,num_return_sequences=a.beams,length_penalty=1.,use_cache=True,
                    return_dict_in_generate=True,output_scores=True,pad_token_id=0,eos_token_id=1,decoder_start_token_id=0,
                    suppress_tokens=[0,2]+list(range(259,model.config.vocab_size)))
            seq=result.sequences.tolist();scores=result.sequences_scores.tolist() if a.beams>1 else [0.]*len(seq)
            for i,row in enumerate(batch):
                record=dict(row,candidates=package(row['source'],seq[i*a.beams:(i+1)*a.beams],scores[i*a.beams:(i+1)*a.beams]))
                output.write(json.dumps(record,ensure_ascii=False)+'\n');rows.append(record)
            output.flush();os.fsync(output.fileno())
            del result,source
            print(json.dumps(dict(event='candidate_audit_progress',rows=len(rows),total=len(bank))),flush=True)
    temp=report_path.with_suffix('.tmp');write_json(temp,dict(completed=True,contract=contract,rows=len(rows),
        groups=summarize(rows),peak_cuda_bytes=torch.cuda.max_memory_allocated(),rows_sha256=sha256_file(rows_path),scope=__doc__))
    temp.replace(report_path);(a.output/'.building').unlink()
    print(json.dumps(dict(event='candidate_audit_complete',rows=len(rows),groups=summarize(rows)['all'])),flush=True)


if __name__=='__main__':main()
