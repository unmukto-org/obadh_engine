"""Training-only repair/preservation preferences from existing source labels.

This adapts edit-wise preference supervision to Bangla graphemes and UTF-8 bytes;
it is not an exact reproduction of the English/Chinese ERRANT-based method.
"""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import regex
from tools.corpus.provenance import digest_json,sha256_file,write_json
from tools.correction.byte_data import encode,read_rows
from tools.correction.contextual_data import verify
from tools.correction.acceptance_union import text_hash


def edit_weights(text,other,edit=2.,pivot=4.):
    a,b=regex.findall(r'\X',text),regex.findall(r'\X',other)
    weights=[1.]*(len(a)+1)
    for tag,i,j,_,_ in SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
        if tag=='equal':continue
        for k in range(i,max(j,i+1)):weights[k]=max(weights[k],edit)
        weights[i]=max(weights[i],pivot)
    return [w for cluster,w in zip(a,weights) for _ in cluster.encode('utf-8')]+[weights[-1]]


def verified_rows(path):
    report=json.loads((path/'report.json').read_text())
    if (path/'.building').exists() or not report['completed'] or sha256_file(path/'rows.jsonl')!=report['rows_sha256']:
        raise ValueError('Incomplete preference source')
    return report,list(map(json.loads,(path/'rows.jsonl').read_text().splitlines()))


def build(args):
    parent=verify(args.parent);receipt,rows=verified_rows(args.candidates);mining_receipt,mining=verified_rows(args.mining)
    forbidden={text_hash(r[k]) for r in read_rows(args.parent/'validation.jsonl.gz') for k in ('source','target')}
    for path in args.exclude:forbidden.update(text_hash(r[k]) for r in map(json.loads,path.read_text().splitlines()) for k in ('source','target'))
    pairs={};counts=Counter()
    def add(r,negative,origin):
        source,target=r['source'],r['target']
        if target==negative:return
        if any(text_hash(t) in forbidden for t in (source,target,negative)):
            counts['heldout_excluded']+=1;return
        if max(len(encode(t)) for t in (source,target,negative))>parent['byte_limit']:
            counts['capacity_excluded']+=1;return
        key=(source,target,negative,r['mode'])
        pairs[key]=dict(id='preference:'+digest_json(key),source=source,chosen=target,rejected=negative,mode=r['mode'],
                       group='preserve' if source==target else 'repair',origin=origin)
    for row in rows:
        wrong=[c for c in row['candidates'] if not c['fallback_reason'] and c['answer']!=row['target']]
        if wrong:add(row,wrong[0]['answer'],'beam_training')
    for row in mining:
        if row['source']==row['target'] and not row['fallback_reason'] and row['answer']!=row['source']:
            add(row,row['answer'],'mined_training_identity')
    # Prove every chosen source/target label exists in the parent training split.
    required={(r['source'],r['chosen'],r['mode']) for r in pairs.values()};unverified=set(required)
    for r in read_rows(args.parent/'train.jsonl.gz'):unverified.discard((r['source'],r['target'],r['mode']))
    if unverified:raise ValueError('Preference labels absent from parent training partition')
    selected=sorted(pairs.values(),key=lambda r:r['id']);groups=Counter(r['group'] for r in selected)
    if min(groups.get('repair',0),groups.get('preserve',0))<100:raise ValueError('Insufficient repair/preservation balance')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'.building').touch()
    (args.output/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    manifest=dict(parent_data_id=parent['data_id'],byte_limit=parent['byte_limit'],rows=len(selected),groups=dict(groups),
        excluded_banks={str(p):sha256_file(p) for p in args.exclude},candidate_contract=receipt['contract'],
        mining_contract=mining_receipt['contract'],candidate_rows_sha256=receipt['rows_sha256'],mining_rows_sha256=mining_receipt['rows_sha256'],
        pairs_sha256=sha256_file(args.output/'pairs.jsonl'),builder_sha256=sha256_file(Path(__file__)),exclusions=dict(counts),scope=__doc__)
    manifest['data_id']=digest_json(manifest);write_json(args.output/'manifest.json',manifest);(args.output/'.building').unlink()
    print(json.dumps(dict(event='preference_data_complete',rows=len(selected),groups=dict(groups))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','candidates','mining','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--exclude',action='append',type=Path,default=[])
    build(p.parse_args())


if __name__=='__main__':main()
