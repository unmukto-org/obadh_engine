"""Mine unwanted edits on existing training identities for preservation replay.

Only original training references supply labels. Model outputs identify difficult
identities; they never become correction targets or synthetic error inputs.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import unicodedata
from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.byte_data import read_rows
from tools.correction.contextual_data import verify


def normalized(text):
    return unicodedata.normalize('NFC', text.strip())


def held_texts(parent, excludes):
    held = {normalized(r[k]) for r in read_rows(parent / 'validation.jsonl.gz') for k in ('source', 'target')}
    for path in excludes:
        held.update(normalized(r[k]) for r in map(json.loads, path.read_text().splitlines()) for k in ('source', 'target'))
    return held


def bank(args):
    parent = verify(args.parent)
    held = held_texts(args.parent, args.exclude)
    pool, seen = [], set()
    for row in read_rows(args.parent / 'train.jsonl.gz'):
        key = (normalized(row['source']), row['mode'])
        if row['family'] != 'base' or row['source'] != row['target'] or key in seen or key[0] in held:
            continue
        seen.add(key)
        pool.append({k: row[k] for k in ('id', 'kind', 'source', 'target', 'mode')})
    random.Random(args.seed).shuffle(pool)
    selected = pool[:args.count]
    if len(selected) != args.count:
        raise ValueError('insufficient independent training identities')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'bank.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    write_json(args.output / 'manifest.json', dict(parent_data_id=parent['data_id'], rows=len(selected),
               bank_sha256=digest_json(selected), file_sha256=sha256_file(args.output / 'bank.jsonl'),
               excluded_banks={str(p): sha256_file(p) for p in args.exclude}, seed=args.seed, scope=__doc__))


def difficult_keys(rows):
    """Count only deployable unwanted changes, not rejected decoding failures."""
    return {(r['source'], r['mode']) for r in rows if r['source'] == r['target']
            and r['answer'] is not None and r['answer'] != r['source'] and not r['fallback_reason']}


def prepare(args):
    parent = verify(args.parent)
    manifest = json.loads((args.bank / 'manifest.json').read_text())
    if manifest['parent_data_id'] != parent['data_id'] or manifest['file_sha256'] != sha256_file(args.bank / 'bank.jsonl'):
        raise ValueError('mining bank parent/hash mismatch')
    bank_rows = list(map(json.loads, (args.bank / 'bank.jsonl').read_text().splitlines()))
    report = json.loads((args.predictions / 'report.json').read_text())
    if (args.predictions / '.building').exists() or not report['completed'] or report['rows_sha256'] != sha256_file(args.predictions / 'rows.jsonl'):
        raise ValueError('incomplete mining predictions')
    rows = list(map(json.loads, (args.predictions / 'rows.jsonl').read_text().splitlines()))
    fields = ('id', 'kind', 'source', 'target', 'mode')
    if digest_json(bank_rows) != manifest['bank_sha256'] or report['contract']['bank_sha256'] != manifest['bank_sha256'] or [tuple(r[k] for k in fields) for r in bank_rows] != [tuple(r[k] for k in fields) for r in rows]:
        raise ValueError('mining prediction bank mismatch')
    selected = difficult_keys(rows)
    if len(selected) < args.minimum:
        raise ValueError('insufficient mined preservation examples')
    # Replay exclusions independently, rather than trusting the bank receipt.
    held = held_texts(args.parent, args.exclude)
    if any(normalized(s) in held for s, _ in selected):
        raise ValueError('mined training/heldout overlap')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / '.building').touch()
    counts, matched = Counter(), set()
    with gzip.open(args.output / 'train.jsonl.gz', 'wt', encoding='utf-8') as out:
        for row in read_rows(args.parent / 'train.jsonl.gz'):
            key = (row['source'], row['mode'])
            if row['family'] == 'base' and row['source'] == row['target'] and key in selected:
                row = dict(row, kind='base/mined_preservation_' + str(row['mode']), original_kind=row['kind'])
                matched.add(key)
            counts[row['kind']] += 1
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    if matched != selected:
        raise ValueError('mined examples were not all original training identities')
    shutil.copyfile(args.parent / 'validation.jsonl.gz', args.output / 'validation.jsonl.gz')
    result = dict(version=1, kind=parent['kind'], parent_data_id=parent['data_id'], byte_limit=parent['byte_limit'],
                  prefixes=parent['prefixes'], counts=dict(train=dict(counts), validation=parent['counts']['validation']),
                  mined_identities=len(selected), mining_contract=report['contract'], mining_rows_sha256=report['rows_sha256'],
                  excluded_banks={str(p): sha256_file(p) for p in args.exclude},
                  builder_sha256=sha256_file(Path(__file__)), scope=__doc__,
                  files=[dict(path=p.name, sha256=sha256_file(p)) for p in sorted(args.output.glob('*.gz'))])
    result['data_id'] = digest_json(result)
    write_json(args.output / 'manifest.json', result)
    (args.output / '.building').unlink()
    print(json.dumps(dict(event='preservation_data_ready',mined_identities=len(selected),data_id=result['data_id'])),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bank');b.add_argument('--count',type=int,default=20000);b.add_argument('--seed',type=int,default=20261006)
    f = sub.add_parser('prepare');f.add_argument('--bank',type=Path,required=True);f.add_argument('--predictions',type=Path,required=True);f.add_argument('--minimum',type=int,default=100)
    for s in (b,f):
        s.add_argument('--parent',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
        s.add_argument('--exclude',type=Path,action='append',default=[])
    args=p.parse_args()
    if getattr(args,'count',1)<1 or getattr(args,'minimum',1)<1: p.error('positive counts required')
    (bank if args.command=='bank' else prepare)(args)


if __name__=='__main__':main()
