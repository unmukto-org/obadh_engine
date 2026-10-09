"""Append independently recovered and critiqued grammar pairs to byte data.

Keep the original validation partition and reject benchmark overlap, conflicting
labels, ambiguous honorific/punctuation changes, and capacity violations.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil

from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.byte_data import read_rows, fits
from tools.correction.contextual_data import verify
from tools.correction.roundtrip_data import admitted
from tools.correction.teacher_roundtrip import select_bank


def prepare(args):
    parent, byte_base = verify(args.parent), verify(args.byte_base)
    report = json.loads((args.teacher / 'report.json').read_text())
    contract = json.loads((args.teacher / 'contract.json').read_text())
    if not report['completed'] or report['contract'] != contract or report['rows_sha256'] != sha256_file(args.teacher / 'rows.jsonl'):
        raise ValueError('Complete verified teacher results required')
    source, bank = select_bank(args.source_data, contract['count'], contract.get('source_offset', 0))
    if source['data_id'] != contract['data_id'] or digest_json(bank) != contract['bank_sha256']:
        raise ValueError('Teacher source bank mismatch')
    if parent['parent_data_id'] != byte_base['data_id'] or byte_base['base_data_id'] != source['data_id']:
        raise ValueError('Context, byte base and teacher source lineage mismatch')
    forbidden = {r[k] for r in read_rows(args.parent / 'validation.jsonl.gz') for k in ('source', 'target')}
    for path in args.exclude:
        forbidden |= {r[k] for r in map(json.loads, path.read_text().splitlines()) for k in ('source', 'target')}
    teacher_rows = [json.loads(s) for s in (args.teacher / 'rows.jsonl').read_text().splitlines()]
    exclusions = Counter()
    proposed = admitted(bank, teacher_rows, exclusions)
    candidates, ambiguous = {}, set()
    for row in proposed:
        item = dict(id='teacher:' + row['id'], source=row['source_text'], target=row['target'], mode=row['mode'],
                    kind='teacher/' + row['kind'], family='teacher', teacher_parent_id=row['parent_id'])
        if item['source'] in forbidden or item['target'] in forbidden or not fits(item, parent['byte_limit']):
            exclusions['heldout_or_capacity'] += 1
            continue
        key = (item['source'], item['mode'])
        if key in candidates and candidates[key]['target'] != item['target']:
            ambiguous.add(key)
        candidates[key] = item
    for row in read_rows(args.parent / 'train.jsonl.gz'):
        key = (row['source'], row['mode'])
        if key in candidates:
            if candidates[key]['target'] != row['target']:
                ambiguous.add(key)
            elif row['family'] == 'teacher':
                ambiguous.add(key)
                exclusions['existing_teacher_duplicate'] += 1
    selected = [r for k, r in candidates.items() if k not in ambiguous]
    errors = sum(r['source'] != r['target'] for r in selected)
    if errors < args.minimum_errors:
        raise ValueError(f'Only {errors} admitted new grammar corrections')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / '.building').touch()
    counts = Counter()
    with gzip.open(args.output / 'train.jsonl.gz', 'wt', encoding='utf-8') as out:
        for rows in (read_rows(args.parent / 'train.jsonl.gz'), selected):
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[row['kind']] += 1
    shutil.copyfile(args.parent / 'validation.jsonl.gz', args.output / 'validation.jsonl.gz')
    manifest = dict(version=1, kind=parent['kind'], parent_data_id=parent['data_id'], byte_limit=parent['byte_limit'],
                    prefixes=parent['prefixes'], counts=dict(train=dict(counts), validation=parent['counts']['validation']),
                    added_grammar_errors=errors, added_rows=len(selected), exclusions=dict(exclusions),
                    conflicting_or_duplicate_inputs=len(ambiguous), generation=contract,
                    teacher_rows_sha256=report['rows_sha256'], builder_sha256=sha256_file(Path(__file__)),
                    admission_policy_sha256=sha256_file(Path(__file__).with_name('roundtrip_data.py')),
                    excluded_banks={str(p): sha256_file(p) for p in args.exclude}, scope=__doc__,
                    files=[dict(path=p.name, sha256=sha256_file(p)) for p in sorted(args.output.glob('*.gz'))])
    manifest['data_id'] = digest_json(manifest)
    write_json(args.output / 'manifest.json', manifest)
    (args.output / '.building').unlink()
    print(json.dumps(dict(data_id=manifest['data_id'], added_grammar_errors=errors, added_rows=len(selected), exclusions=dict(exclusions))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'byte-base', 'source-data', 'teacher', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--exclude', type=Path, action='append', default=[])
    p.add_argument('--minimum-errors', type=int, default=256)
    args = p.parse_args()
    if args.minimum_errors < 1:
        p.error('positive admission minimum required')
    prepare(args)


if __name__ == '__main__':
    main()
