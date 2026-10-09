"""Evaluate isolated-word spelling errors inside held-out corpus contexts.

These are controlled synthetic cases, not human typing gold. Contexts and lexical
labels come exclusively from the existing validation partitions. A pretrained
model can already know these words; this is not a pretraining-contamination test.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import unicodedata

from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.byte_data import read_rows, fits, lexical_reason, word_partition
from tools.correction.contextual_data import verify, words, replace_word


def build(args):
    receipt = verify(args.data)
    forbidden = set()
    for path in args.exclude:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            forbidden.update((row['source'], row['target']))
    validation = list(read_rows(args.data / 'validation.jsonl.gz'))
    contexts = [r for r in validation if r['family'] == 'base' and r['source'] == r['target']]
    if sha256_file(args.lexical_csv) != receipt['lexical_csv_sha256']:
        raise ValueError('External lexical source hash mismatch')
    with args.lexical_csv.open(encoding='utf-8-sig', newline='') as handle:
        external = list(csv.DictReader(handle))
    valid_words = {unicodedata.normalize('NFC', r['Word']).strip() for r in external}
    valid_words |= {w.group() for r in contexts for w in words(r['target'])}
    variants = defaultdict(list)
    for index, row in enumerate(external):
        source, target = (unicodedata.normalize('NFC', row[k]).strip() for k in ('Error', 'Word'))
        # The original lexical builder excluded all sentence-validation
        # vocabulary before partitioning. Read those untouched CSV rows here,
        # keeping only the word-hash validation partition used by augmentation.
        if word_partition(target) != 'validation' or lexical_reason(source, target, row['ErrorType'], valid_words):
            continue
        variants[target].append(dict(source=source, target=target, kind='external/' + row['ErrorType'], id='csv:' + str(index)))
    rng = random.Random(args.seed)
    rng.shuffle(contexts)
    bank, metadata, used_text, used_words = [], [], set(), Counter()
    for original in contexts:
        target = original['target']
        if target in forbidden or target in used_text:
            continue
        spans = [w for w in words(target) if w.group() in variants and used_words[w.group()] < args.per_word]
        rng.shuffle(spans)
        for span in spans:
            options = list(variants[span.group()])
            rng.shuffle(options)
            for lexical in options:
                source = replace_word(target, span, lexical['source'])
                row = dict(source=source, target=target, mode=original['mode'])
                if source in forbidden or source in used_text or not fits(row, receipt['byte_limit']):
                    continue
                identifier = digest_json(row)
                bank.append(dict(row, id='contextual-held:' + identifier, kind='contextual_held/' + lexical['kind']))
                bank.append(dict(source=target, target=target, mode=original['mode'], id='contextual-clean:' + identifier, kind='contextual_held/identity'))
                metadata.append(dict(case_id=identifier, context_parent=original['id'], lexical_parent=lexical['id']))
                used_text.update((source, target))
                used_words[span.group()] += 1
                break
            else:
                continue
            break
        if len(metadata) >= args.count:
            break
    if len(metadata) < args.minimum:
        raise ValueError(f'Only {len(metadata)} independent contexts; require {args.minimum}')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / '.building').touch()
    (args.output / 'bank.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in bank))
    write_json(args.output / 'manifest.json', dict(scope=__doc__, data_id=receipt['data_id'],
               errors=len(metadata), clean_controls=len(metadata), distinct_corrected_words=len(used_words),
               seed=args.seed, per_word=args.per_word, bank_sha256=sha256_file(args.output / 'bank.jsonl'),
               lexical_csv_sha256=receipt['lexical_csv_sha256'],
               exclude_files={str(p): sha256_file(p) for p in args.exclude}, lineage=metadata,
               builder_sha256=sha256_file(Path(__file__))))
    (args.output / '.building').unlink()
    print(json.dumps(dict(errors=len(metadata), distinct_words=len(used_words))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--lexical-csv', type=Path, required=True)
    p.add_argument('--exclude', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=512)
    p.add_argument('--minimum', type=int, default=128)
    p.add_argument('--per-word', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261006)
    args = p.parse_args()
    if not 0 < args.minimum <= args.count or args.per_word < 1:
        p.error('positive limits and minimum <= count required')
    build(args)


if __name__ == '__main__':
    main()
