"""Add training-only lexical evidence to a learned edit-acceptance policy."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import math
from pathlib import Path

from tools.corpus.provenance import sha256_file, write_json
from tools.correction.acceptance import FEATURES, LEXICAL_FEATURES, load_scored
from tools.correction.byte_data import read_rows
from tools.correction.contextual_data import verify, words


def lexical_features(source, candidate, vocabulary):
    a, b = [w.group() for w in words(source)], [w.group() for w in words(candidate)]
    changed_a, changed_b = [], []
    for tag, i, j, k, l in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != 'equal':
            changed_a.extend(a[i:j]);changed_b.extend(b[k:l])
    def stats(tokens):
        counts = [vocabulary.get(w, 0) for w in tokens]
        return sum(n == 0 for n in counts) / max(1, len(counts)), sum(math.log1p(n) for n in counts) / max(1, len(counts))
    a_oov, a_frequency = stats(changed_a)
    b_oov, b_frequency = stats(changed_b)
    return [a_oov, b_oov, a_frequency, b_frequency, len(changed_a), len(changed_b), b_frequency - a_frequency,
            int(bool(changed_a and changed_b) and a_oov == 1 and b_oov == 0)]


def vocabulary(args):
    receipt = verify(args.data)
    counts, seen, lexical_words = Counter(), set(), set()
    for row in read_rows(args.data / 'train.jsonl.gz'):
        if row['source'] != row['target']:
            continue
        if row['family'] == 'lexical':
            lexical_words.add(row['target'])
        if row['kind'] not in ('base/identity', 'teacher/identity') or row['mode'] != 0 or row['target'] in seen:
            continue
        seen.add(row['target'])
        counts.update(w.group() for w in words(row['target']) if any('\u0980' <= c <= '\u09ff' and c.isalpha() for c in w.group()))
    for word in lexical_words:
        counts[word] = max(1, counts[word])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / '.building').touch()
    write_json(args.output / 'vocabulary.json', dict(sorted(counts.items())))
    write_json(args.output / 'report.json', dict(completed=True, data_id=receipt['data_id'], words=len(counts),
               natural_sentence_identities=len(seen), lexical_training_words=len(lexical_words),
               vocabulary_sha256=sha256_file(args.output / 'vocabulary.json'), script_sha256=sha256_file(Path(__file__)),
               scope='Training references only, deduplicated natural sentence identities and lexical training words. No validation/evaluation text.'))
    (args.output / '.building').unlink()
    print(json.dumps(dict(words=len(counts), natural_sentence_identities=len(seen))), flush=True)


def augment(args):
    report = json.loads((args.vocabulary / 'report.json').read_text())
    vocab_path = args.vocabulary / 'vocabulary.json'
    if (args.vocabulary / '.building').exists() or not report['completed'] or sha256_file(vocab_path) != report['vocabulary_sha256']:
        raise ValueError('Incomplete lexical vocabulary')
    vocabulary = json.loads(vocab_path.read_text())
    receipt, rows = load_scored(args.scored)
    if receipt['features'] != FEATURES:
        raise ValueError('Lexical augmentation requires the base feature schema')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / '.building').touch()
    for row in rows:
        if row['features'] is not None:
            row['features'] += lexical_features(row['source'], row['raw_answer'], vocabulary)
    (args.output / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    receipt.update(features=FEATURES + LEXICAL_FEATURES, lexical_vocabulary_sha256=report['vocabulary_sha256'],
                   base_feature_report_sha256=sha256_file(args.scored / 'report.json'),
                   augmentation_script_sha256=sha256_file(Path(__file__)), rows_sha256=sha256_file(args.output / 'rows.jsonl'))
    write_json(args.output / 'report.json', receipt)
    (args.output / '.building').unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('vocabulary')
    builder.add_argument('--data', type=Path, required=True)
    builder.add_argument('--output', type=Path, required=True)
    augmentation = commands.add_parser('augment')
    for name in ('vocabulary', 'scored', 'output'):
        augmentation.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    (vocabulary if args.command == 'vocabulary' else augment)(args)


if __name__ == '__main__':
    main()
