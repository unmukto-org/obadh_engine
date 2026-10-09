"""Offline multi-generator correction selector; not a mobile deployment artifact.

Fit on training proposals, select a threshold on separate calibration sentences,
then apply the frozen policy to evaluation banks. Features never read references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np

from tools.corpus.provenance import sha256_file, write_json
from tools.correction.acceptance import load_scored, fit_logistic, probabilities


def text_hash(text):
    return hashlib.sha256(unicodedata.normalize('NFC', text.strip()).encode()).hexdigest()


def load_pair(left, right):
    a, rows_a = load_scored(left)
    b, rows_b = load_scored(right)
    if a['features'] != b['features'] or a.get('lexical_vocabulary_sha256') != b.get('lexical_vocabulary_sha256'):
        raise ValueError('incompatible feature schema or vocabulary')
    keys = ('id', 'source', 'target', 'mode', 'kind')
    identities = lambda rows: [tuple(r[k] for k in keys) for r in rows]
    if identities(rows_a) != identities(rows_b) or len({r['id'] for r in rows_a}) != len(rows_a):
        raise ValueError('unaligned or duplicate proposal rows')
    if a['generator_contract']['bank_sha256'] != b['generator_contract']['bank_sha256']:
        raise ValueError('unpaired generator banks')
    return [a, b], list(zip(rows_a, rows_b))


def candidates(pair):
    """Unique safe proposals with fixed model-order features and support flags."""
    available = [r['features'] is not None and not r['fallback_reason']
                 and isinstance(r['raw_answer'], str) and bool(r['raw_answer'].strip())
                 and r['raw_answer'] != r['source'] for r in pair]
    answers = list(dict.fromkeys(r['raw_answer'] for r, ok in zip(pair, available) if ok))
    dimension = next((len(r['features']) for r, ok in zip(pair, available) if ok), 0)
    feature_blocks = [r['features'] if ok else [0.] * dimension for r, ok in zip(pair, available)]
    return [dict(answer=answer, features=feature_blocks[0] + feature_blocks[1] +
                 [int(ok) for ok in available] +
                 [int(ok and r['raw_answer'] == answer) for r, ok in zip(pair, available)]) for answer in answers]


def choose(model, pair):
    proposals = candidates(pair)
    if not proposals:
        return dict(answer=pair[0]['source'], probability=None)
    scores = probabilities(model, [p['features'] for p in proposals])
    index = int(np.argmax(scores))  # Stable model-order tie break.
    return dict(answer=proposals[index]['answer'], probability=float(scores[index]))


def select_threshold(chosen, rows):
    errors = sum(r['source'] != r['target'] for r in rows)
    options = []
    for threshold in sorted({0., 1., *[c['probability'] for c in chosen if c['probability'] is not None]}):
        accepted = [(c, r) for c, r in zip(chosen, rows) if c['probability'] is not None and c['probability'] >= threshold]
        tp = sum(c['answer'] == r['target'] for c, r in accepted)
        fp = len(accepted) - tp
        fn = errors - tp  # Includes errors for which neither model offered a repair.
        options.append(dict(threshold=threshold, f05=1.25 * tp / (1.25 * tp + fp + .25 * fn) if tp else 0.,
                            correct=tp, incorrect=fp, missed=fn))
    return max(options, key=lambda v: (v['f05'], -v['incorrect'], v['threshold']))


def identity(receipts):
    return [dict(checkpoint_sha256=r['checkpoint_sha256'], features=r['features'],
                 lexical_vocabulary_sha256=r.get('lexical_vocabulary_sha256')) for r in receipts]


def fit(args):
    train_receipts, train = load_pair(args.train_left, args.train_right)
    calibration_receipts, calibration = load_pair(args.calibration_left, args.calibration_right)
    if identity(train_receipts) != identity(calibration_receipts):
        raise ValueError('generator identity changed across splits')
    hashes = lambda bank: {text_hash(pair[0][k]) for pair in bank for k in ('source', 'target')}
    train_hashes, calibration_hashes = hashes(train), hashes(calibration)
    if train_hashes & calibration_hashes:
        raise ValueError('training/calibration text overlap')
    proposals = [(p['features'], p['answer'] == pair[0]['target']) for pair in train for p in candidates(pair)]
    if len(proposals) < 50 or len({y for _, y in proposals}) != 2:
        raise ValueError('insufficient positive/negative training proposals')
    model = fit_logistic(np.asarray([x for x, _ in proposals]), np.asarray([y for _, y in proposals], dtype=float))
    selected = select_threshold([choose(model, pair) for pair in calibration], [pair[0] for pair in calibration])
    names = train_receipts[0]['features']
    model.update(features=['left/' + n for n in names] + ['right/' + n for n in names] +
                 ['left_available', 'right_available', 'left_support', 'right_support'],
                 generators=identity(train_receipts), calibration=selected, threshold=selected['threshold'],
                 selection='Maximum sentence-level exact-reference F0.5 on disjoint calibration; all missed errors count',
                 receipts=dict(train=train_receipts, calibration=calibration_receipts),
                 excluded_text_hashes=sorted(train_hashes | calibration_hashes),
                 fitter_sha256=sha256_file(Path(__file__)), numpy=np.__version__, scope=__doc__)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'model.json', model)
    write_json(args.output / 'report.json', dict(completed=True, model_sha256=sha256_file(args.output / 'model.json'),
               training_proposals=len(proposals), calibration=selected))
    print(json.dumps(dict(event='union_fit', calibration=selected)), flush=True)


def evaluate(args):
    receipt = json.loads((args.model / 'report.json').read_text())
    if not receipt['completed'] or sha256_file(args.model / 'model.json') != receipt['model_sha256']:
        raise ValueError('invalid selector artifact')
    model = json.loads((args.model / 'model.json').read_text())
    receipts, pairs = load_pair(args.left, args.right)
    if identity(receipts) != model['generators']:
        raise ValueError('generator identity mismatch')
    excluded = set(model['excluded_text_hashes'])
    if any(text_hash(pair[0][k]) in excluded for pair in pairs for k in ('source', 'target')):
        raise ValueError('evaluation text overlaps training/calibration')
    result = []
    for pair in pairs:
        proposed = choose(model, pair)
        accepted = proposed['probability'] is not None and proposed['probability'] >= model['threshold']
        answer = proposed['answer'] if accepted else pair[0]['source']
        result.append(dict(pair[0], raw_answer=answer, answer=answer, proposal_answer=proposed['answer'],
                           acceptance_score=proposed['probability'], accepted_edit=accepted,
                           proposals=[r['raw_answer'] for r in pair],
                           features=None, reference_correct=answer == pair[0]['target'],
                           capacity_exceeded=all(r['capacity_exceeded'] for r in pair),
                           fallback_reason=None if accepted else 'union_abstained'))
    from tools.correction.evaluate import measure
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / '.building').touch()
    (args.output / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    contract = dict(bank_sha256=receipts[0]['generator_contract']['bank_sha256'],
                    generator_checkpoints=[r['checkpoint_sha256'] for r in receipts],
                    acceptance_model_sha256=receipt['model_sha256'], policy='two-generator union with calibrated selection')
    write_json(args.output / 'contract.json', contract)
    write_json(args.output / 'report.json', dict(completed=True, contract=contract, groups=measure(result),
               evaluation_receipts=receipts, rows_sha256=sha256_file(args.output / 'rows.jsonl'), scope=__doc__))
    (args.output / '.building').unlink()
    print(json.dumps(dict(event='union_evaluation', rows=len(result), groups=measure(result)['all'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    f = commands.add_parser('fit')
    for name in ('train-left', 'train-right', 'calibration-left', 'calibration-right', 'output'):
        f.add_argument('--' + name, type=Path, required=True)
    e = commands.add_parser('evaluate')
    for name in ('model', 'left', 'right', 'output'):
        e.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    (fit if args.command == 'fit' else evaluate)(args)


if __name__ == '__main__':
    main()
