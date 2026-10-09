"""Select continuation weights on a common calibration bank, never evaluation rows.

Exact-reference correction F0.5 includes missed errors and false changes to clean
text. This development criterion does not certify grammatical correctness.
"""
import argparse
import json
from pathlib import Path
from tools.corpus.provenance import sha256_file, write_json


def metrics(rows):
    errors = sum(r['source'] != r['target'] for r in rows)
    tp = sum(r['source'] != r['target'] and r['answer'] == r['target'] for r in rows)
    fp = sum(r['answer'] != r['source'] and r['answer'] != r['target'] for r in rows)
    fn = errors - tp
    clean = sum(r['source'] == r['target'] for r in rows)
    preserved = sum(r['source'] == r['target'] == r['answer'] for r in rows)
    return dict(count=len(rows), errors=errors, correct_repairs=tp, incorrect_changes=fp, missed_errors=fn,
                clean=clean, clean_preserved=preserved, f05=1.25 * tp / (1.25 * tp + .25 * fn + fp) if tp else 0.)


def select(runs, *, baseline=None):
    banks, scores, receipts = {}, {}, {}
    for name, root in runs.items():
        report = json.loads((root / 'report.json').read_text())
        if (root / '.building').exists() or not report['completed'] or sha256_file(root / 'rows.jsonl') != report['rows_sha256']:
            raise ValueError('Incomplete selection predictions')
        rows = [json.loads(s) for s in (root / 'rows.jsonl').read_text().splitlines()]
        banks[name] = [(r['id'], r['source'], r['target'], r['mode']) for r in rows]
        scores[name] = metrics(rows)
        receipts[name] = dict(report_sha256=sha256_file(root / 'report.json'), **report['contract'])
    if len(runs) < 2 or any(bank != next(iter(banks.values())) for bank in banks.values()):
        raise ValueError('Selection requires paired candidate banks')
    if len({r['bank_sha256'] for r in receipts.values()}) != 1:
        raise ValueError('Selection bank identity mismatch')
    if baseline is not None:
        if baseline not in scores:
            raise ValueError('Required baseline absent')
        eligible = admissible(scores, baseline)
        # Keep the baseline on exact ties; a new artifact needs an actual gain.
        chosen = max([baseline] + sorted(n for n in eligible if n != baseline),
                     key=lambda n: (scores[n]['correct_repairs'], scores[n]['clean_preserved'], -scores[n]['incorrect_changes']))
        objective = 'Maximize repairs with no loss of clean preservation or increase in incorrect changes versus baseline on calibration'
    else:
        eligible = list(scores)
        chosen = max(sorted(scores), key=lambda n: (scores[n]['f05'], scores[n]['clean_preserved'], scores[n]['correct_repairs']))
        objective = 'Guarded exact-reference correction F0.5 on the common calibration bank'
    result = dict(selected=chosen, metrics=scores, receipts=receipts, objective=objective, scope=__doc__)
    if baseline is not None:
        result.update(baseline=baseline, eligible=eligible)
    return result


def admissible(scores, baseline):
    reference = scores[baseline]
    return [name for name, value in scores.items()
            if value['correct_repairs'] >= reference['correct_repairs']
            and value['clean_preserved'] >= reference['clean_preserved']
            and value['incorrect_changes'] <= reference['incorrect_changes']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, help='NAME=calibration-predictions-directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', help='Require nonregression against this calibration run')
    args = p.parse_args()
    parts = [value.split('=', 1) for value in args.run]
    runs = {name: Path(value) for name, value in parts}
    if len(parts) != len(runs): p.error('duplicate candidate label')
    result = select(runs, baseline=args.baseline)
    result['selector_sha256'] = sha256_file(Path(__file__))
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError('Existing initialization selection differs')
    write_json(args.output, result)
    print(json.dumps(dict(selected=result['selected'], metrics=result['metrics'])), flush=True)


if __name__ == '__main__':
    main()
