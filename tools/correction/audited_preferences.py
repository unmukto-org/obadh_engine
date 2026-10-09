"""Prepare a teacher-filtered preference arm and a size-matched random control.

No targets are rewritten. Preserve valid input over optional stylistic changes;
only repair preferences require a confirmed linguistic advantage. Teacher labels
are noisy experimental supervision, never human evaluation truth.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.preference_audit import resume_rows


def retain(pair, audit):
    if pair['group'] not in ('repair', 'preserve'):
        raise ValueError('Unknown preference group')
    if (pair['source'] == pair['chosen']) != (pair['group'] == 'preserve'):
        raise ValueError('Preference group contradicts chosen target')
    return audit['decision'] == 'confirmed' or (
        pair['group'] == 'preserve' and audit['decision'] == 'both_acceptable')


def arms(pairs, audited, seed):
    if [r['id'] for r in pairs] != [r['id'] for r in audited]:
        raise ValueError('Audit/input alignment mismatch')
    if any(r['group'] != a['group'] for r, a in zip(pairs, audited)):
        raise ValueError('Audit/input group mismatch')
    filtered = [r for r, a in zip(pairs, audited) if retain(r, a)]
    counts = Counter(r['group'] for r in filtered)
    control = []
    for group in ('repair', 'preserve'):
        pool = sorted((r for r in pairs if r['group'] == group),
                      key=lambda r: digest_json([seed, group, r['id']]))
        control.extend(pool[:counts[group]])
    return {k: sorted(v, key=lambda r: r['id']) for k, v in
            dict(filtered=filtered, control=control).items()}


def build(args):
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if (args.data / '.building').exists() or manifest['data_id'] != digest_json({k: v for k, v in manifest.items() if k != 'data_id'}):
        raise ValueError('Invalid parent preferences')
    if sha256_file(args.data / 'pairs.jsonl') != manifest['pairs_sha256']:
        raise ValueError('Parent preference digest mismatch')
    report = json.loads((args.audit / 'report.json').read_text())
    if not report['completed'] or sha256_file(args.audit / 'rows.jsonl') != report['rows_sha256']:
        raise ValueError('Complete verified audit required')
    if report['contract']['data_id'] != manifest['data_id'] or report['contract']['pairs_sha256'] != manifest['pairs_sha256']:
        raise ValueError('Audit belongs to different preferences')
    pairs = list(map(json.loads, (args.data / 'pairs.jsonl').read_text().splitlines()))
    if not (args.audit / 'rows.jsonl').read_bytes().endswith(b'\n'):
        raise ValueError('Incomplete audit record')
    audited = resume_rows(args.audit / 'rows.jsonl', pairs)
    if len(pairs) != manifest['rows'] or len(audited) != report['rows'] or len(audited) != report['total']:
        raise ValueError('Incorrect audit inventory')
    selected = arms(pairs, audited, args.seed)
    for group in ('repair', 'preserve'):
        if sum(r['group'] == group for r in selected['filtered']) < 100:
            raise ValueError('Too few audited pairs for a matched experiment')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / '.building').touch()
    for arm, rows in selected.items():
        folder = args.output / arm
        folder.mkdir()
        (folder / 'pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        receipt = dict(parent_data_id=manifest['data_id'], byte_limit=manifest['byte_limit'],
            audit_report_sha256=sha256_file(args.audit / 'report.json'), audit_rows_sha256=report['rows_sha256'],
            builder_sha256=sha256_file(Path(__file__)), seed=args.seed, arm=arm, rows=len(rows),
            groups=dict(Counter(r['group'] for r in rows)), pairs_sha256=sha256_file(folder / 'pairs.jsonl'),
            excluded_banks=manifest['excluded_banks'], scope=__doc__)
        receipt['data_id'] = digest_json(receipt)
        write_json(folder / 'manifest.json', receipt)
        # Use a disjoint cache contract for a tiny CUDA save/reload check.
        smoke = []
        for group in ('repair', 'preserve'):
            smoke.extend([r for r in rows if r['group'] == group][:8])
        smoke_folder = args.output / (arm + '-smoke')
        smoke_folder.mkdir()
        (smoke_folder / 'pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in smoke))
        small = dict(receipt, rows=len(smoke), groups=dict(Counter(r['group'] for r in smoke)),
                     pairs_sha256=sha256_file(smoke_folder / 'pairs.jsonl'), smoke=True)
        small.pop('data_id')
        small['data_id'] = digest_json(small)
        write_json(smoke_folder / 'manifest.json', small)
    write_json(args.output / 'report.json', dict(completed=True, seed=args.seed,
        groups={k: dict(Counter(r['group'] for r in v)) for k, v in selected.items()},
        overlap_pairs=len({r['id'] for r in selected['filtered']} & {r['id'] for r in selected['control']}),
        caveat='One matched random control and one training seed; not a causal guarantee or a release evaluation.'))
    (args.output / '.building').unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'audit', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261007)
    build(p.parse_args())


if __name__ == '__main__':
    main()
