import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools.tests.dependencies import require
require('numpy')
from tools.correction.acceptance import FEATURES
from tools.correction.acceptance_union import candidates, select_threshold, load_pair, fit, evaluate
from tools.corpus.provenance import sha256_file


class UnionTests(unittest.TestCase):
    def row(self, answer='যাব।', value=1.):
        return dict(id='one', source='যব।', target='যাব।', mode=0, kind='fixture',
                    raw_answer=answer, answer=answer, fallback_reason=None, capacity_exceeded=False,
                    features=[value] + [0.] * (len(FEATURES) - 1))

    def test_unique_candidates_support_flags_and_reference_independence(self):
        pair = [self.row(), self.row()]
        proposals = candidates(pair)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]['features'][-4:], [1, 1, 1, 1])
        mutated = deepcopy(pair)
        for row in mutated:
            row['target'] = 'different label'
        self.assertEqual(candidates(mutated), proposals)
        pair[1]['raw_answer'] = 'যাই।'
        self.assertEqual(len(candidates(pair)), 2)
        pair[1]['fallback_reason'] = 'protected_span_changed'
        self.assertEqual(candidates(pair)[0]['features'][-4:], [1, 0, 1, 0])
        pair[0]['raw_answer'] = pair[0]['source']
        self.assertEqual(candidates(pair), [])

    def test_threshold_counts_unproposed_errors(self):
        rows = [dict(source='a', target='b'), dict(source='c', target='d'), dict(source='e', target='e')]
        chosen = [dict(answer='b', probability=.9), dict(answer='c', probability=None), dict(answer='f', probability=.2)]
        result = select_threshold(chosen, rows)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['missed'], 1)
        self.assertEqual(result['incorrect'], 0)
        self.assertAlmostEqual(result['f05'], 1.25 / 1.5)

    def test_fit_then_apply_frozen_artifact_and_reject_overlap_or_misalignment(self):
        require('torch', 'tokenizers', 'regex')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def bank(name, model, count=64):
                path = root / (name + model)
                path.mkdir()
                rows = []
                for i in range(count):
                    source = name + str(i)
                    good = i % 2 == 0
                    row = self.row(source + '!', 3. if good else -3.)
                    row.update(id=name + str(i), source=source, target=source + '!' if good else source)
                    rows.append(row)
                (path / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                (path / 'report.json').write_text(json.dumps(dict(completed=True, features=FEATURES,
                    checkpoint_sha256=model, generator_contract=dict(bank_sha256=name),
                    rows_sha256=sha256_file(path / 'rows.jsonl'))))
                return path
            train = [bank('train', m) for m in ('left', 'right')]
            calibration = [bank('calibration', m) for m in ('left', 'right')]
            held = [bank('held', m) for m in ('left', 'right')]
            model = root / 'model'
            fit(argparse.Namespace(train_left=train[0], train_right=train[1], calibration_left=calibration[0],
                                   calibration_right=calibration[1], output=model))
            before = (model / 'model.json').read_bytes()
            args = argparse.Namespace(model=model, left=held[0], right=held[1], output=root / 'evaluation')
            evaluate(args)
            self.assertEqual(before, (model / 'model.json').read_bytes())
            rows = [json.loads(s) for s in (args.output / 'rows.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['answer'] == r['target'] for r in rows))
            args.left, args.right = train
            with self.assertRaisesRegex(ValueError, 'overlap'):
                evaluate(args)
            with self.assertRaisesRegex(ValueError, 'unaligned'):
                load_pair(train[0], calibration[1])
            manifest = json.loads((held[1] / 'report.json').read_text())
            manifest['checkpoint_sha256'] = 'changed-generator'
            (held[1] / 'report.json').write_text(json.dumps(manifest))
            args.left, args.right = held
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                evaluate(args)
