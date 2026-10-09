import unittest
import argparse
import json
from pathlib import Path
import tempfile
from tools.tests.dependencies import require
require("numpy")
import numpy as np
from tools.correction.acceptance import features, fit_logistic, probabilities, FEATURES, fit
from tools.correction.byte_data import encode
from tools.corpus.provenance import sha256_file


class AcceptanceTests(unittest.TestCase):
    def test_policy_publication_preserves_proposal_and_rejects_data_overlap(self):
        require('torch', 'tokenizers', 'regex')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def publish(name, letter, count):
                path = root / name
                path.mkdir()
                rows = []
                for i in range(1, count + 1):
                    correct = i % 2 == 1
                    source = letter * i
                    candidate = source + '।'
                    rows.append(dict(id=name + str(i), kind='fixture', source=source,
                                     target=candidate if correct else source, mode=0, raw_answer=candidate,
                                     answer=candidate, fallback_reason=None, capacity_exceeded=False,
                                     features=[3. if correct else -3.] + [0.] * (len(FEATURES) - 1),
                                     reference_correct=correct))
                (path / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                report = dict(completed=True, features=FEATURES, checkpoint_sha256='same-generator',
                              generator_contract=dict(bank_sha256=name, checkpoint_sha256='same-generator'),
                              rows_sha256=sha256_file(path / 'rows.jsonl'))
                (path / 'report.json').write_text(json.dumps(report))
                return path
            train = publish('train', 'ক', 64)
            calibration = publish('calibration', 'গ', 64)
            evaluation = publish('evaluation', 'চ', 2)
            args = argparse.Namespace(train=train, calibration=calibration, evaluation=evaluation, output=root / 'policy')
            fit(args)
            rows = [json.loads(s) for s in (args.output / 'evaluation/rows.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['answer'], rows[0]['target'])
            self.assertEqual(rows[1]['answer'], rows[1]['source'])
            self.assertNotEqual(rows[1]['proposal_answer'], rows[1]['answer'])
            self.assertEqual(rows[1]['fallback_reason'], 'acceptance_rejected')
            self.assertFalse((args.output / '.building').exists())
            calibration_rows = [json.loads(s) for s in (calibration / 'rows.jsonl').read_text().splitlines()]
            calibration_rows[0]['features'] = None
            calibration_rows[0]['raw_answer'] = calibration_rows[0]['source']
            calibration_rows[0]['reference_correct'] = False
            (calibration / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in calibration_rows))
            receipt = json.loads((calibration / 'report.json').read_text())
            receipt['rows_sha256'] = sha256_file(calibration / 'rows.jsonl')
            (calibration / 'report.json').write_text(json.dumps(receipt))
            args.calibration_objective = 'sentence-f05'
            args.output = root / 'sentence-policy'
            fit(args)
            model = json.loads((args.output / 'model.json').read_text())
            self.assertEqual(model['calibration']['missed'], 1)
            self.assertEqual(model['calibration']['correct'], 31)
            self.assertLess(model['calibration']['f05'], 1.)
            args.evaluation = train
            args.output = root / 'overlapping-policy'
            with self.assertRaisesRegex(ValueError, 'overlap'):
                fit(args)

    def test_features_depend_only_on_source_proposal_and_likelihoods(self):
        source, proposal = "আমি যাব।", "আমি যাই।"
        f = features(source, proposal, 0, [-.1] * len(encode(proposal)), [-.2] * len(encode(source)))
        self.assertEqual(len(f), len(FEATURES))
        self.assertAlmostEqual(f[0], .1)
        self.assertTrue(np.isfinite(f).all())
        self.assertGreater(f[6], 0)
        with self.assertRaises(ValueError):
            features(source, proposal, 0, [-.1], [-.2])

    def test_logistic_fit_is_finite_with_constant_features(self):
        x = np.array([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]])
        y = np.array([0., 0., 1., 1.])
        model = fit_logistic(x, y)
        p = probabilities(model, x)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p[:2] < .5).all())
        self.assertTrue((p[2:] > .5).all())
        self.assertEqual(model['scale'][1], 1.)
