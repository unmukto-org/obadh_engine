import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from tools.corpus.provenance import digest_json, sha256_file
from tools.correction.audited_preferences import arms, retain, build
from tools.correction.select_initialization import admissible


class AuditedPreferenceTests(unittest.TestCase):
    def test_preserve_optional_style_but_not_ambiguous_repairs(self):
        clean = dict(source='দেয়া', chosen='দেয়া', group='preserve')
        repair = dict(source='কির', chosen='কী', group='repair')
        self.assertTrue(retain(clean, dict(decision='both_acceptable')))
        self.assertFalse(retain(repair, dict(decision='both_acceptable')))
        for pair in (clean, repair):
            self.assertTrue(retain(pair, dict(decision='confirmed')))
            for label in ('contradicted', 'order_disagreement', 'invalid', 'uncertain', 'neither_acceptable'):
                self.assertFalse(retain(pair, dict(decision=label)))

    def test_matched_control_is_deterministic_and_never_relabels(self):
        pairs = [dict(id=str(i), source='x', chosen='x' if i % 2 else 'y',
                      group='preserve' if i % 2 else 'repair') for i in range(20)]
        audited = [dict(id=str(i), group=pairs[i]['group'], decision='confirmed' if i < 12 else 'contradicted') for i in range(20)]
        result = arms(pairs, audited, 7)
        self.assertEqual(result, arms(pairs, audited, 7))
        for rows in result.values():
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(r['group'] == 'repair' for r in rows), 6)
            self.assertTrue(all(row in pairs for row in rows))
        with self.assertRaises(ValueError):
            arms(pairs, list(reversed(audited)), 7)

    def test_precision_gain_cannot_hide_repair_regression(self):
        scores = dict(baseline=dict(correct_repairs=491, clean_preserved=744, incorrect_changes=84),
            cautious=dict(correct_repairs=477, clean_preserved=759, incorrect_changes=47),
            harmful=dict(correct_repairs=500, clean_preserved=744, incorrect_changes=90),
            improved=dict(correct_repairs=500, clean_preserved=745, incorrect_changes=80))
        self.assertEqual(admissible(scores, 'baseline'), ['baseline', 'improved'])

    def test_publication_verifies_audit_and_preserves_training_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, audit = root / 'parent', root / 'audit'
            data.mkdir()
            audit.mkdir()
            pairs = [dict(id=str(i), source='x', chosen='x' if i % 2 else 'y',
                          rejected='z', mode=0, group='preserve' if i % 2 else 'repair') for i in range(200)]
            (data / 'pairs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pairs))
            manifest = dict(rows=200, byte_limit=768, excluded_banks={}, pairs_sha256=sha256_file(data / 'pairs.jsonl'))
            manifest['data_id'] = digest_json(manifest)
            (data / 'manifest.json').write_text(json.dumps(manifest))
            votes = [dict(answer='{"verdict":"A"}', swapped=False), dict(answer='{"verdict":"B"}', swapped=True)]
            (audit / 'rows.jsonl').write_text(''.join(json.dumps(dict(id=r['id'], group=r['group'],
                decision='confirmed', judgments=votes)) + '\n' for r in pairs))
            report = dict(completed=True, rows=200, total=200, rows_sha256=sha256_file(audit / 'rows.jsonl'),
                          contract=dict(data_id=manifest['data_id'], pairs_sha256=manifest['pairs_sha256']))
            (audit / 'report.json').write_text(json.dumps(report))
            build(SimpleNamespace(data=data, audit=audit, output=root / 'out', seed=7))
            for arm in ('control', 'filtered', 'control-smoke', 'filtered-smoke'):
                folder = root / 'out' / arm
                receipt = json.loads((folder / 'manifest.json').read_text())
                self.assertEqual(receipt['data_id'], digest_json({k: v for k, v in receipt.items() if k != 'data_id'}))
                self.assertEqual(receipt['pairs_sha256'], sha256_file(folder / 'pairs.jsonl'))
                rows = list(map(json.loads, (folder / 'pairs.jsonl').read_text().splitlines()))
                self.assertTrue(all(r in pairs for r in rows))
            report['contract']['data_id'] = 'different-training-data'
            (audit / 'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                build(SimpleNamespace(data=data, audit=audit, output=root / 'wrong', seed=7))
            self.assertFalse((root / 'wrong').exists())


if __name__ == '__main__':
    unittest.main()
