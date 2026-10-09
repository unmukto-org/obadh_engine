import json
from pathlib import Path
import tempfile
import unittest

from tools.correction.preference_audit import decision, prompt, resume_rows, verdict


class PreferenceAuditTests(unittest.TestCase):
    def test_order_bias_is_quarantined(self):
        a = '{"verdict":"A"}'
        b = '{"verdict":"B"}'
        self.assertEqual(decision([verdict(a, False), verdict(b, True)]), 'confirmed')
        self.assertEqual(decision([verdict(b, False), verdict(a, True)]), 'contradicted')
        self.assertEqual(decision([verdict(a, False), verdict(a, True)]), 'order_disagreement')
        self.assertEqual(decision(['both', 'both']), 'both_acceptable')
        for bad in (None, '{"verdict":"A","verdict":"B"}', '{"verdict":true}', '{"verdict":"A","extra":1}'):
            self.assertEqual(verdict(bad, False), 'invalid')

    def test_blinded_prompt_preserves_prefix_mode(self):
        row = dict(mode=1, source='x', chosen='y', rejected='z', group='preserve')
        data = json.loads(prompt(row, True)[1]['content'])
        self.assertEqual(data, dict(mode='prefix', original='x', A='z', B='y'))

    def test_resume_recovers_torn_tail_and_rejects_wrong_bank(self):
        judgments = [dict(answer='{"verdict":"A"}', swapped=False), dict(answer='{"verdict":"B"}', swapped=True)]
        row = dict(id='a', judgments=judgments, decision='confirmed')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            complete = json.dumps(row) + '\n'
            path.write_text(complete + '{"id":')
            self.assertEqual(resume_rows(path, [dict(id='a')]), [row])
            self.assertEqual(path.read_text(), complete)
            with self.assertRaises(ValueError):
                resume_rows(path, [dict(id='b')])


if __name__ == '__main__':
    unittest.main()
