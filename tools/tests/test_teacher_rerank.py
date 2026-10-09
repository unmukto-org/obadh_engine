import json
import unittest
from tools.correction.teacher_rerank import choices, prompt, selected, outcomes


class TeacherRerankTests(unittest.TestCase):
    def test_reference_does_not_affect_choices_or_prompts(self):
        row = dict(id='x', source='সে যায়', target='a-secret-reference', mode=1,
            candidates=[dict(answer='সে যায়।', fallback_reason=None), dict(answer='সে যায়', fallback_reason=None)])
        changed = dict(row, target='another-reference', kind='different-label')
        self.assertEqual(choices(row), choices(changed))
        self.assertEqual(prompt(row, choices(row)), prompt(changed, choices(changed)))
        self.assertNotIn('reference', json.dumps(prompt(row, choices(row))))
        self.assertEqual(json.loads(prompt(row, choices(row))[1]['content'])['mode'], 'prefix')

    def test_invalid_or_order_disagreement_preserves_input(self):
        for raw in ('{"candidate":true}', '{"candidate":-1}', '{"candidate":2}', '{"candidate":0,"candidate":1}', 'no'):
            self.assertIsNone(selected(raw, ['x', 'y']))
        self.assertEqual(selected('{"candidate":0}', ['x', 'y']), 'x')
        row = dict(source='x')
        self.assertEqual(outcomes(row, [dict(selected='y'), dict(selected='z')]), dict(single='y', agreement='x'))
        self.assertEqual(outcomes(row, [dict(selected=None), dict(selected='y')]), dict(single='x', agreement='x'))

    def test_protected_edits_and_fallback_candidates_are_excluded(self):
        row = dict(id='x', source='৫০ টাকা', candidates=[dict(answer='৫১ টাকা', fallback_reason=None),
            dict(answer='ভুল', fallback_reason='invalid_utf8')])
        self.assertEqual(choices(row), ['৫০ টাকা'])
