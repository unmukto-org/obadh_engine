from tools.tests.dependencies import require

require('torch', 'numpy', 'tokenizers', 'regex')

import copy
import json
import unittest
from tools.correction.roundtrip_data import admitted, conservative_rejection


class GrammarAdmissionTests(unittest.TestCase):
    def setUp(self):
        target = "তাঁরা সবাই নেপালি।"
        source = "তাঁরা সবাই নেপালিকে।"
        self.bank = [dict(id="case", split="train", clean_sha256="clean", work_id="training-work", kind="identity", mode=0, source_text=target, target=target, requested_error="case_postposition", parent_id="parent")]
        self.rows = [dict(self.bank[0], proposed_source=source, accepted=True, rejection=None,
                          generation=dict(answer=json.dumps(dict(source=source)), ended=True),
                          verification=dict(answer=target, ended=True),
                          critique=dict(answer=json.dumps(dict(source_has_clear_error=True, target_is_acceptable=True, intent_preserved=True)), ended=True))]

    def test_validated_errors_include_reference_identity_controls(self):
        rows = admitted(self.bank, self.rows)
        self.assertEqual({r["kind"] for r in rows}, {"teacher_grammar/case_postposition", "identity"})
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        self.assertTrue(all(r["split"] == "train" and r["work_id"] == "training-work" for r in rows))

    def test_unapproved_or_mutated_evidence_is_not_laundered_into_labels(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["critique"]["answer"] = '{}'
        with self.assertRaisesRegex(ValueError, "admission"):
            admitted(self.bank, rows)
        rows[0]["accepted"] = False
        self.assertEqual(admitted(self.bank, rows), [])
        rows = copy.deepcopy(self.rows)
        rows[0]["target"] = "changed reference"
        with self.assertRaisesRegex(ValueError, "original evidence"):
            admitted(self.bank, rows)

    def test_teacher_agreement_cannot_certify_punctuation_intent_or_honorific_style(self):
        for source, target, reason in (
            ("তুমি আসবে?", "তুমি আসবে।", "ambiguous_punctuation_intent"),
            ("সে বলল।", "সে বললেন।", "ambiguous_honorific_choice"),
            ("রহিম কাজ করে।", "রহিম কাজ করেন।", "ambiguous_honorific_choice"),
            ("তিনি যাবেন।", "তিনি যাবে।", "ambiguous_honorific_choice"),
        ):
            self.assertEqual(conservative_rejection(source, target), reason)
        self.assertIsNone(conservative_rejection("আমি কাজ করো।", "আমি কাজ করি।"))
        self.assertIsNone(conservative_rejection("সে বলল (আসব।", "সে বলল (আসব)।"))
        target, source = "সে বললেন।", "সে বলল।"
        bank = copy.deepcopy(self.bank)
        bank[0].update(target=target, source_text=target, requested_error="agreement")
        rows = copy.deepcopy(self.rows)
        rows[0].update(bank[0], proposed_source=source)
        rows[0]["generation"]["answer"] = json.dumps(dict(source=source))
        rows[0]["verification"]["answer"] = target
        from collections import Counter
        exclusions = Counter()
        self.assertEqual(admitted(bank, rows, exclusions), [])
        self.assertEqual(exclusions["ambiguous_honorific_choice"], 1)


if __name__ == "__main__":
    unittest.main()
