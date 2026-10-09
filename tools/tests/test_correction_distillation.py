from tools.tests.dependencies import require

require('torch', 'numpy', 'tokenizers', 'regex')

import unittest
from tools.correction.distill_data import admitted_rows


class DistillationTests(unittest.TestCase):
    def setUp(self):
        self.bank = [dict(id="a", split="train", source_text="আমি।", target="আমি।"),
                     dict(id="b", split="train", source_text="এমনন।", target="এমন।")]
        self.rows = [dict(r, ended=True, teacher_answer=r["target"], accepted=True) for r in self.bank]

    def test_exact_agreement_retains_source_evidence(self):
        admitted = admitted_rows(self.bank, self.rows)
        self.assertEqual([r["id"] for r in admitted], ["a", "b"])
        self.assertIn("not human gold", admitted[0]["supervision"])

    def test_incomplete_forged_or_truncated_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            admitted_rows(self.bank, self.rows[:1])
        self.rows[0]["source_text"] = "different"
        with self.assertRaisesRegex(ValueError, "changed original"):
            admitted_rows(self.bank, self.rows)
        self.rows[0]["source_text"] = self.bank[0]["source_text"]
        self.rows[0]["ended"] = False
        with self.assertRaisesRegex(ValueError, "acceptance mismatch"):
            admitted_rows(self.bank, self.rows)

    def test_benchmark_rows_cannot_be_admitted(self):
        self.bank[0]["split"] = "validation"
        self.rows[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training evidence"):
            admitted_rows(self.bank, self.rows)


if __name__ == "__main__":
    unittest.main()
