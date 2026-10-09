import importlib.util
import unittest

from tools.autosuggest.compare_subword_models import completion_rows


@unittest.skipUnless(importlib.util.find_spec("regex"), "optional grapheme dependency")
class CompletionBankTests(unittest.TestCase):
    def test_graphemes_preserve_marks_and_exclude_fully_typed_targets(self):
        rows = [dict(id="a", target="কোথায়"), dict(id="b", target="মা"), dict(id="c", target="!")]
        eligible, excluded = completion_rows(rows, 1)
        self.assertEqual([r["id"] for r in eligible], ["a"])
        self.assertEqual(eligible[0]["typed_prefix"], "কো")
        self.assertEqual(excluded, dict(fully_typed_at_requested_prefix=1, invalid_lexical_target=1))
        self.assertNotIn("typed_prefix", rows[0])
        self.assertEqual(completion_rows(rows, 0), (rows, dict(fully_typed_at_requested_prefix=0, invalid_lexical_target=0)))
        with self.assertRaisesRegex(ValueError, "prefix"):
            completion_rows(rows, 4)


if __name__ == "__main__":
    unittest.main()
