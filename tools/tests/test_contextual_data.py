import unittest
from tools.tests.dependencies import require
require("regex", "numpy")
from tools.correction.contextual_data import words, replace_word, changed_word_pairs


class ContextualDataTests(unittest.TestCase):
    def test_replaces_whole_word_only_and_keeps_quotes(self):
        text = 'রিমা বলল, “প্রশ্নটা প্রশ্ন নয়।”'
        span = next(w for w in words(text) if w.group() == "প্রশ্ন")
        self.assertEqual(replace_word(text, span, "প্রস্ন"), 'রিমা বলল, “প্রশ্নটা প্রস্ন নয়।”')

    def test_pair_exclusion_is_context_independent(self):
        pairs = changed_word_pairs("আমার প্রয়জন আছে।", "আমার প্রয়োজন আছে।")
        self.assertEqual(pairs, {("প্রয়জন", "প্রয়োজন")})
        self.assertEqual(changed_word_pairs("একটি কথা।", "একটি কথা।"), set())

    def test_word_spans_keep_joiners_and_numbers_intact(self):
        text = "ক্ষ্ম নি\u200cয়ে ID42 ১২৩"
        self.assertEqual([m.group() for m in words(text)], ["ক্ষ্ম", "নি\u200cয়ে", "ID42", "১২৩"])
