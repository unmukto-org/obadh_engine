import unittest
from tools.tests.dependencies import require
require('numpy', 'regex')
from tools.correction.acceptance_lexical import lexical_features
from tools.correction.acceptance import LEXICAL_FEATURES


class LexicalAcceptanceTests(unittest.TestCase):
    def test_nonword_repair_and_real_word_change_are_distinct(self):
        vocabulary = {'স্বপ্ন':20,'যাব':10,'যাই':30}
        f = lexical_features('আমি সপ্ন দেখি।','আমি স্বপ্ন দেখি।',vocabulary)
        self.assertEqual(len(f),len(LEXICAL_FEATURES))
        self.assertEqual(f[0:2],[1.,0.])
        self.assertEqual(f[-1],1)
        self.assertGreater(f[-2],0)
        self.assertEqual(lexical_features('আমি যাব।','আমি যাই।',vocabulary)[-1],0)

    def test_punctuation_only_edit_does_not_invent_lexical_evidence(self):
        self.assertEqual(lexical_features('তুমি যাবে','তুমি যাবে।',{}),[0.]*8)
