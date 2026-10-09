import unittest
from tools.correction.select_initialization import metrics


class InitializationSelectionTests(unittest.TestCase):
    def test_missed_and_wrong_corrections_and_clean_changes_all_count(self):
        rows = [dict(source=s,target=t,answer=a) for s,t,a in [
            ('a','b','b'), ('c','d','c'), ('e','f','g'), ('h','h','i'), ('j','j','j')]]
        report = metrics(rows)
        self.assertEqual(report['correct_repairs'], 1)
        self.assertEqual(report['incorrect_changes'], 2)
        self.assertEqual(report['missed_errors'], 2)
        self.assertEqual(report['clean_preserved'], 1)
        self.assertAlmostEqual(report['f05'], 1.25 / 3.75)
