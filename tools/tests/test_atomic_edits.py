from tools.tests.dependencies import require

require('regex')

import unittest
from tools.correction.edit_policy import admit_atomic, token_groups


class AtomicEditTests(unittest.TestCase):
    def test_rejected_replacement_cannot_leave_confident_deletions(self):
        actions, counts = admit_atomic([0, 2, 1, 1, 0], [0] * 5, [1, .2, .99, .99, 1], [1] * 5, [-1, 1, 1, 1, -2], .8)
        self.assertEqual(actions, [0] * 5)
        self.assertEqual(counts, [0] * 5)

    def test_word_group_covers_unchanged_middle_and_preserves_independent_edits(self):
        actions, counts = admit_atomic([2, 0, 1, 0, 2], [0] * 5, [.2, 1, .99, 1, .95], [1] * 5, [0, 0, 0, 1, 2], .8)
        self.assertEqual(actions, [0, 0, 0, 0, 2])
        self.assertEqual(counts, [0] * 5)

    def test_insert_delete_combination_is_atomic_across_adjacent_words(self):
        actions, counts = admit_atomic([1, 0], [0, 2], [.99, 1], [1, .1], [0, 1], .8)
        self.assertEqual((actions, counts), ([0, 0], [0, 0]))

    def test_unicode_offsets_group_all_pieces_of_a_word(self):
        text = "আমি ভালোবাসি।"
        offsets = [(0, 1), (1, 3), (3, 5), (5, 8), (8, 12), (12, 13)]
        groups = token_groups(text, offsets)
        self.assertEqual(groups[:2], [0, 0])
        self.assertEqual(groups[2:5], [1, 1, 1])
        self.assertNotEqual(groups[-1], groups[-2])


if __name__ == "__main__":
    unittest.main()
