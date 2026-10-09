import unittest

from tools.correction.byte_data import decode, encode, fits, lexical_reason, word_partition, target_weights


class ByteContracts(unittest.TestCase):
    def test_edit_loss_weights_complete_utf8_characters(self):
        self.assertEqual(target_weights("ক", "খ", 4), [4., 4., 4., 1.])
        self.assertEqual(target_weights("কখ", "ক", 4), [1., 1., 1., 4.])
        self.assertEqual(target_weights("কখ", "খ", 4), [4., 4., 4., 1.])
        self.assertEqual(target_weights("আমি…", "আমি…", 4), [1.] * len(encode("আমি…")))
        with self.assertRaises(ValueError):
            target_weights("ক", "খ", 0)

    def test_lossless_bangla_and_mixed_script(self):
        for text in ("ধন্যবাদ।", "আমি আজ…", "রিমা বলল, ‘কাল meeting ১২টায়।’", "a\tb"):
            self.assertEqual(decode([0] + encode(text) + [0, 0]), (text, None))

    def test_invalid_output_never_silently_replaces_bytes(self):
        for ids, reason in (([0, 230, 1], "invalid_utf8"), ([0, 2, 1], "empty_or_control_token"),
                            ([0, 100], "missing_eos"), ([0, 1], "empty_or_control_token"),
                            ([0, 259, 1], "empty_or_control_token"), ([0, 3, 1], "invalid_text")):
            self.assertEqual(decode(ids), (None, reason))

    def test_capacity_includes_task_prefix_and_eos(self):
        row = dict(source="আমি", target="আমি", mode=0)
        self.assertTrue(fits(row, len(encode("sentence: আমি"))))
        self.assertFalse(fits(row, len(encode("sentence: আমি")) - 1))

    def test_spelling_filters_valid_alternatives_and_destructive_targets(self):
        self.assertEqual(lexical_reason("যায়", "যান", "Typo Deletion", {"যায়", "যান"}), "valid_word_collision")
        self.assertEqual(lexical_reason("বইখাতা", "বই", "Run-on Error", {"বই"}), "unsafe_error_type")
        self.assertEqual(lexical_reason("বই খাতা", "বইখাতা", "Split-word Error (Left)", {"বইখাতা"}), "unsafe_error_type")
        self.assertIsNone(lexical_reason("ধন্নবাদ", "ধন্যবাদ", "Cognitive Error", {"ধন্যবাদ"}))

    def test_word_partition_does_not_depend_on_error_variant(self):
        word = "নিশ্চিন্ত"
        self.assertEqual(word_partition(word), word_partition(word))
        self.assertIn(word_partition(word), {"train", "validation", "reserved"})


class ByteBatchContracts(unittest.TestCase):
    def test_padding_masks_targets_and_preserves_prefix_mode(self):
        from tools.tests.dependencies import require
        require("torch", "numpy")
        from tools.correction.byte_train import collate
        batch = collate([dict(source="আমি", target="আমি", mode=1), dict(source="ধন্নবাদ", target="ধন্যবাদ", mode=2)], "cpu")
        self.assertTrue(batch["labels"].eq(-100).any())
        self.assertTrue(batch["attention_mask"].eq(batch["input_ids"].ne(0)).all())
        self.assertEqual(decode(batch["input_ids"][0].tolist())[0], "prefix: আমি")


if __name__ == "__main__":
    unittest.main()
