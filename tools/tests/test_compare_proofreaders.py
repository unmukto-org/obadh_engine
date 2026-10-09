import unittest

from tools.autosuggest.compare_proofreaders import parse_answer, protected, summarize


class ProofreaderEvaluationTests(unittest.TestCase):
    def test_reasoning_is_not_scored_as_a_correction(self):
        self.assertIsNone(parse_answer("unfinished thought", "qwen", True))
        self.assertIsNone(parse_answer("unfinished thought", "gemma", True))
        self.assertIsNone(parse_answer("<|channel>thought\nunfinished", "gemma", False))
        for family, raw in (
            ("qwen", "thought</think>আমি যাব।<|im_end|>"),
            ("gemma", "thought<channel|>আমি যাব।<turn|><pad>"),
            ("gemma", "<|channel>thought\nthought<channel|>আমি যাব।<turn|>"),
        ):
            self.assertEqual(parse_answer(raw, family, True), "আমি যাব।")

    def test_mixed_script_and_numeric_preservation(self):
        self.assertEqual(
            protected("কাল Wi-Fi meeting ৪:৩০-এ AB123"),
            ["Wi-Fi", "meeting", "৪:৩০", "AB123"],
        )
        row = dict(
            kind="clean",
            source="কাল meeting আছে।",
            target="কাল meeting আছে।",
            answer="কাল মিটিং আছে।",
        )
        result = summarize([row])["clean"]
        self.assertEqual(result["changed"], 1)
        self.assertEqual(result["protected_changed"], 1)
        self.assertEqual(result["exact_reference"], 0)

    def test_invalid_outputs_count_as_failed_preservation(self):
        row = dict(kind="clean", source="১২টা", target="১২টা", answer=None)
        result = summarize([row])["clean"]
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["protected_changed"], 1)


if __name__ == "__main__":
    unittest.main()
