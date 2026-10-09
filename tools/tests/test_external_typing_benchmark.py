import tempfile
import unittest
from pathlib import Path
from tools.autosuggest.external_typing_benchmark import words, verify_selection
from tools.corpus.provenance import sha256_file


class ExternalTypingBenchmarkTests(unittest.TestCase):
    def test_noisy_mixed_script_context_is_preserved(self):
        self.assertEqual(
            words("আজ meeting আছে🙂 র\u200c্যাব, ১২৩!"),
            ["আজ", "meeting", "আছে", "র\u200c্যাব", "১২৩"],
        )
        self.assertEqual(words("কোথায় কোথায়"), ["কোথায়", "কোথায়"])

    def test_selection_rejects_changed_artifacts_and_test_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            path.write_bytes(b"original")
            selection = {
                "selected_on": "internal-validation-only",
                "word_checkpoint_sha256": sha256_file(path),
            }
            verify_selection(selection, {"word_checkpoint": path})
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "word_checkpoint"):
                verify_selection(selection, {"word_checkpoint": path})
            selection["selected_on"] = "test"
            with self.assertRaisesRegex(ValueError, "before external testing"):
                verify_selection(selection, {})


if __name__ == "__main__":
    unittest.main()
