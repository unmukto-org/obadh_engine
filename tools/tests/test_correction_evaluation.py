from tools.tests.dependencies import require

require('torch', 'numpy', 'tokenizers', 'regex')

import unittest
from tools.correction.evaluate import decode_strict, edits, measure, extra_cases


class CorrectionEvaluationTests(unittest.TestCase):
    def test_strict_decoder_does_not_hide_broken_utf8_or_missing_eos(self):
        pieces = {4: b"\xe0", 5: b"\xa6\x86", 6: b"!"}
        self.assertEqual(decode_strict([4, 5, 6, 2], pieces), ("আ!", None))
        self.assertEqual(decode_strict([4, 2], pieces)[1], "invalid_utf8")
        self.assertEqual(decode_strict([4, 5], pieces)[1], "missing_eos")
        self.assertEqual(decode_strict([1, 2], pieces)[1], "empty_or_control_token")

    def test_fallback_cannot_masquerade_as_raw_identity_success(self):
        row = dict(kind="identity", source="আমি।", target="আমি।", raw_answer=None,
                   answer="আমি।", capacity_exceeded=False, fallback_reason="missing_eos")
        result = measure([row])["all"]
        self.assertEqual(result["raw_identity_preserved"], 0)
        self.assertEqual(result["guarded_identity_preserved"], 1)
        self.assertEqual(result["fallbacks"], 1)
        self.assertEqual(result["raw_invalid"], 1)

    def test_grapheme_edit_positions_preserve_combining_marks(self):
        self.assertEqual(edits("তুমি?", "তুমি।"), {(2, 3, "।")})
        self.assertEqual(edits("আমি", "আমি"), set())

    def test_diagnostic_bank_requires_unique_explicit_case_identities(self):
        import json
        import tempfile
        from pathlib import Path
        row = dict(id="diagnostic:one", kind="diagnostic/identity", source="আমি।", target="আমি।", mode=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(extra_cases(path), [row])
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                extra_cases(path)
            row["mode"] = True
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "record"):
                extra_cases(path)


if __name__ == "__main__":
    unittest.main()
