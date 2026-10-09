"""Data-role and tokenizer-boundary tests for offline LLM supervision."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from tools.tests import test_corpus
from tools.autosuggest.teacher_candidates import candidate_encoding, prepare


class CharacterTokenizer:
    bos_token_id = 7
    eos_token_id = 8

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(10, 10 + len(text))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class TeacherCandidatesTests(unittest.TestCase):
    def test_causal_boundaries_and_bounded_context(self):
        tokenizer = CharacterTokenizer()
        ids, start = candidate_encoding(tokenizer, "abc", "de", 96)
        self.assertEqual(start, 4)  # includes the leading space, excludes abc and BOS
        self.assertEqual(len(ids) - start, 3)
        ids, start = candidate_encoding(tokenizer, "abcdefghij", "de", 5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(ids) - start, 3)
        ids, start = candidate_encoding(tokenizer, "", "de", 5)
        self.assertEqual(ids[0], 7)
        self.assertEqual(start, 1)
        with self.assertRaises(ValueError):
            candidate_encoding(tokenizer, "", "abcdef", 5)
        with self.assertRaises(ValueError):
            candidate_encoding(tokenizer, "abc", "", 5)

    def test_roles_integrity_and_no_gold_insertion_on_validation(self):
        fixture = test_corpus.CorpusTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare()
        model = fixture.build_model()
        output = fixture.root / "bank.jsonl"
        args = argparse.Namespace(
            corpus=fixture.dataset / "test",
            model=model,
            output=output,
            sentences=50,
            pool=1,
            context=4,
        )
        with self.assertRaises(ValueError):
            prepare(args)
        args.corpus = fixture.dataset / "validation"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(args)
        manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
        self.assertFalse(manifest["policy"]["gold_inserted"])
        rows = [json.loads(x) for x in output.read_text().splitlines()]
        self.assertTrue(rows)
        self.assertTrue(any(r["target_id"] not in r["candidate_ids"] for r in rows))
        with self.assertRaises(ValueError):
            prepare(args)


if __name__ == "__main__":
    unittest.main()
