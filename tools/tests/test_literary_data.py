import importlib.util
import unittest
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from tools.autosuggest.literary_data import capped_probabilities, components


class AuthorBalanceTests(unittest.TestCase):
    def test_large_author_cannot_dominate(self):
        weights = [1000000] + list(range(1, 40))
        p = capped_probabilities(weights, 0.05)
        self.assertAlmostEqual(sum(p), 1)
        self.assertLessEqual(max(p), 0.05)
        self.assertAlmostEqual(p[0], 0.05)

    def test_cap_rejects_insufficient_diversity(self):
        with self.assertRaisesRegex(ValueError, "not enough"):
            capped_probabilities([1] * 19, 0.05)

    def test_translator_links_multiple_bylines(self):
        self.assertEqual(
            components([["A", "B"], ["C", "B"], ["D"]]), [["A", "B", "C"], ["D"]]
        )

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("tokenizers"),
        "optional tokenizer dependencies",
    )
    def test_preparation_preserves_boundaries_and_rejects_tampering(self):
        from tokenizers import trainers, pre_tokenizers
        from tools.autosuggest.subword_tokenizer import make_tokenizer
        from tools.autosuggest.literary_data import prepare, LiteraryData
        from tools.corpus.provenance import digest_json, sha256_file, write_json
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            corpus.mkdir()
            files = []
            tokenizer = make_tokenizer("unicode-marks-v1")
            tokenizer.train_from_iterator(
                ["— যাবে?\nহ্যাঁ!"],
                trainer=trainers.BpeTrainer(
                    vocab_size=280,
                    special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                ),
            )
            tokenizer.save(str(root / "tokenizer.json"))
            for split, count in [("train", 20), ("validation", 1), ("test", 0)]:
                path = corpus / (split + ".jsonl.gz")
                with gzip.open(path, "wt") as out:
                    for i in range(count):
                        text = f"— যাবে {i}?\nহ্যাঁ!" + " যাবে" * i
                        out.write(
                            json.dumps(
                                dict(
                                    split=split,
                                    authors=[str(i)],
                                    work_id=str(i),
                                    text=text,
                                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                                )
                            )
                            + "\n"
                        )
                files.append(
                    dict(
                        path=path.name,
                        bytes=path.stat().st_size,
                        sha256=sha256_file(path),
                    )
                )
            manifest = dict(kind="obadh-contextual-text", files=files)
            manifest["dataset_id"] = digest_json(manifest)
            write_json(corpus / "manifest.json", manifest)
            output = root / "blocks"
            prepare(
                argparse.Namespace(
                    corpus=corpus,
                    tokenizer=root / "tokenizer.json",
                    output=output,
                    sequence_length=128,
                    author_cap=0.05,
                )
            )
            data = LiteraryData(
                output,
                tokenizer_sha256=sha256_file(root / "tokenizer.json"),
                sequence_length=128,
            )
            for block in data.blocks["train"]:
                nonpad = block[block != 0]
                self.assertEqual(nonpad[0], 1)
                self.assertEqual(nonpad[-1], 2)
                self.assertEqual(np.count_nonzero(nonpad == 1), 1)
                self.assertIn("\nহ্যাঁ!", tokenizer.decode(nonpad.tolist()))
            # Exercise actual loader probabilities, including unequal tails.
            lengths = (data.blocks["train"][:, 1:] != 0).sum(1)
            probabilities = np.diff(np.concatenate(([0.0], data.cdf["train"])))
            metadata = json.loads((output / "train.json").read_text(encoding="utf-8"))
            exposure = np.bincount(
                metadata["group_ids"], weights=probabilities * lengths
            )
            np.testing.assert_allclose(exposure / exposure.sum(), np.full(20, 0.05))
            with (output / "train.u16").open("ab") as f:
                f.write(b"bad")
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                LiteraryData(
                    output,
                    tokenizer_sha256=sha256_file(root / "tokenizer.json"),
                    sequence_length=128,
                )


if __name__ == "__main__":
    unittest.main()
