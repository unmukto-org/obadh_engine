import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.autosuggest.document_data import DocumentData, prepare
from tools.corpus.provenance import digest_json, sha256_file, write_json


class DocumentDataTests(unittest.TestCase):
    def test_default_training_cli_allows_document_owned_sampling(self):
        from tools.autosuggest import subword_lm

        with (
            patch(
                "sys.argv",
                [
                    "subword_lm",
                    "train",
                    "--data",
                    "data",
                    "--output",
                    "out",
                    "--stop-epoch",
                    "1791388800",
                ],
            ),
            patch.object(subword_lm, "train") as trainer,
        ):
            subword_lm.main()
        self.assertEqual(trainer.call_args.args[0].literary_fraction, 0.0)

    def test_boundaries_exposure_and_integrity(self):
        from tools.tests.dependencies import require
        require("numpy", "torch", "tokenizers")
        import numpy as np
        import torch
        from tokenizers import Tokenizer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            files = []
            originals = {}
            for split in ("train", "validation", "test"):
                rows = []
                for i in range(22 if split == "train" else 3):
                    source = "news" if i == 0 else "wiki" if i == 1 else "epub"
                    text = (
                        f"লেখা {i}।\n— যাবে?\nহ্যাঁ! meeting ১২:৩০ [PAD] [UNK] [EOS] [BOS] "
                        * (i + 1)
                    )
                    row = dict(
                        source=source,
                        authors=[f"author{i}"] if source == "epub" else [],
                        work_id=f"{split}:{i}",
                        window=0,
                        split=split,
                        text=text,
                        sha256=hashlib.sha256(text.encode()).hexdigest(),
                    )
                    rows.append(row)
                    originals[row["work_id"]] = text
                # Invalid training-shaped test content proves it is never parsed.
                if split == "test":
                    rows = [{"not_training_text": "TEST_SENTINEL"}]
                path = corpus / (split + ".jsonl.gz")
                with gzip.open(path, "wt", encoding="utf-8") as out:
                    for row in rows:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
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
            output = root / "prepared"
            args = argparse.Namespace(
                corpus=corpus,
                output=output,
                vocab=512,
                sequence_length=16,
                tokenizer_characters_per_domain=100000,
                domain_weights=[0.5, 0.25, 0.25],
                author_cap=0.05,
            )
            prepare(args)
            tokenizer_sha = sha256_file(output / "tokenizer.json")
            data = DocumentData(
                output, tokenizer_sha256=tokenizer_sha, sequence_length=16
            )
            tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
            tokenizer.encode_special_tokens = True
            meta = json.loads((output / "train.json").read_text())
            groups = np.fromfile(output / "train.groups.u16", dtype="<u2")
            counts = np.fromfile(output / "train.targets.u16", dtype="<u2")
            probabilities = np.diff(np.r_[0.0, data.cdf["train"]])
            exposure = np.bincount(groups, weights=probabilities * counts)
            exposure /= exposure.sum()
            np.testing.assert_allclose(
                exposure, meta["group_target_probabilities"], rtol=1e-10
            )
            self.assertLessEqual(max(exposure[2:] / exposure[2:].sum()), 0.050000001)
            with gzip.open(
                output / "train.index.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                for line in handle:
                    record = json.loads(line)
                    start, count = record["first_block"], record["block_count"]
                    blocks = data.blocks["train"][start : start + count]
                    expected = (
                        [1] + tokenizer.encode(originals[record["work_id"]]).ids + [2]
                    )
                    reconstructed = [int(blocks[0, 0])]
                    for block, n in zip(blocks, counts[start : start + count]):
                        reconstructed.extend(block[1 : int(n) + 1].tolist())
                        self.assertTrue(np.all(block[int(n) + 1 :] == 0))
                    self.assertEqual(reconstructed, expected)
            sample1 = data.sample("train", 8, torch.Generator().manual_seed(9))
            sample2 = data.sample("train", 8, torch.Generator().manual_seed(9))
            np.testing.assert_array_equal(sample1, sample2)
            self.assertEqual(sample1.shape, (8, 17))
            with self.assertRaises(ValueError):
                prepare(args)
            with (output / "train.targets.u16").open("ab") as out:
                out.write(b"\x00\x00")
            with self.assertRaisesRegex(ValueError, "integrity"):
                DocumentData(output, tokenizer_sha256=tokenizer_sha, sequence_length=16)


if __name__ == "__main__":
    unittest.main()
