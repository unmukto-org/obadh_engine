from tools.tests.dependencies import require

require('torch', 'numpy', 'tokenizers', 'regex')

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch
from tools.correction.distillation_sampling import TeacherPairData, KindBalancedPairData, verify_lineage


class TeacherSamplingTests(unittest.TestCase):
    def test_square_root_sampler_balances_tasks_and_resumes_exactly(self):
        data = KindBalancedPairData.__new__(KindBalancedPairData)
        data.arrays = {"train": dict(source=np.arange(102)[:, None], target=np.arange(102)[:, None], mode=np.zeros(102), weight=np.ones((102, 1)), identity=np.asarray([1] + [0] * 101))}
        data.pools = {"train": {1: np.asarray([0]), 0: np.arange(1, 102)}}
        data.error_kinds = [np.arange(1, 101), np.asarray([101])]
        data.kind_probabilities = torch.tensor([10 / 11, 1 / 11], dtype=torch.float64)
        generator = torch.Generator().manual_seed(97)
        first = data.sample("train", 10000, .4, generator)["source"].flatten().numpy()
        self.assertEqual(int((first == 0).sum()), 4000)
        self.assertTrue(450 < int((first == 101).sum()) < 650)
        saved = generator.get_state()
        expected = data.sample("train", 128, .65, generator)
        resumed = data.sample("train", 128, .65, torch.Generator().set_state(saved))
        for name in expected:
            self.assertTrue(torch.equal(expected[name], resumed[name]))
        clean = data.sample("train", 8, 1., generator)["source"]
        self.assertEqual(clean.tolist(), [[0]] * 8)
        data.kind_probabilities = torch.tensor([.5, .5], dtype=torch.float64)
        uniform = data.sample("train", 10000, .4, generator)["source"].flatten().numpy()
        self.assertTrue(2800 < int((uniform == 101).sum()) < 3200)

    def test_rare_error_categories_are_not_drowned_by_larger_categories(self):
        data = TeacherPairData.__new__(TeacherPairData)
        data.arrays = {"train": dict(source=np.arange(102)[:, None], target=np.arange(102)[:, None], mode=np.zeros(102), weight=np.ones((102, 1)), identity=np.asarray([1] + [0] * 101))}
        data.pools = {"train": {1: np.asarray([0]), 0: np.arange(1, 102)}}
        data.error_kinds = [np.arange(1, 101), np.asarray([101])]
        a = data.sample("train", 10000, .4, torch.Generator().manual_seed(41))["source"].flatten().numpy()
        b = data.sample("train", 10000, .4, torch.Generator().manual_seed(41))["source"].flatten().numpy()
        np.testing.assert_array_equal(a, b)
        self.assertEqual(int((a == 0).sum()), 4000)
        self.assertTrue(2800 < int((a == 101).sum()) < 3200)

    def test_cross_derivation_requires_same_corpus_and_disjoint_works_and_texts(self):
        with tempfile.TemporaryDirectory() as directory:
            base, teacher = Path(directory) / "base", Path(directory) / "teacher"
            base.mkdir(); teacher.mkdir()
            receipt = dict(corpus_id="same", tokenizer_sha256="same", vocab_size=20, sequence_length=16)
            def write(root, split, work, text):
                with gzip.open(root / (split + ".jsonl.gz"), "wt") as f:
                    f.write(json.dumps(dict(split=split, work_id=work, clean_sha256=text)) + "\n")
            write(base, "validation", "held", "held-text")
            write(teacher, "train", "training", "training-text")
            a = SimpleNamespace(root=base, receipt=receipt)
            b = SimpleNamespace(root=teacher, receipt=dict(receipt, teacher_validated=True))
            self.assertEqual(len(verify_lineage(a, b)), 1)
            write(teacher, "train", "held", "different-window")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                verify_lineage(a, b)
            write(teacher, "train", "training", "held-text")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                verify_lineage(a, b)
            b.receipt["corpus_id"] = "different"
            with self.assertRaisesRegex(ValueError, "lineage"):
                verify_lineage(a, b)


if __name__ == "__main__":
    unittest.main()
