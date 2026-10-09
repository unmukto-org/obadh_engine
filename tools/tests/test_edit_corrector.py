from tools.tests.dependencies import require

require('torch', 'numpy', 'tokenizers', 'regex')

import unittest
import torch
from tools.correction.edit_model import align, apply_edits, EditConfig, EditCorrector


class EditCorrectorTests(unittest.TestCase):
    def test_random_supported_alignments_always_reconstruct_the_reference(self):
        import random
        rng = random.Random(781)
        supported = 0
        for _ in range(1000):
            source = [1] + [rng.randrange(4, 20) for _ in range(rng.randrange(1, 16))] + [2]
            target = [1] + [rng.randrange(4, 20) for _ in range(rng.randrange(1, 16))] + [2]
            labels = align(source, target)
            if labels is not None:
                supported += 1
                self.assertEqual(apply_edits(source, *labels), target)
        self.assertGreater(supported, 100)

    def test_alignment_roundtrip_and_capacity_are_explicit(self):
        source = [1, 7, 8, 9, 2]
        for target in ([1, 7, 8, 9, 2], [1, 7, 9, 2], [1, 7, 11, 9, 2], [1, 10, 7, 8, 9, 2], [1, 7, 10, 11, 12, 13, 9, 2], [1, 7, 8, 9, 10, 2]):
            labels = align(source, target)
            self.assertIsNotNone(labels)
            self.assertEqual(apply_edits(source, *labels), list(target))
        self.assertIsNone(align(source, [1, 10, 11, 12, 13, 7, 8, 9, 2]))

    def test_explicit_keep_copies_rare_tokens_exactly(self):
        torch.set_num_threads(1)
        model = EditCorrector(EditConfig(vocab_size=32, width=16, layers=2, heads=2, ff_width=32, sequence_length=16)).eval()
        with torch.no_grad():
            for head in (model.action, model.append):
                head.weight.zero_()
                head.bias.fill_(-10)
                head.bias[0] = 10
            source = torch.tensor([[1, 29, 31, 2, 0], [1, 7, 8, 9, 2]])
            result, _, ended = model.greedy(source, torch.tensor([0, 1]))
            self.assertEqual(result.tolist(), [[29, 31, 2, 2], [7, 8, 9, 2]])
            self.assertTrue(bool(ended.all()))
            batch = dict(source=source, target=torch.tensor([[29, 31, 2, 0, 0], [7, 10, 9, 2, 0]]), mode=torch.tensor([0, 1]))
        loss, supported, unsupported = model.loss(batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertEqual((supported, unsupported), (2, 0))

    def test_edit_detector_cannot_override_weak_replacement_evidence(self):
        model = EditCorrector(EditConfig(vocab_size=32, width=16, layers=2, heads=2, ff_width=32, sequence_length=16)).eval()
        with torch.no_grad():
            model.action.weight.zero_(); model.action.bias.fill_(-10); model.action.bias[2] = 10
            model.append.weight.zero_(); model.append.bias.fill_(-10); model.append.bias[0] = 10
            for projection in model.slots:
                projection[0].weight.zero_()
            source = torch.tensor([[1, 29, 31, 2]])
            result, _, _ = model.greedy(source, torch.tensor([0]), threshold=.5, confidence="joint")
            self.assertEqual(result.tolist(), [[29, 31, 2]])


if __name__ == "__main__":
    unittest.main()
