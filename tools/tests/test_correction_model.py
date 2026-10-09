from tools.tests.dependencies import require

require('torch', 'numpy', 'tokenizers', 'regex')

import unittest
import torch
from tools.correction.model import CorrectionConfig, Corrector
from tools.correction.data import corruptions, edited_target_weights, sentences


class CorrectionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = Corrector(CorrectionConfig(vocab_size=40, width=16, layers=2, heads=2, ff_width=32, sequence_length=16, decoder_layers=2)).eval()

    def test_teacher_forced_and_incremental_decoder_match_and_copy_is_normalized(self):
        source = torch.tensor([[1, 7, 8, 7, 2, 0], [1, 9, 11, 2, 0, 0]])
        mode = torch.tensor([0, 1])
        previous = torch.tensor([[1, 7, 8, 2], [1, 9, 11, 2]])
        with torch.no_grad():
            logits, attention, gate = self.model(source, mode, previous)
            memory, valid, state = self.model.encode(source, mode)
            for index in range(previous.shape[1]):
                a, b, c, state = self.model.decode(memory, valid, previous[:, index:index + 1], state)
                for actual, expected in ((a, logits[:, index:index + 1]), (b, attention[:, index:index + 1]), (c, gate[:, index:index + 1])):
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            mixture = logits.softmax(-1) * gate.sigmoid()
            mixture.scatter_add_(2, source[:, None].expand(-1, previous.shape[1], -1), attention * (-gate).sigmoid())
            torch.testing.assert_close(mixture.sum(-1), torch.ones_like(mixture[..., 0]))
            target = torch.tensor([[7, 8, 7, 2], [9, 11, 2, 2]])
            expected = mixture.gather(-1, target[..., None]).squeeze(-1).log()
            actual = self.model.target_log_prob(logits, attention, gate, source, target)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(float(mixture[..., [0, 1, 3]].sum()), 0)

    def test_padding_does_not_change_valid_encoder_states_or_decoder(self):
        with torch.no_grad():
            short = torch.tensor([[1, 8, 9, 2]])
            padded = torch.tensor([[1, 8, 9, 2, 0, 0, 0]])
            mode = torch.tensor([0])
            a, _, _ = self.model.encode(short, mode)
            b, _, _ = self.model.encode(padded, mode)
            torch.testing.assert_close(a, b[:, :4], atol=1e-6, rtol=1e-6)
            x = self.model(short, mode, torch.tensor([[1, 8]]))[0]
            y = self.model(padded, mode, torch.tensor([[1, 8]]))[0]
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)
            self.assertLessEqual(self.model.greedy(short, mode, maximum=3)[0].shape[1], 3)
            with self.assertRaisesRegex(ValueError, "capacity"):
                self.model.greedy(short, mode, maximum=17)

    def test_corruptions_preserve_protected_text_and_do_not_invent_question_targets(self):
        from tools.autosuggest.compare_proofreaders import protected
        clean = "আমি সকালে meeting করি, ID AB123, সময় ১২:৩০।"
        variants = corruptions(clean, 19)
        self.assertTrue(variants)
        for _, source in variants:
            self.assertEqual(protected(source), protected(clean))
            self.assertNotEqual(source, clean)
        self.assertIn("agreement_candidate", {k for k, _ in variants})
        self.assertNotIn(("punctuation", "তুমি আসবে?"), corruptions("তুমি আসবে।", 19))
        self.assertEqual(list(sentences('ড. করিম এলেন। “যাবে?” হ্যাঁ!')), ['ড. করিম এলেন।', '“যাবে?”', 'হ্যাঁ!'])
        self.assertEqual(edited_target_weights([7, 8, 2], [7, 9, 2]), [1, 4, 1])
        self.assertEqual(edited_target_weights([7, 8, 2], [7, 2]), [1, 4])

    def test_realistic_matra_hasanta_and_conjunct_errors_are_representable(self):
        generated = {source for seed in range(100) for _, source in corruptions("আমি ভালোবাসি।", seed)}
        self.assertIn("আমি ভালোবসি।", generated)
        generated = {source for seed in range(100) for _, source in corruptions("আমি নিশ্চিন্ত।", seed)}
        self.assertIn("আমি নিশচিন্ত।", generated)
        generated = {source for seed in range(100) for _, source in corruptions("তোমাকে স্বাগতম।", seed)}
        self.assertIn("তোমাকে সাগতম।", generated)


if __name__ == "__main__":
    unittest.main()
