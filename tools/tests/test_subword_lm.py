import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class SubwordModelTests(unittest.TestCase):
    def test_prefix_projection_ignores_future_padding(self):
        import torch
        from tools.autosuggest.subword_lm import ModelConfig, make_model
        from tools.autosuggest.export_subword_lm import PrefixScorer

        model = make_model(
            ModelConfig(
                vocab_size=32,
                width=16,
                layers=2,
                heads=2,
                ff_width=32,
                sequence_length=8,
            )
        ).eval()
        scorer = PrefixScorer(model)
        tokens = torch.tensor([[1, 7, 9, 0, 0, 0, 0, 0]], dtype=torch.int32)
        with torch.no_grad():
            expected = model(tokens[:, :3].long())[:, -1]
            actual = scorer(tokens, torch.tensor([2], dtype=torch.int32))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            tokens[:, 3:] = torch.tensor([19, 22, 13, 2, 4])
            torch.testing.assert_close(
                scorer(tokens, torch.tensor([2], dtype=torch.int32)),
                expected,
                atol=1e-6,
                rtol=1e-6,
            )

    def test_causality_and_checkpoint_roundtrip(self):
        import torch
        from tools.autosuggest.subword_lm import ModelConfig, make_model

        torch.manual_seed(17)
        config = ModelConfig(
            vocab_size=32, width=16, layers=2, heads=2, ff_width=32, sequence_length=8
        )
        model = make_model(config).eval()
        a = torch.tensor([[1, 7, 9, 3, 8, 2]])
        b = torch.tensor([[1, 7, 9, 12, 14, 15]])
        with torch.no_grad():
            x, y = model(a), model(b)
        torch.testing.assert_close(x[:, :3], y[:, :3], atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(x).all())
        copy = make_model(config).eval()
        copy.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(copy(a), x)
        model.train()
        model(a).square().mean().backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )


@unittest.skipUnless(
    importlib.util.find_spec("tokenizers"), "optional tokenizers dependency"
)
class SubwordTokenizerTests(unittest.TestCase):
    def test_indic_marks_roundtrip_and_grouping(self):
        import unicodedata
        from tokenizers import trainers, pre_tokenizers
        from tools.autosuggest.subword_tokenizer import make_tokenizer

        text = "আমি তোমাকে ভালোবাসি"
        aware = make_tokenizer("unicode-marks-v1")
        legacy = make_tokenizer("gpt2-regex-v0")
        self.assertEqual(len(aware.pre_tokenizer.pre_tokenize_str(text)), 3)
        self.assertGreater(len(legacy.pre_tokenizer.pre_tokenize_str(text)), 3)
        samples = [
            text,
            "তুমি কোথায় যাবে",
            "র‌্যাব ক্ষুদ্র\nকি  হলো?",
            "আজ meeting আছে",
            "🙂 #বাংলা ১২৩",
        ]
        aware.train_from_iterator(
            samples,
            trainer=trainers.BpeTrainer(
                vocab_size=512,
                special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            ),
        )
        for source in samples:
            self.assertEqual(
                aware.decode(aware.encode(source).ids),
                unicodedata.normalize("NFC", source),
            )
        with self.assertRaises(ValueError):
            make_tokenizer("unknown")


if __name__ == "__main__":
    unittest.main()
