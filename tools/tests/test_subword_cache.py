import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class SubwordCacheTests(unittest.TestCase):
    def test_prefill_and_incremental_logits_match_full_context(self):
        import torch
        from tools.autosuggest.subword_lm import make_model, ModelConfig
        from tools.autosuggest.subword_cache import (
            PrefillScorer,
            CachedStepScorer,
            cache_shape,
        )

        torch.manual_seed(19)
        config = ModelConfig(
            vocab_size=32, width=16, layers=2, heads=2, ff_width=32, sequence_length=8
        )
        model = make_model(config).eval()
        prefill = PrefillScorer(model)
        step = CachedStepScorer(model)
        tokens = torch.tensor([[1, 8, 3, 22, 9, 7, 4, 11]], dtype=torch.int32)
        with torch.no_grad():
            expected = model(tokens.long())
            padded = tokens.clone()
            padded[:, 3:] = 0
            logits, keys, values = prefill(padded, torch.tensor([2], dtype=torch.int32))
            self.assertEqual(tuple(keys.shape), cache_shape(config))
            torch.testing.assert_close(logits, expected[:, 2], atol=1e-6, rtol=1e-6)
            # Future cache entries are deliberately poisoned: the mask must
            # prevent them from changing any valid continuation.
            keys[:, :, :, 3:] = 100
            values[:, :, :, 3:] = -100
            for position in range(3, 8):
                logits, keys, values = step(
                    tokens[:, position : position + 1],
                    torch.tensor([position], dtype=torch.int32),
                    keys,
                    values,
                )
                torch.testing.assert_close(
                    logits, expected[:, position], atol=1e-6, rtol=1e-6
                )
            # Reset uses fresh buffers, without carrying previous-field state.
            keys = torch.zeros(cache_shape(config))
            values = torch.zeros_like(keys)
            logits, _, _ = step(
                tokens[:, :1], torch.tensor([0], dtype=torch.int32), keys, values
            )
            torch.testing.assert_close(logits, expected[:, 0], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
