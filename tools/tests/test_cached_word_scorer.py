import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class CachedWordScorerTests(unittest.TestCase):
    def test_promoted_coreml_outputs_are_owned_and_overflow_is_rejected(self):
        import numpy as np
        from tools.autosuggest.cached_word_scorer import CachedWordScorer
        from tools.autosuggest.subword_cache import cache_shape
        model, _ = self.make_host()
        host = CachedWordScorer(model.config, None, None, cache_dtype=np.float16)
        raw = dict(logits=np.zeros((1, model.config.vocab_size), dtype=np.float32),
                   next_keys=np.ones(cache_shape(model.config), dtype=np.float32),
                   next_values=np.ones(cache_shape(model.config), dtype=np.float32))
        _, keys, values = host.checked(raw)
        self.assertEqual(keys.dtype, np.float16)
        self.assertEqual(values.dtype, np.float16)
        raw["next_keys"].fill(70000)
        self.assertTrue(np.all(keys == 1))
        with self.assertRaisesRegex(ValueError, "overflows"):
            host.checked(raw)
        raw["next_keys"] = raw["next_keys"].astype(np.int32)
        with self.assertRaisesRegex(ValueError, "invalid"):
            host.checked(raw)

    def make_host(self, vocab=32, length=8):
        import torch
        from tools.autosuggest.subword_lm import make_model, ModelConfig
        from tools.autosuggest.subword_cache import PrefillScorer, CachedStepScorer
        from tools.autosuggest.cached_word_scorer import CachedWordScorer

        torch.manual_seed(41)
        model = make_model(ModelConfig(
            vocab_size=vocab, width=16, layers=2, heads=2,
            ff_width=32, sequence_length=length,
        )).eval()
        names = ("logits", "next_keys", "next_values")

        def backend(module, inputs):
            # Simulate a runtime that reuses its output arrays on every call.
            buffers = {}
            def call(features):
                import numpy as np
                with torch.no_grad():
                    result = module(*(torch.from_numpy(features[k]) for k in inputs))
                for key, value in zip(names, result):
                    value = value.numpy()
                    if key not in buffers:
                        buffers[key] = value.copy()
                    else:
                        np.copyto(buffers[key], value)
                return buffers
            return call

        host = CachedWordScorer(
            model.config,
            backend(PrefillScorer(model), ("tokens", "last_index")),
            backend(CachedStepScorer(model), ("token", "position", "key_cache", "value_cache")),
        )
        return model, host

    def test_fork_reorder_recycled_buffers_and_reset_match_full_prefix(self):
        import numpy as np
        import torch
        model, host = self.make_host()
        ids = [1, 7, 8]

        def check(paths):
            actual = host.score(paths)
            with torch.no_grad():
                expected = model(torch.tensor([ids + p for p in paths]))[:, -1].numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

        host.begin(ids)
        check([[]])
        check([[9], [10]])
        check([[10, 11], [9, 12], [10, 13]])
        check([[10, 13, 14], [9, 12, 15]])
        self.assertEqual(host.model_calls, 8)
        ids = [1, 20]
        host.begin(ids)
        check([[]])
        check([[21]])
        self.assertEqual(host.model_calls, 2)
        host.close()
        with self.assertRaisesRegex(ValueError, "inactive"):
            host.score([[]])

    def test_capacity_parent_and_duplicate_paths_fail_closed(self):
        _, host = self.make_host()
        host.begin([1] * 7)
        host.score([[]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            host.score([[8], [8]])
        host.score([[8]])
        with self.assertRaisesRegex(ValueError, "capacity"):
            host.score([[8, 9]])
        host.begin([1])
        host.score([[]])
        host.score([[8]])
        with self.assertRaisesRegex(ValueError, "parent"):
            host.score([[9, 10]])

    @unittest.skipUnless(importlib.util.find_spec("tokenizers"), "optional tokenizer dependency")
    def test_complete_word_ranking_and_scores_match_uncached_decoder(self):
        import torch
        from tokenizers import trainers, pre_tokenizers
        from tools.autosuggest.subword_tokenizer import make_tokenizer
        from tools.autosuggest.decode_subword_words import WordDecoder

        torch.set_num_threads(1)
        tokenizer = make_tokenizer("unicode-marks-v1")
        tokenizer.train_from_iterator(
            ["আমি এখন বাড়ি যাব। তুমি কেমন আছ? meeting কাল হবে।"] * 20,
            trainer=trainers.BpeTrainer(
                vocab_size=320, min_frequency=2,
                special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
            ),
        )
        model, host = self.make_host(tokenizer.get_vocab_size(), 32)
        reference = WordDecoder(tokenizer, model, "cpu", beam=5, max_pieces=4)
        cached = WordDecoder(tokenizer, model, "cpu", beam=5, max_pieces=4, scorer=host)
        for context, prefix in (
            ("", ""), ("আমি এখন", ""), ("meeting কাল", ""), ("আমি " * 40, ""),
            ("আমি", "আ"), ("meeting কাল", "ZZZ"), ("", "আ"),
        ):
            expected, _ = reference.predict(context, prefix=prefix)
            actual, _ = cached.predict(context, prefix=prefix)
            self.assertEqual([w for w, _ in actual], [w for w, _ in expected])
            self.assertTrue(actual)
            self.assertTrue(all(word.startswith(prefix) for word, _ in actual))
            for (_, a), (_, b) in zip(actual, expected):
                self.assertAlmostEqual(a, b, places=5)
        for invalid in ("two words", "!", "ক" * 100):
            with self.assertRaisesRegex(ValueError, "prefix"):
                cached.predict("আমি", prefix=invalid)


if __name__ == "__main__":
    unittest.main()
