from tools.tests.dependencies import require

require('torch', 'numpy', 'tokenizers', 'regex')

import unittest
import torch
from tokenizers import trainers, pre_tokenizers
from tools.autosuggest.subword_tokenizer import make_tokenizer
from tools.correction.edit_model import EditConfig, EditCorrector
from tools.correction.rank_candidates import EditCandidateScorer


class EditCandidateTests(unittest.TestCase):
    def test_identity_competes_and_candidate_order_cannot_change_scores(self):
        torch.set_num_threads(1)
        tokenizer = make_tokenizer("unicode-marks-v1")
        tokenizer.train_from_iterator(["আমি বাড়ি যাই। তুমি বাড়ি যাও।"] * 10, trainer=trainers.BpeTrainer(vocab_size=320, special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        model = EditCorrector(EditConfig(vocab_size=tokenizer.get_vocab_size(), width=16, layers=2, heads=2, ff_width=32, sequence_length=32)).eval()
        scorer = EditCandidateScorer(model, tokenizer, "cpu")
        candidates = ["আমি বাড়ি যাও।", "আমি বাড়ি যাই।"]
        a = scorer.rank("আমি বাড়ি যায়।", candidates)
        b = scorer.rank("আমি বাড়ি যায়।", candidates[::-1])
        self.assertEqual(a, b)
        self.assertEqual(sum(r["unchanged"] for r in a), 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(r["score"])) for r in a))
        with self.assertRaisesRegex(ValueError, "budget"):
            scorer.rank("আমি", [str(i) for i in range(40)])


if __name__ == "__main__":
    unittest.main()
