"""Optional CPU smoke tests for actual neural CLI data/lineage integration.

Run with the existing ML dependencies installed, for example:
uv run --with torch --with numpy python -m unittest discover -s tools/tests -v
"""

import importlib.util
import json
import os
import subprocess
import sys
import unittest

from tools.tests import test_corpus


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "requires optional training dependency torch"
)
class NeuralCorpusTests(unittest.TestCase):
    def setUp(self):
        fixture = test_corpus.CorpusTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.prepare()
        self.model = fixture.build_model()
        self.root, self.dataset = fixture.root, fixture.dataset

    def run_trainer(self, module, extra=(), validation="validation", success=True):
        report = self.root / f"{module}.json"
        args = [
            sys.executable,
            "-m",
            f"tools.autosuggest.{module}",
            "--model",
            str(self.model),
            "--corpus-dir",
            str(self.dataset / "train"),
            "--validation-corpus-dir",
            str(self.dataset / validation),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--embedding-dim",
            "8",
            "--hidden-dim",
            "8",
            "--context-window",
            "4",
            "--output-report",
            str(report),
            "--log-every-targets",
            "0",
            *extra,
        ]
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(report.read_text())
            self.assertEqual(output["data_provenance"]["status"], "partition_verified")
            self.assertEqual(
                output["data_provenance"]["validation_corpus"]["split"], "validation"
            )
            return output
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_next_word_training_resume_and_test_split_rejection(self):
        checkpoint = self.root / "next-word.pt"
        small = [
            "--baseline-top-k",
            "3",
            "--hybrid-pool-k",
            "3",
            "--hybrid-rank-penalties",
            "0",
        ]
        best = self.root / "best.pt"
        report = self.run_trainer(
            "train_next_word_lm",
            [
                *small,
                "--output-checkpoint",
                str(checkpoint),
                "--best-checkpoint",
                str(best),
            ],
        )
        self.assertTrue(best.exists())
        self.assertEqual(report["final_eval_by_source"]["chat"]["total_targets"], 6)
        self.assertEqual(report["final_eval_by_source"]["chat"]["eligible_targets"], 3)
        self.run_trainer(
            "train_next_word_lm", [*small, "--input-checkpoint", str(checkpoint)]
        )
        self.run_trainer(
            "train_next_word_lm",
            [
                *small,
                "--distill-teacher-checkpoint",
                str(checkpoint),
                "--distill-alpha",
                "0.5",
            ],
        )
        error = self.run_trainer(
            "train_next_word_lm", small, validation="test", success=False
        )
        self.assertIn("forbidden corpus split", error)

    def test_candidate_reranker_training_and_test_split_rejection(self):
        self.run_trainer("train_candidate_reranker", ["--pool-size", "3"])
        error = self.run_trainer(
            "train_candidate_reranker", validation="test", success=False
        )
        self.assertIn("forbidden corpus split", error)

    def test_mid_epoch_checkpoint_resume_matches_uninterrupted_cpu_training(self):
        import torch

        full, partial, resumed = (
            self.root / name for name in ("full.pt", "partial.pt", "resumed.pt")
        )
        small = [
            "--baseline-top-k",
            "3",
            "--hybrid-pool-k",
            "3",
            "--hybrid-rank-penalties",
            "0",
        ]
        self.run_trainer(
            "train_next_word_lm", [*small, "--output-checkpoint", str(full)]
        )
        report = self.run_trainer(
            "train_next_word_lm",
            [*small, "--max-steps", "1", "--output-checkpoint", str(partial)],
        )
        self.assertEqual(report["training_state"]["status"], "step_limit")
        self.assertEqual(report["training_state"]["next_batch"], 1)
        self.run_trainer(
            "train_next_word_lm",
            [
                *small,
                "--input-checkpoint",
                str(partial),
                "--output-checkpoint",
                str(resumed),
            ],
        )
        expected = torch.load(full, map_location="cpu", weights_only=False)
        actual = torch.load(resumed, map_location="cpu", weights_only=False)
        self.assertEqual(actual["training_state"]["status"], "completed")
        self.assertEqual(actual["epoch"], expected["epoch"])
        for key, value in expected["state_dict"].items():
            self.assertTrue(torch.equal(value, actual["state_dict"][key]), key)
        error = self.run_trainer(
            "train_next_word_lm",
            [*small, "--input-checkpoint", str(partial), "--batch-size", "2"],
            success=False,
        )
        self.assertIn("training contract differs", error)

    def test_atomic_checkpoint_failure_preserves_previous_checkpoint(self):
        import torch
        from unittest.mock import patch
        from tools.autosuggest.checkpoints import atomic_torch_save

        path = self.root / "latest.pt"
        atomic_torch_save({"step": 1}, path)
        before = path.read_bytes()
        with patch(
            "tools.autosuggest.checkpoints.torch.save", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                atomic_torch_save({"step": 2}, path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(torch.load(path, weights_only=False)["step"], 1)
        self.assertFalse(list(self.root.glob(".latest.pt-*")))

    def test_evaluation_only_cannot_overwrite_recovery_checkpoint(self):
        checkpoint = self.root / "recover.pt"
        small = [
            "--baseline-top-k",
            "3",
            "--hybrid-pool-k",
            "3",
            "--hybrid-rank-penalties",
            "0",
        ]
        self.run_trainer(
            "train_next_word_lm",
            [*small, "--max-steps", "1", "--output-checkpoint", str(checkpoint)],
        )
        before = checkpoint.read_bytes()
        error = self.run_trainer(
            "train_next_word_lm",
            [
                *small,
                "--epochs",
                "0",
                "--input-checkpoint",
                str(checkpoint),
                "--output-checkpoint",
                str(checkpoint),
            ],
            success=False,
        )
        self.assertIn("evaluation-only runs cannot write training checkpoints", error)
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_unknown_only_source_remains_in_all_target_metrics(self):
        import torch
        from tools.autosuggest.train_next_word_lm import (
            ExampleSet,
            NextWordLm,
            evaluate_model_by_source,
        )

        examples = ExampleSet(
            contexts=torch.empty((0, 4), dtype=torch.long),
            labels=torch.empty(0, dtype=torch.long),
            source_ids=torch.empty(0, dtype=torch.long),
            source_names=("unknown",),
            total_targets=7,
            eligible_targets=0,
            examples_by_source={},
            scanned_sentences_by_source={"unknown": 1},
            total_targets_by_source={"unknown": 7},
        )
        model = NextWordLm(8, 4, 8, 8, "gru", 0, 1, 1)
        result = evaluate_model_by_source(model, examples, torch.device("cpu"))
        self.assertEqual(result["unknown"]["total_targets"], 7)
        self.assertEqual(result["unknown"]["top5_all_targets"], 0)

    def test_frozen_checkpoint_test_evaluation_and_train_rejection(self):
        from argparse import Namespace
        from tools.autosuggest.eval_next_word_lm import evaluate

        checkpoint = self.root / "frozen.pt"
        self.run_trainer(
            "train_next_word_lm",
            [
                "--baseline-top-k",
                "3",
                "--hybrid-pool-k",
                "3",
                "--hybrid-rank-penalties",
                "0",
                "--output-checkpoint",
                str(checkpoint),
            ],
        )
        args = Namespace(
            model=self.model,
            checkpoint=checkpoint,
            corpus_dir=self.dataset / "test",
            device="cpu",
            max_sentences_per_source=100,
            max_examples_per_source=100,
            pool_k=3,
            rank_penalty=0.5,
            lock_first=False,
        )
        report = evaluate(args)
        self.assertEqual(
            report["data_provenance"]["evaluation_corpus"]["split"], "test"
        )
        self.assertEqual(report["by_source"]["chat"]["hybrid"]["total_targets"], 6)
        self.assertEqual(report["macro_by_source"]["hybrid"]["top5_all_targets"], 0.5)
        args.corpus_dir = self.dataset / "train"
        with self.assertRaisesRegex(ValueError, "forbidden corpus split"):
            evaluate(args)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "requires optional ONNX export dependencies",
    )
    def test_export_verifies_checkpoint_and_rejects_test_policy_selection(self):
        self.check_export("gru")

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "requires optional ONNX export dependencies",
    )
    def test_transformer_exports_with_masked_and_empty_contexts(self):
        self.check_export("transformer")

    def check_export(self, architecture):
        checkpoint = self.root / "export.pt"
        self.run_trainer(
            "train_next_word_lm",
            [
                "--baseline-top-k",
                "3",
                "--hybrid-pool-k",
                "3",
                "--hybrid-rank-penalties",
                "0",
                "--output-checkpoint",
                str(checkpoint),
                "--architecture",
                architecture,
                "--max-grad-norm",
                "1",
            ],
        )
        report_path = self.root / "export-report.json"
        args = [
            sys.executable,
            "-m",
            "tools.autosuggest.export_next_word_lm",
            "--checkpoint",
            str(checkpoint),
            "--artifact",
            str(self.model),
            "--output",
            str(self.root / "scorer.onnx"),
            "--report",
            str(report_path),
            "--pool-k",
            "3",
            "--benchmark-iterations",
            "2",
            "--compare-samples",
            "2",
            "--max-examples-per-source",
            "1",
            "--rank-penalties",
            "0.5",
            "--no-quantize",
            "--native-fixtures",
            str(self.root / "native-fixtures.json"),
            "--corpus-dir",
            str(self.dataset / "validation"),
        ]
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["data_provenance"]["status"], "partition_verified")
        self.assertLess(report["verification"]["onnx_vs_pytorch"]["max_abs_diff"], 1e-5)
        fixtures = json.loads((self.root / "native-fixtures.json").read_text())
        self.assertEqual(fixtures["checkpointSHA256"], report["checkpoint_sha256"])
        self.assertEqual(len(fixtures["contexts"][0]), 4)
        self.assertEqual(len(fixtures["candidates"][0]), 3)
        import numpy as np
        import onnxruntime as ort
        import torch
        from tools.autosuggest.export_next_word_lm import load_model
        from tools.autosuggest.train_next_word_lm import score_candidate_pool

        model, _ = load_model(checkpoint)
        contexts = np.zeros((1, 4), dtype=np.int64)
        candidates = np.full((1, 3), 3, dtype=np.int64)
        session = ort.InferenceSession(
            str(self.root / "scorer.onnx"), providers=["CPUExecutionProvider"]
        )
        actual = session.run(
            ["scores"], {"contexts": contexts, "candidate_ids": candidates}
        )[0]
        with torch.no_grad():
            expected = score_candidate_pool(
                model, torch.from_numpy(contexts), torch.from_numpy(candidates)
            ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        args[-1] = str(self.dataset / "test")
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden corpus split", result.stderr)

    def test_transformer_padding_contract_and_finite_empty_context_gradients(self):
        import torch
        from tools.autosuggest.train_next_word_lm import NextWordLm

        model = NextWordLm(8, 4, 8, 16, "transformer", 0, 2, 2, "masked-v1")
        contexts = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 3], [1, 3, 4, 5]])
        scores = model(contexts)
        self.assertTrue(torch.isfinite(scores).all())
        scores.square().mean().backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        legacy = NextWordLm(8, 4, 8, 16, "transformer", 0, 2, 2)
        self.assertEqual(legacy.transformer_padding_mode, "unmasked-v0")
        with self.assertRaisesRegex(ValueError, "padding contract"):
            NextWordLm(8, 4, 8, 16, "transformer", 0, 2, 2, "unknown-mode")

    def test_transformer_initialization_scale_and_independent_layers(self):
        import torch
        from tools.autosuggest.train_next_word_lm import NextWordLm

        torch.manual_seed(17)
        model = NextWordLm(
            128, 16, 32, 64, "transformer", 0, 2, 4, "masked-v1", "independent-v1"
        )
        self.assertLess(float(model.position_embedding.weight.detach().std()), 0.03)
        self.assertGreater(float(model.position_embedding.weight.detach().std()), 0.01)
        self.assertFalse(
            torch.equal(
                model.encoder.layers[0].self_attn.in_proj_weight,
                model.encoder.layers[1].self_attn.in_proj_weight,
            )
        )
        self.assertTrue(torch.equal(model.token_embedding.weight[0], torch.zeros(32)))
        self.assertEqual(model.transformer_initialization, "independent-v1")


if __name__ == "__main__":
    unittest.main()
