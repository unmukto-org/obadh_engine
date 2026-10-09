"""Corpus integrity and real vocabulary/retrieval pipeline regression tests.

No data submodules, network, NumPy, or ML runtime are required.
"""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.autosuggest.build_ngram_lm import SourceWeights, build_ngram_lm
from tools.autosuggest.build_vocab import build_vocab
from tools.autosuggest.eval_ngram_lm import evaluate
from tools.corpus.partition import (
    FIELDS,
    OUTPUT_FIELDS,
    assigned_split,
    build,
    group_key,
)
from tools.corpus.provenance import (
    SPLITS,
    evaluation_provenance,
    load_partition,
    neural_training_provenance,
    read_object,
    sha256_file,
    training_provenance,
    verify_checkpoint_provenance,
)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.dataset = self.root / "partitioned"
        self.ids = {}
        for i in range(1000):
            name = f"doc-{i}"
            split = assigned_split(group_key("chat", name), "obadh-v1", 9000, 500)
            self.ids.setdefault(SPLITS[split], name)
            if len(self.ids) == 3:
                break
        self.rows = [
            ["chat", self.ids[split], i, 2, f"আমি {split}{i}", ""]
            for split in SPLITS
            for i in range(1, 4)
        ]

    def write_rows(self, rows=None, fields=OUTPUT_FIELDS):
        (self.raw / "sentences").mkdir(parents=True, exist_ok=True)
        with gzip.open(
            self.raw / "sentences/chat.tsv.gz", "wt", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(fields)
            writer.writerows(self.rows if rows is None else rows)

    def prepare(self):
        self.write_rows()
        return build(self.raw, self.dataset)

    def read_rows(self, split):
        rows = []
        for path in sorted((self.dataset / split / "sentences").glob("*.gz")):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle, delimiter="\t"))
        return rows

    def build_model(self, **overrides):
        vocab = self.root / "vocab.tsv"
        if not vocab.exists():
            build_vocab(self.dataset / "train", vocab, 128, 1)
        args = dict(
            corpus_dir=self.dataset / "train",
            vocab_path=vocab,
            output=self.root / "model.bin",
            backend="memory",
            sqlite_path=self.root / "counts.sqlite",
            sources=None,
            source_weights=SourceWeights({}),
            max_sentences=None,
            skip_sentences_per_source=0,
            max_sentences_per_source=None,
            reuse_sqlite=False,
            log_every_sentences=0,
            max_candidates_per_prefix=64,
            unigram_size=128,
            min_count=1,
            bigram_min_count=None,
            trigram_min_count=None,
            fourgram_min_count=None,
            batch_size=100,
            smoothing=64.0,
            backoff_alpha=0.4,
            kneser_ney_discount=0.75,
            score_mode="count",
            max_context_order=3,
            compact_count_records=True,
        )
        args.update(overrides)
        build_ngram_lm(**args)
        return args["output"]

    def test_reproducible_bytes_and_whole_documents(self):
        first = self.prepare()
        second_root = self.root / "second"
        second = build(self.raw, second_root)
        self.assertEqual(first, second)
        for split in SPLITS:
            self.assertEqual(
                {r["document_id"] for r in self.read_rows(split)}, {self.ids[split]}
            )
            self.assertEqual(len(self.read_rows(split)), 3)
            for path in (self.dataset / split).rglob("*"):
                if path.is_file():
                    self.assertEqual(
                        path.read_bytes(),
                        (
                            second_root / split / path.relative_to(self.dataset / split)
                        ).read_bytes(),
                    )
            self.assertEqual(
                load_partition(self.dataset / split)["dataset_id"], first["dataset_id"]
            )

    def test_v1_assignment_contract_golden_values(self):
        # Changing serialization or hashing must require a new policy version;
        # deriving expected membership with assigned_split would miss that drift.
        self.assertEqual(
            group_key("chat", "doc-0"),
            "2fd09eb625522f8b6df6c670801b488898b665ce4d5b1228e284fd6940ed9a98",
        )
        self.assertEqual(
            group_key("chat", "ignored", "বাংলা"),
            "7acf5939015fbb89ee19d49b64bb36430346b57e5949564ee86f2f785d3744f7",
        )
        for document, expected in (
            ("doc-0", "train"),
            ("doc-2", "validation"),
            ("doc-9", "test"),
        ):
            actual = assigned_split(group_key("chat", document), "obadh-v1", 9000, 500)
            self.assertEqual(SPLITS[actual], expected)

    def test_input_order_does_not_change_membership_or_partition_bytes(self):
        first = self.prepare()
        self.write_rows(list(reversed(self.rows)))
        second = build(self.raw, self.root / "reordered")
        self.assertEqual(first["splits"], second["splits"])
        # Raw input byte provenance intentionally differs.
        self.assertNotEqual(first["dataset_id"], second["dataset_id"])

    def test_normalized_duplicates_quarantined_everywhere_original_text_preserved(self):
        unique = "নাম \u09dc"
        duplicates = [
            ["chat", self.ids["train"], 9, 2, "অক্ষর \u09dc", ""],
            ["chat", self.ids["test"], 9, 2, "অক্ষর \u09a1\u09bc", ""],
            ["chat", self.ids["validation"], 10, 2, unique, ""],
        ]
        self.write_rows(self.rows + duplicates)
        manifest = build(self.raw, self.dataset)
        self.assertEqual(
            manifest["quarantine"],
            {"sentences": 2, "tokens": 4, "distinct_sentences": 1},
        )
        self.assertTrue(
            all(
                not r["tokens"].startswith("অক্ষর")
                for split in SPLITS
                for r in self.read_rows(split)
            )
        )
        self.assertIn(unique, [r["tokens"] for r in self.read_rows("validation")])

    def test_explicit_global_groups_cross_sources(self):
        self.write_rows(
            self.rows
            + [
                ["chat", "original", 1, 2, "মূল বাক্য", "pair-1"],
                ["synthetic", "variant", 1, 2, "ভুল বাক্য", "pair-1"],
            ]
        )
        build(self.raw, self.dataset)
        membership = {
            r["document_id"]: split for split in SPLITS for r in self.read_rows(split)
        }
        self.assertEqual(membership["original"], membership["variant"])

    def test_declared_holdout_groups_override_hash_assignment_globally(self):
        self.write_rows(
            self.rows
            + [
                ["books", "work", 1, 2, "বই পড়ি", "reserved-book"],
                ["alternate", "edition", 1, 2, "বই রাখি", "reserved-book"],
            ]
        )
        manifest = build(
            self.raw, self.dataset, group_assignments={"reserved-book": "test"}
        )
        self.assertEqual(
            manifest["policy"]["group_assignments"], {"reserved-book": "test"}
        )
        membership = {
            row["document_id"]: split
            for split in SPLITS
            for row in self.read_rows(split)
        }
        self.assertEqual(membership["work"], "test")
        self.assertEqual(membership["edition"], "test")

    def test_stale_or_invalid_holdout_registry_is_rejected(self):
        self.write_rows()
        for mapping in ({"missing": "test"}, {"missing": "development"}, {"": "train"}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                build(self.raw, self.dataset, group_assignments=mapping)
            self.assertFalse(self.dataset.exists())

    def test_unpublished_acquisition_cannot_be_partitioned(self):
        self.write_rows()
        (self.raw / ".building").touch()
        with self.assertRaisesRegex(
            ValueError, "input corpus publication is incomplete"
        ):
            build(self.raw, self.dataset)

    def test_legacy_five_column_corpus_supported(self):
        self.write_rows([row[:5] for row in self.rows], fields=FIELDS)
        build(self.raw, self.dataset)
        self.assertEqual(load_partition(self.dataset / "train")["split"], "train")

    def test_invalid_input_never_publishes(self):
        invalid_rows = [
            ["../escape", "doc", 1, 2, "আমি যাই", ""],
            ["CON", "doc", 1, 2, "আমি যাই", ""],
            ["Chat", "doc", 1, 2, "আমি যাই", ""],
            ["chat", "", 1, 2, "আমি যাই", ""],
            ["chat", "doc", -1, 2, "আমি যাই", ""],
            ["chat", "doc", 1, 3, "আমি যাই", ""],
            ["chat", "doc", 1, 2, "আমি  যাই", ""],
            ["chat", "doc", 1, 2, "আমি \x00যাই", ""],
            self.rows[0],  # Duplicate sentence identity.
            ["chat", self.ids["train"], 99, 2, "অন্য বাক্য", "changed-group"],
        ]
        for row in invalid_rows:
            with self.subTest(row=row):
                self.write_rows(self.rows + [row])
                with self.assertRaises(ValueError):
                    build(self.raw, self.dataset)
                self.assertFalse(self.dataset.exists())
                self.assertFalse(list(self.root.glob(".partitioned-*")))

    def test_empty_split_and_invalid_policy_rejected(self):
        self.write_rows(self.rows[:3])
        with self.assertRaisesRegex(ValueError, "partition is empty"):
            build(self.raw, self.dataset)
        for kwargs in (
            {"seed": ""},
            {"train_bps": 0},
            {"validation_bps": 0},
            {"train_bps": 9999},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(self.raw, self.dataset, **kwargs)

    def test_existing_output_is_not_modified(self):
        self.prepare()
        before = (self.dataset / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            build(self.raw, self.dataset)
        self.assertEqual(before, (self.dataset / "manifest.json").read_bytes())

    def test_tampering_and_incomplete_publication_rejected(self):
        self.prepare()
        shard = self.dataset / "train/sentences/chat.tsv.gz"
        original = shard.read_bytes()
        shard.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            training_provenance(self.dataset / "train")
        shard.write_bytes(original)
        extra = shard.with_name("extra.tsv.gz")
        shutil.copyfile(shard, extra)
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            training_provenance(self.dataset / "train")
        extra.unlink()
        (self.dataset / ".building").touch()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            training_provenance(self.dataset / "train")
        (self.dataset / ".building").unlink()
        (self.dataset / "train/manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing partition manifest"):
            training_provenance(self.dataset / "train")

    def test_train_only_vocab_retrieval_and_verified_evaluation(self):
        self.prepare()
        model = self.build_model()
        vocab = (self.root / "vocab.tsv").read_text()
        self.assertIn("train1", vocab)
        self.assertNotIn("test1", vocab)
        self.assertNotIn("validation1", vocab)
        result = evaluate(model, self.dataset / "test", None, 0, None, None, 3, "full")
        self.assertEqual(
            result["evaluation_provenance"]["status"], "partition_verified"
        )
        self.assertEqual(result["total_targets"], 6)
        self.assertEqual(result["skipped_unknown_targets"], 3)
        with self.assertRaisesRegex(ValueError, "forbidden corpus split"):
            build_vocab(self.dataset / "test", self.root / "bad.tsv", 128, 1)
        with self.assertRaisesRegex(ValueError, "forbidden corpus split"):
            evaluation_provenance(model, self.dataset / "train")

    def test_foreign_vocab_model_and_modified_model_rejected(self):
        self.prepare()
        model = self.build_model()
        sidecar = model.with_suffix(".manifest.json")
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, "same partition set"):
            evaluation_provenance(model, self.dataset / "test")
        self.build_model()
        model.write_bytes(model.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "derived artifact digest"):
            evaluation_provenance(model, self.dataset / "test")
        (self.root / "vocab.manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "not built from this training"):
            self.build_model()

    def test_sqlite_reuse_rejects_changed_ingestion_selection(self):
        self.prepare()
        model = self.build_model(backend="sqlite")
        before = read_object(model.with_suffix(".manifest.json"))
        self.build_model(backend="sqlite", reuse_sqlite=True)
        after = read_object(model.with_suffix(".manifest.json"))
        for key in (
            "observed_sentences",
            "observed_tokens",
            "source_sentences",
            "artifact_sha256",
        ):
            self.assertEqual(before[key], after[key])
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.build_model(backend="sqlite", reuse_sqlite=True, max_sentences=1)

    def test_incomplete_sqlite_counts_cannot_be_reused(self):
        from tools.autosuggest.build_ngram_lm import SQLITE_COUNT_METADATA_TABLE

        self.prepare()
        self.build_model(backend="sqlite")
        with sqlite3.connect(self.root / "counts.sqlite") as connection:
            connection.execute(
                f"UPDATE {SQLITE_COUNT_METADATA_TABLE} SET value = '0' WHERE key = 'counting_complete'"
            )
        connection.close()
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.build_model(backend="sqlite", reuse_sqlite=True)

    def test_corrupt_manifest_cannot_downgrade_partition_to_legacy(self):
        self.prepare()
        path = self.dataset / "train/manifest.json"
        manifest = read_object(path)
        manifest["artifact"] = "legacy"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "invalid partition manifest kind"):
            training_provenance(self.dataset / "train")

    def test_neural_training_requires_independent_validation_and_checkpoint_lineage(
        self,
    ):
        self.prepare()
        model = self.build_model()
        with self.assertRaisesRegex(ValueError, "validation-corpus-dir"):
            neural_training_provenance(model, self.dataset / "train", None)
        with self.assertRaisesRegex(ValueError, "forbidden corpus split"):
            neural_training_provenance(
                model, self.dataset / "train", self.dataset / "test"
            )
        provenance = neural_training_provenance(
            model, self.dataset / "train", self.dataset / "validation"
        )
        checkpoint = {
            "report": {
                "data_provenance": provenance,
                "artifact": {"sha256": sha256_file(model)},
            }
        }
        verify_checkpoint_provenance(checkpoint, provenance, model)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            verify_checkpoint_provenance({}, provenance, model)
        checkpoint["report"]["artifact"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            verify_checkpoint_provenance(checkpoint, provenance, model)

    def test_legacy_evaluation_is_explicitly_unverified(self):
        self.write_rows()
        self.assertIsNone(training_provenance(self.raw))
        self.assertEqual(
            evaluation_provenance(self.root / "legacy.bin", self.raw)["status"],
            "unverified",
        )

    def test_cli_verify_and_error_exit(self):
        self.prepare()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.corpus.partition",
                "verify",
                "--dataset",
                str(self.dataset),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["test"]["split"], "test")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.corpus.partition",
                "build",
                "--corpus-dir",
                str(self.raw),
                "--output",
                str(self.dataset),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
