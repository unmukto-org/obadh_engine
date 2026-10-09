from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools.corpus.contextual_fulltext import (
    benchmark_patterns,
    overlaps_benchmark,
    build,
)
from tools.corpus.partition import assigned_split, group_key
from tools.corpus.provenance import digest_json, sha256_file, write_json


class BenchmarkExclusionTests(unittest.TestCase):
    def test_ingestion_recovery_matches_fresh_output_and_quarantines_bad_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            books, benchmark, stage = root / "books", root / "benchmark", root / "stage"
            for p in (books, benchmark, stage):
                p.mkdir()
            (stage / ".building").touch()
            (benchmark / "dev.src").write_text(
                "এখানে রাখা এই বাক্য শুধুই মূল্যায়নের জন্য আলাদা করে সংরক্ষিত", encoding="utf-8"
            )
            files = []
            for split in ("train", "validation", "test"):
                path = books / (split + ".jsonl.gz")
                with gzip.open(path, "wt", encoding="utf-8") as out:
                    pass
                files.append(
                    dict(
                        path=path.name,
                        bytes=path.stat().st_size,
                        sha256=sha256_file(path),
                    )
                )
            manifest = dict(kind="obadh-contextual-text", files=files)
            manifest["dataset_id"] = digest_json(manifest)
            write_json(books / "manifest.json", manifest)
            inventory = root / "documents.tsv.gz"
            with gzip.open(inventory, "wt", encoding="utf-8") as out:
                out.write(
                    "source\tdocument_id\tsource_ref\nnews\tn1\thttps://example.test/n1\nnews\tn2\thttps://example.test/n2\nwiki\tw1\thttps://example.test/w1\n"
                )
            news_path = root / "news.json"
            news_path.write_text("[]")
            text = "— তুমি যাবে?\nহ্যাঁ! meeting আছে।"
            digest = hashlib.sha256(text.encode()).hexdigest()
            role = ("train", "validation", "test")[
                assigned_split(group_key("news", "n1"), "obadh-v1", 9000, 500)
            ]
            db = sqlite3.connect(stage / "staging.sqlite")
            db.executescript(
                "CREATE TABLE records(id INTEGER PRIMARY KEY,source TEXT,work TEXT,split TEXT,reference TEXT,ordinal INTEGER,authors TEXT,text TEXT,digest TEXT); CREATE TABLE anchors(key BLOB,record INTEGER,split TEXT); CREATE TABLE excluded_works(work TEXT PRIMARY KEY); CREATE TABLE documents(source TEXT,work TEXT,digest TEXT,PRIMARY KEY(source,work));"
            )
            db.execute("INSERT INTO documents VALUES(?,?,?)", ("news", "n1", digest))
            db.execute(
                "INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    1,
                    "news",
                    "n1",
                    role,
                    "https://example.test/n1",
                    0,
                    "[]",
                    text,
                    digest,
                ),
            )
            db.commit()
            db.close()
            news = [
                ("n1", "https://example.test/n1", "title", [text], 0),
                ("n2", "https://example.test/n2", "bad", ["ক" * 9000], 0),
            ]
            wiki = [
                (
                    "w1",
                    "https://example.test/w1",
                    "title",
                    [(benchmark / "dev.src").read_text(encoding="utf-8")],
                    0,
                )
            ]
            outputs = []
            for recover in (None, stage):
                output = root / ("recovered" if recover else "fresh")
                args = argparse.Namespace(
                    news_json=news_path,
                    wiki_dir=root,
                    document_inventory=inventory,
                    books=books,
                    exclude_benchmark=benchmark,
                    output=output,
                    limit=None,
                    recover_stage=recover,
                )
                with (
                    patch(
                        "tools.corpus.contextual_fulltext.iter_news_documents",
                        return_value=iter(news),
                    ),
                    patch(
                        "tools.corpus.contextual_fulltext.iter_wiki_documents",
                        return_value=iter(wiki),
                    ),
                ):
                    build(args)
                report = json.loads((output / "manifest.json").read_text())
                self.assertEqual(report["counts"]["quality_excluded_works"], 1)
                self.assertEqual(report["counts"]["benchmark_excluded_works"], 1)
                outputs.append(output)
            for split in ("train", "validation", "test"):
                self.assertEqual(
                    (outputs[0] / (split + ".jsonl.gz")).read_bytes(),
                    (outputs[1] / (split + ".jsonl.gz")).read_bytes(),
                )
            self.assertTrue((stage / ".building").exists())

    def test_source_and_gold_spans_survive_punctuation_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "আমি আজ সকালে বন্ধুদের সঙ্গে অনেক দূরের একটি গ্রামে গিয়েছিলাম"
            target = source.replace("গিয়েছিলাম", "গিয়েছি")
            (root / "dev.src").write_text(source + "\n", encoding="utf-8")
            (root / "dev.tgt").write_text(target + "\n", encoding="utf-8")
            patterns, receipts = benchmark_patterns(root)
            self.assertEqual(len(receipts), 2)
            self.assertTrue(
                overlaps_benchmark(
                    "শুরু। " + source.replace(" ", ", ") + "। শেষ।", patterns
                )
            )
            self.assertTrue(overlaps_benchmark(target, patterns))
            self.assertFalse(overlaps_benchmark("আমি আজ সকালে যাব।", patterns))

    def test_missing_or_unbounded_short_exclusions_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                benchmark_patterns(root)
            (root / "dev.src").write_text("হ্যাঁ।\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark_patterns(root)


if __name__ == "__main__":
    unittest.main()
