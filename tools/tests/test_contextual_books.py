import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.corpus.contextual_books import (
    author_members,
    build,
    normalize_text,
    passage_keys,
    text_windows,
)


class ContextualBooksTests(unittest.TestCase):
    def test_preserves_dialogue_and_typing_symbols(self):
        source = "— যাবে?\r\nহ্যাঁ!\r\n\r\nনা। meeting ১২:৩০ 🙂 র‌্যাব"
        self.assertEqual(normalize_text(source), source.replace("\r\n", "\n"))
        self.assertEqual(list(text_windows("হ্যাঁ!")), ["হ্যাঁ!"])

    def test_author_aliases_apply_to_each_contributor(self):
        self.assertEqual(
            author_members("Alias | Translator | Real", {"Alias": "Real"}),
            ["Real", "Translator"],
        )

    def test_long_overlap_ignores_punctuation_but_short_replies_remain(self):
        text = " ".join("শব্দ" + str(i) for i in range(40))
        self.assertEqual(passage_keys(text), passage_keys(text.replace(" ", ", ")))
        self.assertEqual(passage_keys("হ্যাঁ।"), [])

    def test_work_isolation_provenance_and_holdout_alias_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            epubs = root / "epubs"
            epubs.mkdir()
            common = " ".join("শব্দ" + str(i) for i in range(40))
            books = []
            for i, role in enumerate(("train", "validation"), 1):
                books.append(
                    dict(
                        course_id=i,
                        group_id=str(i),
                        title="বই",
                        author=f"writer{i}",
                        split=role,
                    )
                )
                text = common + "\n— হ্যাঁ!"
                rows = [
                    dict(
                        index=0,
                        url=f"https://example.test/{i}",
                        text=text,
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    ),
                    dict(
                        index=1,
                        url=f"https://example.test/{i}/1",
                        text=f"কথা{i}?\nহ্যাঁ!",
                        text_sha256=hashlib.sha256(f"কথা{i}?\nহ্যাঁ!".encode()).hexdigest(),
                    ),
                    dict(
                        index=2,
                        url=f"https://example.test/{i}/2",
                        text="হ্যাঁ!",
                        text_sha256=hashlib.sha256("হ্যাঁ!".encode()).hexdigest(),
                    ),
                ]
                (raw / f"{i}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            plan = root / "plan.json"
            plan.write_text(json.dumps({"books": books}))
            args = argparse.Namespace(
                raw_books=[raw],
                epubs=epubs,
                original_plan=plan,
                admission_plan=None,
                attribution=None,
                output=root / "result",
            )
            build(args)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["counts"]["quarantined_windows"], 2)
            for split in ("train", "validation"):
                with gzip.open(args.output / (split + ".jsonl.gz"), "rt") as f:
                    rows = [json.loads(x) for x in f]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[1]["text"], "হ্যাঁ!")
                self.assertTrue(rows[0]["text"].endswith("হ্যাঁ!"))
                self.assertIn("reference", rows[0])
                self.assertEqual(
                    rows[0]["sha256"],
                    hashlib.sha256(rows[0]["text"].encode()).hexdigest(),
                )
            new = root / "new.json"
            new.write_text(
                json.dumps({"books": [], "author_holdouts": {"alias": "validation"}})
            )
            attribution = root / "attribution.json"
            attribution.write_text(json.dumps({"aliases": {"alias": "writer1"}}))
            args.admission_plan = new
            args.attribution = attribution
            args.output = root / "bad"
            with self.assertRaisesRegex(ValueError, "holdout contamination"):
                build(args)


if __name__ == "__main__":
    unittest.main()
