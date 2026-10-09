"""Offline acquisition contract tests; parser integration needs epub-exporter."""

import hashlib
import importlib.util
import io
import json
from email.message import Message
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.corpus.acquire_books import (
    PageCache,
    acquire,
    canonical_url,
    extract_chapter,
    validate_plan,
)
from tools.corpus.partition import build
from tools.corpus.provenance import read_object, sha256_file


class Response(io.BytesIO):
    def __init__(self, url, body):
        super().__init__(body)
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = "text/html; charset=utf-8"


class AcquisitionCacheTests(unittest.TestCase):
    def test_urls_are_canonical_and_bounded_to_source_pages(self):
        self.assertEqual(
            canonical_url("https://ebanglalibrary.com/books/বই/#section"),
            "https://www.ebanglalibrary.com/books/%E0%A6%AC%E0%A6%87/",
        )
        for url in (
            "http://www.ebanglalibrary.com/books/a/",
            "https://evil.test/books/a/",
            "https://www.ebanglalibrary.com/wp-admin/",
            "https://www.ebanglalibrary.com/books/a/?next=x",
            "https://www.ebanglalibrary.com/books/%2e%2e/wp-admin/",
            "https://user:password@www.ebanglalibrary.com/books/a/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                canonical_url(url)

    def test_cache_replay_is_offline_and_checks_actual_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            url = "https://www.ebanglalibrary.com/books/a/"
            body = "<html>বাংলা বই</html>".encode()
            opener = Mock()
            opener.open.return_value = Response(url, body)
            with patch("tools.corpus.acquire_books.build_opener", return_value=opener):
                text, receipt = PageCache(root).fetch(url)
                self.assertEqual(
                    PageCache(root, offline=True).fetch(url), (text, receipt)
                )
                self.assertEqual(opener.open.call_count, 1)
                self.assertNotIn(
                    "Authorization", dict(opener.open.call_args.args[0].header_items())
                )
                (root / "blobs" / f"{receipt['sha256']}.html").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    PageCache(root, offline=True).fetch(url)

    def test_offline_cache_miss_and_page_size_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            url = "https://www.ebanglalibrary.com/books/a/"
            with self.assertRaisesRegex(ValueError, "offline cache"):
                PageCache(root, offline=True).fetch(url)
            opener = Mock()
            opener.open.return_value = Response(url, b"larger than allowed")
            with (
                patch("tools.corpus.acquire_books.build_opener", return_value=opener),
                patch("tools.corpus.acquire_books.MAX_PAGE_BYTES", 4),
            ):
                with self.assertRaisesRegex(ValueError, "size bound"):
                    PageCache(root).fetch(url)
            self.assertFalse((root / "receipts").exists())

    def test_chapter_redirect_to_book_front_matter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            opener = Mock()
            opener.open.return_value = Response(
                "https://www.ebanglalibrary.com/books/a/", b"<html>intro</html>"
            )
            with patch("tools.corpus.acquire_books.build_opener", return_value=opener):
                with self.assertRaisesRegex(ValueError, "redirect changed page kind"):
                    PageCache(Path(temporary)).fetch(
                        "https://www.ebanglalibrary.com/lessons/a/"
                    )

    def test_plan_rejects_duplicate_or_conflicting_books(self):
        entry = {
            "url": "https://www.ebanglalibrary.com/books/a/",
            "course_id": 1,
            "chapters": 1,
            "group_id": "work",
            "split": "train",
        }
        plan = {
            "artifact": "obadh-book-acquisition-plan",
            "version": 1,
            "authorization": "fixture",
            "books": [entry],
        }
        validate_plan(plan)
        plan["books"].append(dict(entry))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_plan(plan)
        plan["books"][1].update(
            url="https://www.ebanglalibrary.com/books/b/", course_id=2, split="test"
        )
        with self.assertRaisesRegex(ValueError, "different split"):
            validate_plan(plan)


@unittest.skipUnless(
    importlib.util.find_spec("wp_learndash_epub"),
    "requires optional epub-exporter parser",
)
class BookParserIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = self.root / "plan.json"
        self.pages = {}
        books = []
        for index, split in enumerate(("train", "validation", "test"), 1):
            url = f"https://www.ebanglalibrary.com/books/{index}/"
            lesson = f"https://www.ebanglalibrary.com/lessons/{index}/"
            self.pages[url] = (
                f'<body class="postid-{index}"><h1>বই {index}</h1><div class="entry-terms-authors"><a>লেখক</a></div><div class="ld-lesson-list"><a class="ld-item-name" href="{lesson}">অধ্যায়</a></div></body>'
            )
            # Distinct words ensure the fixture survives cross-split quarantine.
            word = ("আকাশ", "বাতাস", "পাতাল")[index - 1]
            self.pages[lesson] = (
                f'<h1>অধ্যায়</h1><div class="ld-tab-content entry-content"><div id="ftwp-postcontent"><p>আমি <strong>{word}</strong> নিয়ে আজ একটি নতুন গল্প লিখতে বসেছি। আমাদের এই গল্পের ভিতরে অনেক নতুন কথা আছে।</p><div class="ld-content-actions">BAD NAVIGATION</div><div id="ftwp-container">BAD TOC</div><script>BAD SCRIPT</script></div></div><div class="comments">BAD COMMENTS</div>'
            )
            books.append(
                {
                    "url": url,
                    "course_id": index,
                    "chapters": 1,
                    "group_id": f"work-{index}",
                    "split": split,
                }
            )
        self.plan.write_text(
            json.dumps(
                {
                    "artifact": "obadh-book-acquisition-plan",
                    "version": 1,
                    "authorization": "fixture",
                    "books": books,
                }
            )
        )

    def cache(self):
        pages = self.pages

        class OfflineFixture:
            max_pages = 10

            def fetch(self, url):
                text = pages[url]
                return text, {
                    "url": url,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "bytes": len(text.encode()),
                }

        return OfflineFixture()

    def test_clean_book_acquisition_and_declared_partitions(self):
        output = self.root / "corpus"
        manifest = acquire(self.plan, output, self.cache())
        self.assertEqual(len(manifest["books"]), 3)
        for book in manifest["books"]:
            chapter = json.loads(
                (output / book["text_file"]).read_text().splitlines()[0]
            )
            self.assertNotIn("BAD", chapter["text"])
            self.assertIn("আমি ", chapter["text"])
            self.assertEqual(
                book["text_file_sha256"], sha256_file(output / book["text_file"])
            )
        partitions = build(
            output,
            self.root / "partitioned",
            group_assignments=manifest["group_assignments"],
        )
        for split in ("train", "validation", "test"):
            self.assertEqual(partitions["splits"][split]["statistics"]["documents"], 1)
        second = acquire(self.plan, self.root / "repeat", self.cache())
        self.assertEqual(manifest, second)
        with self.assertRaises(FileExistsError):
            acquire(self.plan, output, self.cache())

    def test_missing_chapter_never_publishes_partial_collection(self):
        plan = read_object(self.plan)
        plan["books"][1]["chapters"] = 2
        self.plan.write_text(json.dumps(plan))
        output = self.root / "corpus"
        with self.assertRaisesRegex(ValueError, "chapter inventory"):
            acquire(self.plan, output, self.cache())
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".corpus-*")))

    def test_navigation_only_page_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "insufficient Bangla"):
            extract_chapter(
                '<div class="ld-tab-content entry-content"><form>Login</form></div>',
                "chapter",
            )

    def test_duplicate_chapter_bodies_reject_the_entire_collection(self):
        plan = read_object(self.plan)
        plan["books"][0]["chapters"] = 2
        self.plan.write_text(json.dumps(plan))
        book_url = "https://www.ebanglalibrary.com/books/1/"
        second = "https://www.ebanglalibrary.com/lessons/repeated/"
        self.pages[book_url] = self.pages[book_url].replace(
            "</div></body>",
            f'<a class="ld-item-name" href="{second}">দ্বিতীয় অধ্যায়</a></div></body>',
        )
        self.pages[second] = self.pages["https://www.ebanglalibrary.com/lessons/1/"]
        with self.assertRaisesRegex(ValueError, "duplicate chapter body"):
            acquire(self.plan, self.root / "corpus", self.cache())
        self.assertFalse((self.root / "corpus").exists())


if __name__ == "__main__":
    unittest.main()
