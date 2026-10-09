"""Acquire a declared eBanglaLibrary collection using epub-exporter's parser.

Install the optional local exporter package for this command only. Network access
is bounded, serial, cached by content hash, and never reads account credentials.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from tools.autosuggest.build_sentence_dataset import bangla_tokens, sentence_spans
from tools.corpus.partition import gzip_writer
from tools.corpus.provenance import (
    SPLITS,
    digest_json,
    read_object,
    sha256_file,
    write_json,
)

SOURCE = "ebanglalibrary"
MAX_PAGE_BYTES = 5 * 1024 * 1024
USER_AGENT = "ObadhCorpus/1.0 (bounded book acquisition; one request per second)"


def check_page_kind(requested: str, final: str) -> None:
    final = canonical_url(final)
    if urlsplit(requested).path.split("/")[1] != urlsplit(final).path.split("/")[1]:
        raise ValueError(
            "chapter/book redirect changed page kind; refusing front matter as chapter text"
        )


def canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc not in ("www.ebanglalibrary.com", "ebanglalibrary.com")
        or parsed.query
        or not parsed.path.startswith(("/books/", "/lessons/", "/authors/", "/genres/"))
    ):
        raise ValueError(f"unsupported acquisition URL: {url}")
    decoded = unquote(parsed.path)
    if (
        any(part in (".", "..") for part in decoded.split("/"))
        or "\\" in decoded
        or any(unicodedata.category(c) == "Cc" for c in decoded)
    ):
        raise ValueError("invalid acquisition URL path")
    path = quote(decoded, safe="/-._~")
    return urlunsplit(("https", "www.ebanglalibrary.com", path, "", ""))


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".writing-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SameSiteRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return super().redirect_request(
            request, fp, code, message, headers, canonical_url(new_url)
        )


class PageCache:
    def __init__(self, root: Path, *, max_pages: int = 300, offline: bool = False):
        self.root = root
        self.max_pages = max_pages
        self.offline = offline
        self.pages = 0
        self.last_request = 0.0
        self.opener = build_opener(SameSiteRedirect())

    def fetch(self, url: str) -> tuple[str, dict]:
        url = canonical_url(url)
        self.pages += 1
        if self.pages > self.max_pages:
            raise ValueError("declared page budget exceeded")
        index = (
            self.root
            / "receipts"
            / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        )
        if index.exists():
            receipt = read_object(index)
            digest = receipt.get("sha256", "")
            if (
                receipt.get("url") != url
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"invalid cached receipt: {index}")
            check_page_kind(url, receipt["final_url"])
            data = (self.root / "blobs" / f"{digest}.html").read_bytes()
            if (
                len(data) != receipt["bytes"]
                or hashlib.sha256(data).hexdigest() != digest
            ):
                raise ValueError(f"cached page digest mismatch: {url}")
            return data.decode(receipt["charset"]), receipt
        if self.offline:
            raise ValueError(f"page absent from offline cache: {url}")
        for attempt in range(3):
            time.sleep(max(0.0, 1.0 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                request = Request(
                    url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"}
                )
                with self.opener.open(request, timeout=30) as response:
                    final_url = canonical_url(response.url)
                    check_page_kind(url, final_url)
                    if response.headers.get_content_type() != "text/html":
                        raise ValueError(f"expected an HTML page: {url}")
                    data = response.read(MAX_PAGE_BYTES + 1)
                    charset = response.headers.get_content_charset() or "utf-8"
                if len(data) > MAX_PAGE_BYTES:
                    raise ValueError(f"page exceeds size bound: {url}")
                text = data.decode(
                    charset
                )  # No silent replacement of broken source bytes.
                receipt = {
                    "url": url,
                    "final_url": final_url,
                    "bytes": len(data),
                    "charset": charset,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_bytes(self.root / "blobs" / f"{receipt['sha256']}.html", data)
                atomic_bytes(
                    index, (json.dumps(receipt, sort_keys=True) + "\n").encode()
                )
                return text, receipt
            except (HTTPError, URLError, TimeoutError) as error:
                if isinstance(error, HTTPError) and error.code not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
                if attempt == 2:
                    raise
                delay = 2 ** (attempt + 1)
                if isinstance(error, HTTPError):
                    retry_after = error.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = min(60, max(delay, int(retry_after)))
                time.sleep(delay)
        raise AssertionError("unreachable retry state")


def extract_book(markup: str, url: str) -> dict:
    from bs4 import BeautifulSoup
    from wp_learndash_epub.source import (
        book_title,
        book_authors,
        course_id,
        lesson_links,
    )

    soup = BeautifulSoup(markup, "html.parser")
    lessons = lesson_links(soup, url)
    if not lessons or len(lessons) > 100:
        raise ValueError("book must expose between 1 and 100 ordered chapters")
    # This importer supports complete single-page tables of contents only.
    # Never silently publish the first page of a paginated book.
    for node in soup.select(".ld-pagination"):
        if node.select("a[href]") or "2" in node.get_text(" ", strip=True):
            raise ValueError(
                "paginated table of contents needs explicit support before acquisition"
            )
    return {
        "title": book_title(soup),
        "authors": book_authors(soup),
        "course_id": course_id(soup),
        "lessons": [
            {"title": item.title, "url": canonical_url(item.url)} for item in lessons
        ],
    }


def extract_chapter(markup: str, fallback_title: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup
    from wp_learndash_epub.source import extract_lesson_content, lesson_title
    from wp_learndash_epub.html_clean import sanitize_fragment

    soup = BeautifulSoup(markup, "html.parser")
    title = lesson_title(soup, fallback_title)
    content = extract_lesson_content(soup)
    cleaned = BeautifulSoup(
        sanitize_fragment(BeautifulSoup(str(content), "html.parser")), "html.parser"
    )
    for node in cleaned.find_all("br"):
        node.replace_with("\n")
    for node in cleaned.find_all(
        ["p", "div", "li", "h1", "h2", "h3", "h4", "blockquote", "tr"]
    ):
        node.insert_before("\n")
        node.append("\n")
    text = "\n".join(
        line
        for part in cleaned.get_text().splitlines()
        if (line := " ".join(part.split()))
    )
    if sum("\u0980" <= char <= "\u09ff" for char in text) < 50:
        raise ValueError(
            "chapter has insufficient Bangla body text; refusing an incomplete book"
        )
    return title, text


def validate_plan(plan: dict) -> list[dict]:
    if (
        plan.get("artifact") != "obadh-book-acquisition-plan"
        or plan.get("version") != 1
    ):
        raise ValueError("unsupported book acquisition plan")
    if not plan.get("authorization"):
        raise ValueError(
            "acquisition plan must record the project owner's authorization"
        )
    books = plan.get("books")
    if not isinstance(books, list) or not 1 <= len(books) <= 100:
        raise ValueError("plan must declare between 1 and 100 books")
    ids, urls, groups = set(), set(), {}
    for book in books:
        url = canonical_url(book["url"])
        if not urlsplit(url).path.startswith("/books/"):
            raise ValueError("plan entries must be book URLs")
        course = book["course_id"]
        role = book["split"]
        group = book["group_id"]
        if (
            type(course) is not int
            or course < 1
            or course in ids
            or url in urls
            or role not in SPLITS
            or not isinstance(group, str)
            or not group
            or any(unicodedata.category(char) == "Cc" for char in group)
            or type(book["chapters"]) is not int
            or not 1 <= book["chapters"] <= 100
        ):
            raise ValueError("invalid or duplicate book declaration")
        if group in groups and groups[group] != role:
            raise ValueError("related editions cannot have different split assignments")
        ids.add(course)
        urls.add(url)
        groups[group] = role
    return books


def acquire(
    plan_path: Path, output: Path, cache: PageCache, *, base_corpus: Path | None = None
) -> dict:
    from wp_learndash_epub import source, html_clean
    from tools.autosuggest import build_sentence_dataset

    plan_digest = sha256_file(plan_path)
    plan = read_object(plan_path)
    books = validate_plan(plan)
    if sum(1 + book["chapters"] for book in books) > cache.max_pages:
        raise ValueError("page budget is smaller than the declared collection")
    implementation = {
        "acquirer": Path(__file__),
        "source": Path(source.__file__),
        "cleaner": Path(html_clean.__file__),
        "tokenizer": Path(build_sentence_dataset.__file__),
    }
    implementation_digests = {
        name: sha256_file(path) for name, path in implementation.items()
    }
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (stage / "sentences").mkdir()
        (stage / "books").mkdir()
        base_inputs = []
        if base_corpus is not None:
            paths = sorted((base_corpus / "sentences").glob("*.tsv.gz"))
            if not paths:
                raise ValueError("base corpus has no sentence shards")
            for path in paths:
                if path.name == f"{SOURCE}.tsv.gz":
                    raise ValueError("base corpus already has an eBanglaLibrary shard")
                digest = sha256_file(path)
                target = stage / "sentences" / path.name
                shutil.copyfile(path, target)
                if sha256_file(target) != digest or sha256_file(path) != digest:
                    raise ValueError("base corpus changed while copying")
                base_inputs.append(
                    {
                        "path": f"sentences/{path.name}",
                        "sha256": digest,
                        "bytes": target.stat().st_size,
                    }
                )
        records = []
        seen_lessons: dict[str, str] = {}
        with gzip_writer(stage / "sentences" / f"{SOURCE}.tsv.gz") as writer:
            for declaration in books:
                url = canonical_url(declaration["url"])
                markup, receipt = cache.fetch(url)
                book = extract_book(markup, url)
                if (
                    book["course_id"] != declaration["course_id"]
                    or len(book["lessons"]) != declaration["chapters"]
                ):
                    raise ValueError(
                        f"course identity/chapter inventory differs from acquisition plan: {url}"
                    )
                document_id = f"ebanglalibrary:course:{book['course_id']}"
                group = declaration["group_id"]
                statistics = {
                    "sentences": 0,
                    "tokens": 0,
                    "skipped_short_spans": 0,
                    "long_span_chunks": 0,
                }
                chapters = []
                chapter_text_hashes: set[str] = set()
                book_path = stage / "books" / f"{book['course_id']}.jsonl"
                with book_path.open("w", encoding="utf-8", newline="\n") as body:
                    for index, lesson in enumerate(book["lessons"], 1):
                        chapter_url = lesson["url"]
                        if (
                            chapter_url in seen_lessons
                            and seen_lessons[chapter_url] != group
                        ):
                            raise ValueError(
                                "chapter occurs in independently grouped books; declare a shared work group"
                            )
                        seen_lessons[chapter_url] = group
                        markup, chapter_receipt = cache.fetch(chapter_url)
                        title, chapter_text = extract_chapter(markup, lesson["title"])
                        chapter_digest = hashlib.sha256(
                            chapter_text.encode()
                        ).hexdigest()
                        if chapter_digest in chapter_text_hashes:
                            raise ValueError(
                                f"duplicate chapter body within book; refusing incomplete or redirected content: {url}"
                            )
                        chapter_text_hashes.add(chapter_digest)
                        first_sentence = statistics["sentences"] + 1
                        for span in sentence_spans(chapter_text):
                            tokens = bangla_tokens(span)
                            if len(tokens) < 2:
                                statistics["skipped_short_spans"] += 1
                                continue
                            for offset in range(0, len(tokens), 80):
                                chunk = tokens[offset : offset + 80]
                                if len(chunk) < 2:
                                    statistics["skipped_short_spans"] += 1
                                    continue
                                statistics["long_span_chunks"] += int(offset > 0)
                                statistics["sentences"] += 1
                                statistics["tokens"] += len(chunk)
                                writer.writerow(
                                    (
                                        SOURCE,
                                        document_id,
                                        statistics["sentences"],
                                        len(chunk),
                                        " ".join(chunk),
                                        group,
                                    )
                                )
                        if statistics["sentences"] < first_sentence:
                            raise ValueError(
                                f"chapter produced no usable sentences: {chapter_url}"
                            )
                        chapter = {
                            "index": index,
                            "title": title,
                            "url": chapter_url,
                            "source_receipt": chapter_receipt,
                            "text_sha256": chapter_digest,
                            "first_sentence_id": first_sentence,
                            "last_sentence_id": statistics["sentences"],
                        }
                        body.write(
                            json.dumps(
                                {**chapter, "text": chapter_text}, ensure_ascii=False
                            )
                            + "\n"
                        )
                        chapters.append(chapter)
                        print(
                            json.dumps(
                                {
                                    "event": "book_chapter_acquired",
                                    "course_id": book["course_id"],
                                    "chapter": index,
                                    "chapters": len(book["lessons"]),
                                }
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                records.append(
                    {
                        "document_id": document_id,
                        "group_id": group,
                        "split": declaration["split"],
                        "title": book["title"],
                        "authors": book["authors"],
                        "url": url,
                        "source_receipt": receipt,
                        "chapters": chapters,
                        "statistics": statistics,
                        "text_file": f"books/{book_path.name}",
                        "text_file_sha256": sha256_file(book_path),
                    }
                )
        assignments = {book["group_id"]: book["split"] for book in books}
        if sha256_file(plan_path) != plan_digest or any(
            sha256_file(path) != implementation_digests[name]
            for name, path in implementation.items()
        ):
            raise ValueError(
                "acquisition plan or extraction implementation changed during acquisition"
            )
        write_json(stage / "group-assignments.json", assignments)
        manifest = {
            "artifact": "obadh-acquired-book-corpus",
            "version": 1,
            "plan_sha256": plan_digest,
            "authorization": plan["authorization"],
            "source": SOURCE,
            "extractor": {
                "package": "wp-learndash-epub",
                "version": importlib.metadata.version("wp-learndash-epub"),
                "acquirer_sha256": implementation_digests["acquirer"],
                "source_sha256": sha256_file(Path(source.__file__)),
                "cleaner_sha256": sha256_file(Path(html_clean.__file__)),
            },
            "tokenizer": {
                "module": "tools.autosuggest.build_sentence_dataset",
                "sha256": sha256_file(Path(build_sentence_dataset.__file__)),
                "policy": "existing Bangla-only token view, 2..80 tokens; original chapter text retained",
            },
            "base_inputs": base_inputs,
            "base_manifest_sha256": sha256_file(base_corpus / "manifest.json")
            if base_corpus is not None
            else None,
            "books": records,
            "group_assignments": assignments,
            "sentence_files": [
                {
                    "path": f"sentences/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted((stage / "sentences").glob("*.tsv.gz"))
            ],
        }
        manifest["acquisition_id"] = digest_json(manifest)
        write_json(stage / "manifest.json", manifest)
        output.mkdir()  # Exclusive publication guard; manifest moved last.
        try:
            (output / ".building").touch()
            for name in (
                "sentences",
                "books",
                "group-assignments.json",
                "manifest.json",
            ):
                (stage / name).rename(output / name)
            (output / ".building").unlink()
        except BaseException:
            shutil.rmtree(output)
            raise
        return manifest
    finally:
        shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--base-corpus", type=Path)
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        result = acquire(
            args.plan,
            args.output,
            PageCache(args.cache, max_pages=args.max_pages, offline=args.offline),
            base_corpus=args.base_corpus,
        )
        print(
            json.dumps(
                {
                    "acquisition_id": result["acquisition_id"],
                    "books": len(result["books"]),
                    "sentences": sum(
                        book["statistics"]["sentences"] for book in result["books"]
                    ),
                    "tokens": sum(
                        book["statistics"]["tokens"] for book in result["books"]
                    ),
                },
                indent=2,
            )
        )
    except (
        ValueError,
        OSError,
        RuntimeError,
        KeyError,
        TypeError,
        ImportError,
    ) as error:
        parser.exit(2, f"book acquisition error: {error}\n")


if __name__ == "__main__":
    main()
