"""Build a punctuation-preserving, work-partitioned literary text corpus.

Raw chapter receipts remain immutable. New work/author holdouts are fixed by an
admission plan before training. Short replies, joiners, punctuation and paragraph
breaks survive normalization. Quarantine whole windows sharing long passages
across splits; never remove common one-word replies merely for being common.
"""

from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
import zipfile
from xml.etree import ElementTree

from tools.corpus.partition import assigned_split, group_key
from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.autosuggest.build_sentence_dataset import epub_spine_member_names, stable_id


def normalize_text(text):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(
        c for c in text if c == "\n" or c == "\t" or unicodedata.category(c) != "Cc"
    )
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def author_members(label, aliases):
    """Resolve reviewed aliases per contributor, including translated works."""
    aliases = {normalize_text(k): normalize_text(v) for k, v in aliases.items()}
    members = sorted(
        {
            aliases.get(normalize_text(x), normalize_text(x))
            for x in label.split("|")
            if normalize_text(x)
        }
    )
    if not members or any(x.lower() == "unknown" for x in members):
        raise ValueError("missing author attribution")
    return members


def text_windows(text, maximum=8192):
    """Adjacent, non-overlapping windows; break at whitespace, never mid-word."""
    text = normalize_text(text)
    while len(text) > maximum:
        cut = text.rfind("\n", maximum // 2, maximum + 1)
        if cut < 0:
            cut = text.rfind(" ", maximum // 2, maximum + 1)
        if cut < 0:
            raise ValueError("unbounded whitespace-free text span")
        yield text[:cut]
        text = text[cut:].lstrip()
    if text:
        yield text


def passage_keys(text):
    # 32 lexical words, stride 16, catches punctuation/format changes and shared
    # long passages across editions. It is not a semantic near-duplicate test.
    words = []
    word = []
    for char in unicodedata.normalize("NFC", text):
        if unicodedata.category(char)[0] in "LMN" or char in "\u200c\u200d":
            word.append(char)
        elif word:
            words.append("".join(word))
            word = []
    if word:
        words.append("".join(word))
    starts = list(range(0, max(0, len(words) - 31), 16))
    if len(words) >= 32:
        starts.append(len(words) - 32)
    return sorted(
        {
            hashlib.sha256(" ".join(words[i : i + 32]).encode()).digest()[:16]
            for i in starts
        }
    )


def html_text(markup):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup, "html.parser")
    for node in soup.select("script,style,nav,noscript,svg"):
        node.decompose()
    for node in soup.find_all("br"):
        node.replace_with("\n")
    for node in soup.find_all(
        ["p", "div", "li", "h1", "h2", "h3", "h4", "blockquote", "tr"]
    ):
        node.insert_before("\n")
        node.append("\n")
    return normalize_text(soup.get_text())


def epub_sections(path):
    with zipfile.ZipFile(path) as archive:
        authors = []
        for name in archive.namelist():
            if name.endswith(".opf"):
                xml = ElementTree.fromstring(archive.read(name))
                authors += [
                    normalize_text(e.text or "")
                    for e in xml.iter()
                    if e.tag.rsplit("}", 1)[-1] == "creator" and e.text
                ]
        authors = sorted(set(authors))
        for name in epub_spine_member_names(archive):
            raw = archive.read(name).decode("utf-8")
            yield name, html_text(raw), authors


def build(args):
    if args.output.exists():
        raise ValueError("refusing to overwrite corpus")
    old = json.loads(args.original_plan.read_text(encoding="utf-8"))
    new = (
        json.loads(args.admission_plan.read_text(encoding="utf-8"))
        if args.admission_plan
        else {"books": []}
    )
    attribution = (
        json.loads(args.attribution.read_text(encoding="utf-8"))
        if args.attribution
        else {}
    )
    aliases = attribution.get("aliases", {})
    held = {
        person: role
        for name, role in new.get("author_holdouts", {}).items()
        for person in author_members(name, aliases)
    }
    books = {str(b["course_id"]): b for b in old["books"]}
    for b in new["books"]:
        key = str(b["course_id"])
        if key in books and books[key]["split"] != b["split"]:
            raise ValueError("cannot reassign an existing held-out work")
        books[key] = b
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    db = sqlite3.connect(args.output / "staging.sqlite")
    db.executescript(
        "CREATE TABLE records(id INTEGER PRIMARY KEY, split TEXT, author TEXT, source TEXT, work TEXT, reference TEXT, ordinal INTEGER, text TEXT, digest TEXT); CREATE TABLE anchors(key BLOB, record INTEGER, split TEXT);"
    )
    inputs = []
    work_metadata = {}
    seen_chapters = set()
    counts = Counter()

    def admit(source, work, author, role, reference, text):
        if role not in ("train", "validation", "test"):
            raise ValueError("invalid split")
        members = author_members(author, aliases)
        if any(person in held and held[person] != role for person in members):
            raise ValueError(f"author holdout contamination: {work}: {members}")
        author = " | ".join(members)
        metadata = {"source": source, "author": author, "split": role}
        if work in work_metadata and work_metadata[work] != metadata:
            raise ValueError("inconsistent work metadata")
        work_metadata[work] = metadata
        for ordinal, chunk in enumerate(text_windows(text)):
            if not any("\u0980" <= c <= "\u09ff" for c in chunk):
                continue
            digest = hashlib.sha256(chunk.encode()).hexdigest()
            row = db.execute(
                "INSERT INTO records(split,author,source,work,reference,ordinal,text,digest) VALUES(?,?,?,?,?,?,?,?)",
                (role, author, source, work, reference, ordinal, chunk, digest),
            ).lastrowid
            db.executemany(
                "INSERT INTO anchors VALUES(?,?,?)",
                [(key, row, role) for key in passage_keys(chunk)],
            )
            counts["input_windows"] += 1

    for raw_root in args.raw_books:
        for path in sorted(raw_root.glob("*.jsonl")):
            course = path.stem
            if course not in books:
                raise ValueError(f"book absent from admission plan: {course}")
            book = books[course]
            inputs.append(
                {
                    "source": "ebanglalibrary/books/" + path.name,
                    "sha256": sha256_file(path),
                }
            )
            # Original acquisition plans predate the explicit author field.
            author = normalize_text(
                book.get("author") or re.split(" – | — ", book["title"])[-1]
            )
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                key = (course, row["index"])
                if key in seen_chapters:
                    raise ValueError("duplicate chapter input")
                seen_chapters.add(key)
                if (
                    row["text_sha256"]
                    != hashlib.sha256(row["text"].encode()).hexdigest()
                ):
                    raise ValueError("raw chapter digest mismatch")
                admit(
                    "ebanglalibrary",
                    book["group_id"],
                    author,
                    book["split"],
                    row["url"],
                    row["text"],
                )
    for path in sorted(args.epubs.glob("*.epub")):
        if path.name in attribution.get("excluded_epubs", {}):
            counts["excluded_epubs"] += 1
            continue
        reference = "epubs/" + path.name
        document = stable_id("epub", reference)
        role = ("train", "validation", "test")[
            assigned_split(group_key("epub", document), "obadh-v1", 9000, 500)
        ]
        inputs.append({"source": reference, "sha256": sha256_file(path)})
        for section, text, authors in epub_sections(path):
            if not authors:
                raise ValueError(
                    f"EPUB author metadata missing: {path}; supply reviewed attribution"
                )
            admit(
                "epub",
                document,
                " | ".join(authors),
                role,
                reference + "#" + section,
                text,
            )
    db.commit()
    db.execute("CREATE INDEX anchor_keys ON anchors(key)")
    db.executescript(
        "CREATE TEMP TABLE bad AS SELECT DISTINCT record FROM anchors WHERE key IN (SELECT key FROM anchors GROUP BY key HAVING COUNT(DISTINCT split)>1); CREATE INDEX bad_records ON bad(record);"
    )
    # Common standalone short replies are not evidence of a duplicated work.
    db.execute(
        "INSERT INTO bad SELECT id FROM records WHERE LENGTH(text)>=80 AND digest IN (SELECT digest FROM records GROUP BY digest HAVING COUNT(DISTINCT split)>1)"
    )
    counts["quarantined_windows"] = db.execute(
        "SELECT COUNT(DISTINCT record) FROM bad"
    ).fetchone()[0]
    files = []
    stats = {}
    seen = set()
    for split in ("train", "validation", "test"):
        path = args.output / (split + ".jsonl.gz")
        stats[split] = {
            "windows": 0,
            "characters": 0,
            "authors": Counter(),
            "works": set(),
        }
        with (
            path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out,
        ):
            for row in db.execute(
                "SELECT id,author,source,work,reference,ordinal,text,digest FROM records WHERE split=? AND id NOT IN(SELECT record FROM bad) ORDER BY source,work,id",
                (split,),
            ):
                index, author, source, work, reference, ordinal, text, digest = row
                if (split, digest) in seen:
                    counts["same_split_duplicate_windows"] += 1
                    continue
                seen.add((split, digest))
                record = {
                    "source": source,
                    "work_id": work,
                    "author": author,
                    "authors": author_members(author, aliases),
                    "reference": reference,
                    "window": ordinal,
                    "split": split,
                    "text": text,
                    "sha256": digest,
                }
                out.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
                s = stats[split]
                s["windows"] += 1
                s["characters"] += len(text)
                s["authors"][author] += len(text)
                s["works"].add(work)
        stats[split]["authors"] = dict(stats[split]["authors"])
        stats[split]["works"] = sorted(stats[split]["works"])
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    db.close()
    (args.output / "staging.sqlite").unlink()
    manifest = {
        "version": 1,
        "kind": "obadh-contextual-text",
        "files": files,
        "inputs": inputs,
        "statistics": stats,
        "counts": dict(counts),
        "works": work_metadata,
        "policy": {
            "normalization": "NFC, LF newlines, horizontal whitespace collapsed; paragraph breaks and all scripts/punctuation retained",
            "short_replies": "retained",
            "max_window_characters": 8192,
            "cross_split_quarantine": "exact windows of at least 80 characters and any shared 32-lexical-word anchor at stride 16; not semantic duplicate detection",
            "speaker_labels": "none inferred",
            "split": "original work assignments preserved; new admission plan fixed before training",
        },
        "original_plan_sha256": sha256_file(args.original_plan),
        "builder_sha256": sha256_file(Path(__file__)),
        "unicode_version": unicodedata.unidata_version,
        "admission_plan_sha256": sha256_file(args.admission_plan)
        if args.admission_plan
        else None,
        "attribution": attribution,
    }
    manifest["dataset_id"] = digest_json(manifest)
    write_json(args.output / "manifest.json", manifest)
    (args.output / ".building").unlink()
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "counts": dict(counts),
                "statistics": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-books", type=Path, action="append", required=True)
    p.add_argument("--epubs", type=Path, required=True)
    p.add_argument("--original-plan", type=Path, required=True)
    p.add_argument("--admission-plan", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attribution", type=Path)
    build(p.parse_args())


if __name__ == "__main__":
    main()
