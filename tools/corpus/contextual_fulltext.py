"""Rebuild news/wiki/books from original text with stable work splits.

Preserves punctuation and paragraphs; stages on disk. Legacy news/wiki IDs must
match the original document inventory. Published correction benchmarks are
excluded conservatively by lexical spans, without using them as training labels.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import unicodedata

from tools.corpus.contextual_books import normalize_text, text_windows, passage_keys
from tools.corpus.partition import assigned_split, group_key
from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.autosuggest.build_sentence_dataset import (
    iter_news_documents,
    iter_wiki_documents,
)


def lexical_words(text):
    result = []
    word = []
    for c in unicodedata.normalize("NFC", text):
        if unicodedata.category(c)[0] in "LMN" or c in "\u200c\u200d":
            word.append(c)
        elif word:
            result.append("".join(word))
            word = []
    if word:
        result.append("".join(word))
    return result


def benchmark_patterns(directory):
    patterns = defaultdict(set)
    receipts = []
    for path in sorted(directory.rglob("*")):
        if path.suffix not in (".src", ".tgt"):
            continue
        receipts.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": sha256_file(path),
            }
        )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                words = lexical_words(line)
                n = min(9, len(words))
                if n < 4:
                    raise ValueError(
                        "benchmark sentence too short for safe span exclusion"
                    )
                patterns[n].update(
                    tuple(words[i : i + n]) for i in range(len(words) - n + 1)
                )
    if not receipts:
        raise ValueError("missing correction benchmark exclusions")
    return patterns, receipts


def overlaps_benchmark(text, patterns):
    words = lexical_words(text)
    return any(
        tuple(words[i : i + n]) in spans
        for n, spans in patterns.items()
        for i in range(len(words) - n + 1)
    )


def build(args):
    if args.output.exists():
        raise ValueError("refusing to overwrite full text corpus")
    patterns, exclusions = benchmark_patterns(args.exclude_benchmark)
    legacy = {}
    fallback = {}
    with gzip.open(args.document_inventory, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            source = row["source"]
            if source not in ("news", "wiki"):
                continue
            doc = row["document_id"]
            role = ("train", "validation", "test")[
                assigned_split(group_key(source, doc), "obadh-v1", 9000, 500)
            ]
            legacy[(source, doc)] = role
            if source == "news" and "#" in row["source_ref"]:
                suffix = row["source_ref"].rsplit("#", 1)[-1]
                if suffix.isdigit():
                    fallback[suffix] = doc
    book_manifest = json.loads(
        (args.books / "manifest.json").read_text(encoding="utf-8")
    )
    if (args.books / ".building").exists() or book_manifest[
        "dataset_id"
    ] != digest_json({k: v for k, v in book_manifest.items() if k != "dataset_id"}):
        raise ValueError("invalid book corpus")
    for item in book_manifest["files"]:
        if (
            item["path"]
            not in ("train.jsonl.gz", "validation.jsonl.gz", "test.jsonl.gz")
            or sha256_file(args.books / item["path"]) != item["sha256"]
        ):
            raise ValueError("book integrity failure")
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    recovery = None
    recover_stage = getattr(args, "recover_stage", None)
    if recover_stage:
        if (
            not (recover_stage / ".building").exists()
            or (recover_stage / "manifest.json").exists()
        ):
            raise ValueError("recovery requires an incomplete ingestion stage")
        with closing(
            sqlite3.connect(
                f"file:{recover_stage / 'staging.sqlite'}?mode=ro", uri=True
            )
        ) as source_db:
            tables = {
                r[0]
                for r in source_db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if tables != {"records", "anchors", "excluded_works", "documents"}:
                raise ValueError(
                    "recovery only supports the original ingestion-stage schema"
                )
            if source_db.execute(
                "SELECT COUNT(*) FROM records WHERE source NOT IN ('news','wiki')"
            ).fetchone()[0]:
                raise ValueError("recovery must precede book ingestion")
            with closing(sqlite3.connect(args.output / "staging.sqlite")) as target_db:
                source_db.backup(target_db)
        recovery = {
            "copied_staging_sha256": sha256_file(args.output / "staging.sqlite"),
            "policy": "reuse staged windows; replay and verify every original normalized document, recompute benchmark exclusions, preserve original stage for audit",
        }
    db = sqlite3.connect(args.output / "staging.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    if not recovery:
        db.executescript(
            "CREATE TABLE records(id INTEGER PRIMARY KEY, source TEXT, work TEXT, split TEXT, reference TEXT, ordinal INTEGER, authors TEXT, text TEXT, digest TEXT); CREATE TABLE anchors(key BLOB, record INTEGER, split TEXT); CREATE TABLE excluded_works(work TEXT PRIMARY KEY); CREATE TABLE documents(source TEXT,work TEXT,digest TEXT,PRIMARY KEY(source,work));"
        )
    db.execute(
        "CREATE TABLE quality_excluded_works(work TEXT PRIMARY KEY,reference TEXT,reason TEXT)"
    )
    recovered_documents = set(db.execute("SELECT source,work FROM documents"))
    for source, work, role in db.execute(
        "SELECT DISTINCT source,work,split FROM records"
    ):
        if legacy.get((source, work)) != role:
            raise ValueError("recovered document split mismatch")
    db.execute("DELETE FROM excluded_works")
    counts = Counter()
    counts["input_windows"] = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    if recovery:
        recovery.update(
            documents=len(recovered_documents), windows=counts["input_windows"]
        )
    started = time.monotonic()
    inputs = []

    def admit(source, work, role, reference, ordinal, authors, text):
        text = normalize_text(text)
        if not any("\u0980" <= c <= "\u09ff" for c in text):
            counts["no_bangla_windows"] += 1
            return
        if overlaps_benchmark(text, patterns):
            db.execute("INSERT OR IGNORE INTO excluded_works VALUES(?)", (work,))
            counts["benchmark_matching_windows"] += 1
        digest = hashlib.sha256(text.encode()).hexdigest()
        record = db.execute(
            "INSERT INTO records(source,work,split,reference,ordinal,authors,text,digest) VALUES(?,?,?,?,?,?,?,?)",
            (
                source,
                work,
                role,
                reference,
                ordinal,
                json.dumps(authors, ensure_ascii=False),
                text,
                digest,
            ),
        ).lastrowid
        db.executemany(
            "INSERT INTO anchors VALUES(?,?,?)",
            [(key, record, role) for key in passage_keys(text)],
        )
        counts["input_windows"] += 1

    for source, documents in [
        ("news", iter_news_documents(args.news_json, args.limit)),
        ("wiki", iter_wiki_documents(args.wiki_dir)),
    ]:
        source_count = 0
        input_digest = hashlib.sha256()
        seen_documents = set()
        for doc, reference, title, blocks, _ in documents:
            if args.limit and source_count >= args.limit:
                break
            if (source, doc) not in legacy and source == "news" and "#" in reference:
                doc = fallback.get(reference.rsplit("#", 1)[-1], doc)
            if (source, doc) not in legacy:
                raise ValueError(
                    f"document outside original split inventory: {source}/{doc}"
                )
            text = normalize_text("\n\n".join(blocks))
            digest = hashlib.sha256(text.encode()).hexdigest()
            previous = db.execute(
                "SELECT digest FROM documents WHERE source=? AND work=?", (source, doc)
            ).fetchone()
            if previous:
                if previous[0] != digest:
                    raise ValueError(
                        f"conflicting text under one document identity: {doc}"
                    )
                if doc in seen_documents:
                    counts["duplicate_documents"] += 1
                    continue
            else:
                db.execute("INSERT INTO documents VALUES(?,?,?)", (source, doc, digest))
            seen_documents.add(doc)
            recovered_documents.discard((source, doc))
            input_digest.update(
                digest_json([doc, title, blocks]).encode("ascii") + b"\n"
            )
            if not reference.startswith(("http://", "https://")):
                reference = source + ":" + Path(reference).name
            source_count += 1
            counts[source + "_documents"] += 1
            if overlaps_benchmark(text, patterns):
                db.execute("INSERT OR IGNORE INTO excluded_works VALUES(?)", (doc,))
            if not previous:
                try:
                    chunks = list(text_windows(text))
                except ValueError as error:
                    db.execute(
                        "INSERT INTO quality_excluded_works VALUES(?,?,?)",
                        (doc, reference, str(error)),
                    )
                    chunks = []
                    print(
                        json.dumps(
                            dict(
                                event="quarantined_work",
                                work=doc,
                                reference=reference,
                                reason=str(error),
                            )
                        ),
                        flush=True,
                    )
                for ordinal, chunk in enumerate(chunks):
                    admit(
                        source,
                        doc,
                        legacy[(source, doc)],
                        reference,
                        ordinal,
                        [],
                        chunk,
                    )
            if source_count % 1000 == 0:
                db.commit()
                print(
                    json.dumps(
                        {
                            "source": source,
                            "documents": source_count,
                            "windows": counts["input_windows"],
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
        inputs.append(
            {
                "source": source,
                "documents": source_count,
                "document_stream_sha256": input_digest.hexdigest(),
                "document_stream_format": "ordered ASCII digest_json([id,title,raw_blocks]) plus LF per unique document",
            }
        )
    if recovered_documents:
        raise ValueError("recovered documents missing from replayed sources")
    for split in ("train", "validation", "test"):
        with gzip.open(
            args.books / (split + ".jsonl.gz"), "rt", encoding="utf-8"
        ) as rows:
            for line in rows:
                row = json.loads(line)
                if (
                    row["split"] != split
                    or hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]
                ):
                    raise ValueError("invalid book record")
                admit(
                    row["source"],
                    row["work_id"],
                    split,
                    row["reference"],
                    row["window"],
                    row["authors"],
                    row["text"],
                )
    db.commit()
    print(json.dumps({"event": "indexing", "counts": dict(counts)}), flush=True)
    db.execute("CREATE INDEX anchor_keys ON anchors(key)")
    db.executescript(
        "CREATE TEMP TABLE bad AS SELECT DISTINCT record FROM anchors WHERE key IN (SELECT key FROM anchors GROUP BY key HAVING COUNT(DISTINCT split)>1); CREATE INDEX bad_records ON bad(record); CREATE INDEX record_digests ON records(digest);"
    )
    db.execute(
        "INSERT INTO bad SELECT id FROM records WHERE LENGTH(text)>=80 AND digest IN (SELECT digest FROM records GROUP BY digest HAVING COUNT(DISTINCT split)>1)"
    )
    counts["benchmark_excluded_works"] = db.execute(
        "SELECT COUNT(*) FROM excluded_works"
    ).fetchone()[0]
    quality_exclusions = [
        dict(work_id=w, reference=r, reason=e)
        for w, r, e in db.execute(
            "SELECT work,reference,reason FROM quality_excluded_works ORDER BY work"
        )
    ]
    counts["quality_excluded_works"] = len(quality_exclusions)
    counts["cross_split_quarantined_windows"] = db.execute(
        "SELECT COUNT(DISTINCT record) FROM bad"
    ).fetchone()[0]
    db.execute("CREATE INDEX work_ids ON records(work)")
    db.execute("CREATE TABLE seen(split TEXT,digest TEXT,PRIMARY KEY(split,digest))")
    files = []
    statistics = {}
    for split in ("train", "validation", "test"):
        path = args.output / (split + ".jsonl.gz")
        stats = defaultdict(Counter)
        with (
            path.open("wb") as raw,
            gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6
            ) as output,
        ):
            for source, work, reference, ordinal, authors, text, digest in db.execute(
                "SELECT source,work,reference,ordinal,authors,text,digest FROM records WHERE split=? AND id NOT IN(SELECT record FROM bad) AND work NOT IN(SELECT work FROM excluded_works) AND work NOT IN(SELECT work FROM quality_excluded_works) ORDER BY source,work,reference,ordinal,id",
                (split,),
            ):
                cursor = db.execute(
                    "INSERT OR IGNORE INTO seen VALUES(?,?)", (split, digest)
                )
                if not cursor.rowcount:
                    counts["same_split_duplicate_windows"] += 1
                    continue
                row = {
                    "source": source,
                    "work_id": work,
                    "reference": reference,
                    "window": ordinal,
                    "authors": json.loads(authors),
                    "split": split,
                    "text": text,
                    "sha256": digest,
                }
                output.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
                stats[source]["windows"] += 1
                stats[source]["characters"] += len(text)
        statistics[split] = {s: dict(c) for s, c in stats.items()}
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    db.close()
    for suffix in ("", "-wal", "-shm"):
        (args.output / ("staging.sqlite" + suffix)).unlink(missing_ok=True)
    manifest = {
        "version": 1,
        "kind": "obadh-contextual-text",
        "files": files,
        "statistics": statistics,
        "counts": dict(counts),
        "legacy_document_inventory_sha256": sha256_file(args.document_inventory),
        "news_json_sha256": sha256_file(args.news_json),
        "book_corpus_id": book_manifest["dataset_id"],
        "benchmark_exclusions": exclusions,
        "quality_exclusions": quality_exclusions,
        "ingestion_recovery": recovery,
        "window_rejection_counter_scope": "newly staged windows; recovered windows are included in input_windows but not window-level rejection counters",
        "input_sources": inputs,
        "builder_sha256": sha256_file(Path(__file__)),
        "text_helpers_sha256": sha256_file(
            Path(__file__).with_name("contextual_books.py")
        ),
        "policy": {
            "split": "original obadh-v1 news/wiki document membership and explicit book roles preserved",
            "benchmark_exclusion": "whole work excluded on any normalized 9-word benchmark span, or whole shorter benchmark sentence; source and corrected forms, dev and test",
            "duplicates": "shared 32-word anchors across splits plus exact windows >=80 characters; semantic duplicates not detected",
            "normalization": "NFC, LF and horizontal whitespace; punctuation, paragraphs, short replies, scripts and joiners retained",
            "limit_per_formal_source": args.limit,
        },
    }
    manifest["dataset_id"] = digest_json(manifest)
    write_json(args.output / "manifest.json", manifest)
    (args.output / ".building").unlink()
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "counts": dict(counts),
                "statistics": statistics,
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "news-json",
        "wiki-dir",
        "document-inventory",
        "books",
        "exclude-benchmark",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--recover-stage", type=Path)
    build(p.parse_args())


if __name__ == "__main__":
    main()
