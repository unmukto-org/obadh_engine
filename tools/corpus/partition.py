"""Create reproducible train/validation/test corpora with bounded RAM.

Run from the engine root with ``python3 -m tools.corpus.partition build ...``.
SQLite holds rows and duplicate identities on disk. Whole document groups are
assigned by a seeded SHA-256 hash. Sentences appearing in multiple partitions
are quarantined from *every* partition, without changing document assignments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from tools.corpus.provenance import (
    PARTITION_KIND,
    PARTITIONS_KIND,
    SPLITS,
    VERSION,
    digest_json,
    load_partition,
    sha256_file,
    write_json,
)

FIELDS = ("source", "document_id", "sentence_id", "token_count", "tokens")
OUTPUT_FIELDS = (*FIELDS, "group_id")
SOURCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
RESERVED_SOURCES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def group_key(source: str, document_id: str, group_id: str = "") -> str:
    # Explicit groups are global across sources (translations, conversations,
    # authors, or clean sentences plus synthetic variants).
    return digest_json(
        ["group", group_id] if group_id else ["document", source, document_id]
    )


def assigned_split(key: str, seed: str, train_bps: int, validation_bps: int) -> int:
    # Basis points avoid platform floating-point differences. No Python hash().
    value = int(digest_json(["obadh-corpus-split-v1", seed, key]), 16)
    bucket = value * 10_000 // (1 << 256)
    return 0 if bucket < train_bps else 1 if bucket < train_bps + validation_bps else 2


@contextmanager
def gzip_writer(path: Path) -> Iterator[csv.writer]:
    # Exclude timestamps and filenames from gzip headers for byte reproducibility.
    with path.open("wb") as raw:
        with gzip.GzipFile(
            fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, delimiter="\t", lineterminator="\n")
                writer.writerow(OUTPUT_FIELDS)
                yield writer


def validated_rows(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        fields = reader.fieldnames or []
        if (
            len(fields) != len(set(fields))
            or not set(FIELDS).issubset(fields)
            or set(fields) - set(OUTPUT_FIELDS)
        ):
            raise ValueError(f"invalid corpus columns: {path}")
        for row in reader:
            location = f"{path}:{reader.line_num}"
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"invalid column count: {location}")
            source = row["source"]
            if (
                not SOURCE_PATTERN.fullmatch(source)
                or source.lower() in RESERVED_SOURCES
            ):
                raise ValueError(f"invalid source identifier: {location}")
            for name in ("document_id", "group_id"):
                value = row.get(name, "")
                if (name == "document_id" and not value) or any(
                    unicodedata.category(c) == "Cc" for c in value
                ):
                    raise ValueError(f"invalid {name}: {location}")
            try:
                sentence_id, token_count = (
                    int(row["sentence_id"]),
                    int(row["token_count"]),
                )
            except ValueError as error:
                raise ValueError(f"invalid numeric field: {location}") from error
            tokens = row["tokens"]
            if (
                not 0 <= sentence_id < (1 << 63)
                or token_count < 1
                or tokens != " ".join(tokens.split())
                or not tokens
            ):
                raise ValueError(
                    f"invalid sentence identity or token spacing: {location}"
                )
            if len(tokens.split(" ")) != token_count or any(
                unicodedata.category(c) == "Cc" for c in tokens
            ):
                raise ValueError(
                    f"invalid token count or control character: {location}"
                )
            row.update(
                sentence_id=sentence_id,
                token_count=token_count,
                group_id=row.get("group_id", ""),
            )
            yield row


def ingest(
    connection: sqlite3.Connection,
    paths: list[Path],
    seed: str,
    train_bps: int,
    validation_bps: int,
    group_assignments: dict[str, str],
) -> list[dict]:
    connection.executescript("""
        CREATE TABLE documents (
            source TEXT NOT NULL, document_id TEXT NOT NULL, group_key TEXT NOT NULL,
            PRIMARY KEY(source, document_id)
        ) WITHOUT ROWID;
        CREATE TABLE sentences (
            source TEXT NOT NULL, document_id TEXT NOT NULL, sentence_id INTEGER NOT NULL,
            token_count INTEGER NOT NULL, tokens TEXT NOT NULL, group_id TEXT NOT NULL,
            split INTEGER NOT NULL, content_hash BLOB NOT NULL,
            PRIMARY KEY(source, document_id, sentence_id)
        ) WITHOUT ROWID;
        CREATE TABLE contents (hash BLOB PRIMARY KEY, split_mask INTEGER NOT NULL) WITHOUT ROWID;
    """)

    @lru_cache(maxsize=8192)
    def register_document(source: str, document_id: str, key: str) -> None:
        # Changing the grouping of a document is invalid, even across shards.
        connection.execute(
            """
            INSERT INTO documents VALUES (?, ?, ?)
            ON CONFLICT(source, document_id) DO UPDATE SET group_key =
            CASE WHEN group_key = excluded.group_key THEN group_key ELSE NULL END
        """,
            (source, document_id, key),
        )

    used_assignments: set[str] = set()

    @lru_cache(maxsize=8192)
    def route(source: str, document_id: str, group_id: str) -> tuple[str, int]:
        key = group_key(source, document_id, group_id)
        if group_id in group_assignments:
            used_assignments.add(group_id)
            return key, SPLITS.index(group_assignments[group_id])
        return key, assigned_split(key, seed, train_bps, validation_bps)

    receipts = []
    count = 0
    for path in paths:
        before = sha256_file(path)
        for row in validated_rows(path):
            key, split = route(row["source"], row["document_id"], row["group_id"])
            content_hash = hashlib.sha256(
                unicodedata.normalize("NFC", row["tokens"]).encode("utf-8")
            ).digest()
            try:
                register_document(row["source"], row["document_id"], key)
                connection.execute(
                    "INSERT INTO sentences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["source"],
                        row["document_id"],
                        row["sentence_id"],
                        row["token_count"],
                        row["tokens"],
                        row["group_id"],
                        split,
                        content_hash,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    f"duplicate sentence ID or inconsistent document group in {path}: {row['source']}/{row['document_id']}/{row['sentence_id']}"
                ) from error
            connection.execute(
                """
                INSERT INTO contents VALUES (?, ?)
                ON CONFLICT(hash) DO UPDATE SET split_mask = split_mask | excluded.split_mask
            """,
                (content_hash, 1 << split),
            )
            count += 1
            if count % 10_000 == 0:
                connection.commit()
            if count % 250_000 == 0:
                print(
                    json.dumps({"event": "corpus_partition_ingest", "rows": count}),
                    file=sys.stderr,
                    flush=True,
                )
        if before != sha256_file(path):
            raise ValueError(f"input changed during partitioning: {path}")
        receipts.append(
            {
                "path": f"sentences/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": before,
            }
        )
    connection.commit()
    if count == 0:
        raise ValueError("input corpus contains no sentences")
    if used_assignments != set(group_assignments):
        raise ValueError(
            f"assigned groups absent from corpus: {sorted(set(group_assignments) - used_assignments)}"
        )
    return receipts


def export(connection: sqlite3.Connection, stage: Path) -> tuple[dict, dict]:
    sources = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT source FROM documents ORDER BY source"
        )
    ]
    if len({source.lower() for source in sources}) != len(sources):
        raise ValueError("source identifiers collide on case-insensitive filesystems")
    splits = {}
    quarantine = {"sentences": 0, "tokens": 0}
    # Disk sort/index; never materialize the corpus in a Python collection.
    connection.execute(
        "CREATE INDEX sentences_by_split ON sentences(split, source, document_id, sentence_id)"
    )
    for split_id, split in enumerate(SPLITS):
        root = stage / split
        (root / "sentences").mkdir(parents=True)
        files = []
        stats = {"sentences": 0, "tokens": 0, "documents": 0, "sources": {}}
        for source in sources:
            path = root / "sentences" / f"{source}.tsv.gz"
            source_stats = Counter(
                sentences=0, tokens=0, documents=0, quarantined_sentences=0
            )
            previous_document = None
            with gzip_writer(path) as writer:
                rows = connection.execute(
                    """
                    SELECT s.document_id, s.sentence_id, s.token_count, s.tokens, s.group_id, c.split_mask
                    FROM sentences s JOIN contents c ON s.content_hash = c.hash
                    WHERE s.split = ? AND s.source = ? ORDER BY s.document_id, s.sentence_id
                """,
                    (split_id, source),
                )
                for (
                    document_id,
                    sentence_id,
                    token_count,
                    tokens,
                    group_id,
                    mask,
                ) in rows:
                    if mask & (mask - 1):
                        source_stats["quarantined_sentences"] += 1
                        quarantine["sentences"] += 1
                        quarantine["tokens"] += token_count
                        continue
                    writer.writerow(
                        (
                            source,
                            document_id,
                            sentence_id,
                            token_count,
                            tokens,
                            group_id,
                        )
                    )
                    source_stats["sentences"] += 1
                    source_stats["tokens"] += token_count
                    if document_id != previous_document:
                        source_stats["documents"] += 1
                        previous_document = document_id
            for name in ("sentences", "tokens", "documents"):
                stats[name] += source_stats[name]
            stats["sources"][source] = dict(source_stats)
            files.append(
                {
                    "path": f"sentences/{path.name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        if stats["sentences"] == 0:
            raise ValueError(
                f"{split} partition is empty; supply more document groups or another declared split policy"
            )
        splits[split] = {"files": files, "statistics": stats}
    quarantine["distinct_sentences"] = connection.execute(
        "SELECT COUNT(*) FROM contents WHERE (split_mask & (split_mask - 1)) != 0"
    ).fetchone()[0]
    return splits, quarantine


def build(
    corpus_dir: Path,
    output: Path,
    *,
    seed: str = "obadh-v1",
    train_bps: int = 9000,
    validation_bps: int = 500,
    group_assignments: dict[str, str] | None = None,
) -> dict:
    if (
        not isinstance(seed, str)
        or not seed
        or type(train_bps) is not int
        or type(validation_bps) is not int
        or train_bps <= 0
        or validation_bps <= 0
        or train_bps + validation_bps >= 10_000
    ):
        raise ValueError(
            "seed must be nonempty and all three split proportions must be positive"
        )
    if group_assignments is not None and not isinstance(group_assignments, dict):
        raise ValueError("group assignments must be a JSON object")
    group_assignments = {} if group_assignments is None else dict(group_assignments)
    if any(
        not isinstance(key, str)
        or not key
        or any(unicodedata.category(c) == "Cc" for c in key)
        or value not in SPLITS
        for key, value in group_assignments.items()
    ):
        raise ValueError(
            "group assignments must map nonempty global group IDs to train/validation/test"
        )
    corpus_dir, output = corpus_dir.resolve(), output.absolute()
    if (corpus_dir / ".building").exists():
        raise ValueError("input corpus publication is incomplete")
    if output.exists():
        raise FileExistsError(
            f"output already exists; use a new dataset directory: {output}"
        )
    if output.resolve().is_relative_to(corpus_dir):
        raise ValueError("output must be outside the input corpus")
    paths = sorted((corpus_dir / "sentences").glob("*.tsv.gz"))
    if not paths:
        raise ValueError(f"no sentence shards found: {corpus_dir}")
    input_manifest = corpus_dir / "manifest.json"
    input_manifest_digest = (
        sha256_file(input_manifest) if input_manifest.exists() else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        database = stage / "index.sqlite"
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("PRAGMA cache_size = -32768")
            connection.execute("PRAGMA temp_store = FILE")
            inputs = ingest(
                connection, paths, seed, train_bps, validation_bps, group_assignments
            )
            splits, quarantine = export(connection, stage)
        database.unlink()
        # Include input manifest bytes when available, without trusting its claims.
        if (
            (corpus_dir / ".building").exists()
            or sorted((corpus_dir / "sentences").glob("*.tsv.gz")) != paths
            or (sha256_file(input_manifest) if input_manifest.exists() else None)
            != input_manifest_digest
        ):
            raise ValueError(
                "input corpus inventory or manifest changed during partitioning"
            )
        manifest = {
            "artifact": PARTITIONS_KIND,
            "version": VERSION,
            "policy": {
                "seed": seed,
                "train_basis_points": train_bps,
                "validation_basis_points": validation_bps,
                "test_basis_points": 10_000 - train_bps - validation_bps,
                "assignment": "sha256-document-group-v1",
                "overlap": "quarantine-all-cross-split-exact-NFC-sentences",
                "normalization": "NFC for overlap identity only; original tokens preserved",
                "near_duplicates_checked": False,
            },
            "inputs": inputs,
            "input_manifest_sha256": input_manifest_digest,
            "splits": splits,
            "quarantine": quarantine,
        }
        if group_assignments:
            manifest["policy"]["group_assignments"] = group_assignments
        manifest["dataset_id"] = digest_json(manifest)
        write_json(stage / "manifest.json", manifest)
        for split, receipt in splits.items():
            write_json(
                stage / split / "manifest.json",
                {
                    "artifact": PARTITION_KIND,
                    "version": VERSION,
                    "dataset_id": manifest["dataset_id"],
                    "split": split,
                    **receipt,
                },
            )
            load_partition(stage / split)
        # mkdir is the no-overwrite publication guard. Consumers only see a
        # manifest after every partition has been moved into place.
        output.mkdir()
        try:
            (output / ".building").touch()
            for split in SPLITS:
                (stage / split).rename(output / split)
            (stage / "manifest.json").rename(output / "manifest.json")
            (output / ".building").unlink()
        except BaseException:
            shutil.rmtree(output)
            raise
        return manifest
    finally:
        shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser(
        "build", help="build immutable document-level corpus partitions"
    )
    create.add_argument("--corpus-dir", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--seed", default="obadh-v1")
    create.add_argument("--train-basis-points", type=int, default=9000)
    create.add_argument("--validation-basis-points", type=int, default=500)
    create.add_argument(
        "--group-assignments",
        type=Path,
        help="JSON mapping explicit global group IDs to fixed train/validation/test roles",
    )
    verify = commands.add_parser(
        "verify", help="verify all partition receipts and file bytes"
    )
    verify.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build(
                args.corpus_dir,
                args.output,
                seed=args.seed,
                train_bps=args.train_basis_points,
                validation_bps=args.validation_basis_points,
                group_assignments=json.loads(
                    args.group_assignments.read_text(encoding="utf-8")
                )
                if args.group_assignments
                else None,
            )
        else:
            result = {split: load_partition(args.dataset / split) for split in SPLITS}
            if any(value is None for value in result.values()):
                raise ValueError("verify requires a versioned partition set")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        csv.Error,
        sqlite3.Error,
    ) as error:
        parser.exit(2, f"corpus partition error: {error}\n")


if __name__ == "__main__":
    main()
