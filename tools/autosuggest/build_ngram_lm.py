#!/usr/bin/env python3
"""Build Obadh's compact next-word n-gram autosuggest artifact."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sqlite3
import struct
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from tools.autosuggest.common import BOS_ID, PAD_ID, UNK_ID, load_vocab, sentence_paths
from tools.corpus.provenance import (
    digest_json, sha256_file, training_provenance, verify_artifact_training,
)


MAGIC = b"OBAUTOSUGLM_V1\0\0"
MAGIC_V2 = b"OBAUTOSUGLM_V2\0\0"
MAGIC_V3 = b"OBAUTOSUGLM_V3\0\0"
VERSION = 1
VERSION_V2 = 2
VERSION_V3 = 3
U32 = struct.Struct("<I")
I32 = struct.Struct("<i")
CANDIDATE_RECORD_LEN = 12
COUNT_CANDIDATE_RECORD_LEN = 8
SCORE_SCALE = 1_000_000.0
MIN_PROBABILITY = 1e-12
FNV32_OFFSET = 0x811C9DC5
FNV32_PRIME = 0x01000193
FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
SQLITE_COUNT_METADATA_TABLE = "obadh_count_metadata"
SQLITE_COUNT_METADATA_VERSION = "1"


@dataclass(frozen=True)
class OrderMinCounts:
    bigram: int
    trigram: int
    fourgram: int

    @classmethod
    def from_cli(
        cls,
        min_count: int,
        *,
        bigram_min_count: int | None,
        trigram_min_count: int | None,
        fourgram_min_count: int | None,
    ) -> "OrderMinCounts":
        if min_count < 1:
            raise ValueError("--min-count must be at least 1")
        values = cls(
            bigram=bigram_min_count if bigram_min_count is not None else min_count,
            trigram=trigram_min_count if trigram_min_count is not None else min_count,
            fourgram=fourgram_min_count if fourgram_min_count is not None else min_count,
        )
        if values.bigram < 1 or values.trigram < 1 or values.fourgram < 1:
            raise ValueError("order-specific min-count values must be at least 1")
        return values


@dataclass(frozen=True)
class SourceWeights:
    values: dict[str, int]

    @classmethod
    def from_cli(cls, entries: list[str] | None) -> "SourceWeights":
        weights: dict[str, int] = {}
        for entry in entries or []:
            if "=" not in entry:
                raise ValueError("--source-weight entries must look like source=weight")
            source, raw_weight = entry.split("=", 1)
            source = source.strip()
            if not source:
                raise ValueError("--source-weight source must not be empty")
            try:
                weight = int(raw_weight)
            except ValueError as error:
                raise ValueError(f"invalid source weight for {source}: {raw_weight}") from error
            if weight < 1:
                raise ValueError("--source-weight values must be positive integers")
            weights[source] = weight
        return cls(weights)

    def weight_for(self, source: str) -> int:
        return self.values.get(source, 1)


@dataclass(frozen=True)
class ModifiedDiscounts:
    d1: float
    d2: float
    d3_plus: float

    @classmethod
    def fixed(cls, value: float) -> "ModifiedDiscounts":
        discount = max(0.0, min(1.0, value))
        return cls(discount, discount, discount)

    @classmethod
    def from_count_histogram(
        cls,
        histogram: dict[int, int],
        fallback_discount: float,
    ) -> "ModifiedDiscounts":
        n1 = histogram.get(1, 0)
        n2 = histogram.get(2, 0)
        n3 = histogram.get(3, 0)
        n4 = histogram.get(4, 0)
        fallback = cls.fixed(fallback_discount)
        if n1 <= 0 or n2 <= 0:
            return fallback
        y = n1 / (n1 + 2.0 * n2)
        d1 = 1.0 - 2.0 * y * n2 / n1
        d2 = fallback.d2 if n3 <= 0 else 2.0 - 3.0 * y * n3 / n2
        d3_plus = fallback.d3_plus if n4 <= 0 else 3.0 - 4.0 * y * n4 / n3
        return cls(
            clamp_discount(d1, 1.0, fallback.d1),
            clamp_discount(d2, 2.0, fallback.d2),
            clamp_discount(d3_plus, 3.0, fallback.d3_plus),
        )

    def for_count(self, count: int) -> float:
        if count <= 0:
            return 0.0
        if count == 1:
            return self.d1
        if count == 2:
            return self.d2
        return self.d3_plus

    def backoff_mass(self, singleton_count: int, doubleton_count: int, high_count: int) -> float:
        return (
            self.d1 * singleton_count
            + self.d2 * doubleton_count
            + self.d3_plus * high_count
        )


class CandidateScorer(Protocol):
    score_name: str

    def score(
        self,
        token_id: int,
        count: int,
        context_total: int | None,
        order: int,
        context: tuple[int, ...] = (),
    ) -> int:
        ...


class MemoryCounts:
    def __init__(self, max_context_order: int) -> None:
        self.max_context_order = max_context_order
        self.unigrams: Counter[int] = Counter()
        self.bigrams: dict[int, Counter[int]] = defaultdict(Counter)
        self.trigrams: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.fourgrams: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)

    def observe(self, encoded: list[int], weight: int = 1) -> None:
        for index in range(1, len(encoded)):
            target = encoded[index]
            if not is_target_id(target):
                continue
            self.unigrams[target] += weight
            previous = encoded[index - 1]
            if is_context_id(previous):
                self.bigrams[previous][target] += weight
            if index >= 2:
                previous2 = encoded[index - 2]
                if is_context_id(previous2) and is_context_id(previous):
                    self.trigrams[(previous2, previous)][target] += weight
            if self.max_context_order >= 3 and index >= 3:
                previous3 = encoded[index - 3]
                previous2 = encoded[index - 2]
                if (
                    is_context_id(previous3)
                    and is_context_id(previous2)
                    and is_context_id(previous)
                ):
                    self.fourgrams[(previous3, previous2, previous)][target] += weight

    def finalize(self) -> None:
        return

    def rows(
        self,
        max_candidates_per_prefix: int,
        min_counts: OrderMinCounts,
        scorer: CandidateScorer,
    ) -> tuple[
        list[tuple[int, int, int]],
        list[tuple[int, int, list[tuple[int, int, int]]]],
        list[tuple[int, int, int, list[tuple[int, int, int]]]],
        list[tuple[int, int, int, int, list[tuple[int, int, int]]]],
    ]:
        unigrams = sorted(
            (
                (token_id, count, scorer.score(token_id, count, None, order=1))
                for token_id, count in self.unigrams.items()
            ),
            key=candidate_sort_key,
        )
        bigrams = [
            (
                prefix,
                0,
                top_candidates(
                    counter,
                    max_candidates_per_prefix,
                    min_counts.bigram,
                    scorer,
                    order=2,
                    context=(prefix,),
                ),
            )
            for prefix, counter in sorted(self.bigrams.items())
        ]
        trigrams = [
            (
                prefix1,
                prefix2,
                0,
                top_candidates(
                    counter,
                    max_candidates_per_prefix,
                    min_counts.trigram,
                    scorer,
                    order=3,
                    context=(prefix1, prefix2),
                ),
            )
            for (prefix1, prefix2), counter in sorted(self.trigrams.items())
        ]
        fourgrams = [
            (
                prefix1,
                prefix2,
                prefix3,
                0,
                top_candidates(
                    counter,
                    max_candidates_per_prefix,
                    min_counts.fourgram,
                    scorer,
                    order=4,
                    context=(prefix1, prefix2, prefix3),
                ),
            )
            for (prefix1, prefix2, prefix3), counter in sorted(self.fourgrams.items())
        ]
        return (
            unigrams,
            [(prefix, total, candidates) for prefix, total, candidates in bigrams if candidates],
            [
                (prefix1, prefix2, total, candidates)
                for prefix1, prefix2, total, candidates in trigrams
                if candidates
            ],
            [
                (prefix1, prefix2, prefix3, total, candidates)
                for prefix1, prefix2, prefix3, total, candidates in fourgrams
                if candidates
            ],
        )


class SqliteCounts:
    def __init__(
        self,
        path: Path,
        batch_size: int,
        max_context_order: int,
        metadata: dict[str, str],
        reset: bool = True,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset and path.exists():
            path.unlink()
        self.path = path
        self.batch_size = batch_size
        self.max_context_order = max_context_order
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        if not reset:
            try:
                self._verify_existing(metadata)
            except BaseException:
                self.connection.close()
                raise
            self.unigram_batch = Counter()
            self.bigram_batch = Counter()
            self.trigram_batch = Counter()
            self.fourgram_batch = Counter()
            self.pending = 0
            return
        self._create_metadata(metadata)
        self.connection.execute(
            "CREATE TABLE unigrams(token INTEGER PRIMARY KEY, count INTEGER NOT NULL)"
        )
        self.connection.execute(
            """
            CREATE TABLE bigrams(
              prefix INTEGER NOT NULL,
              token INTEGER NOT NULL,
              count INTEGER NOT NULL,
              PRIMARY KEY(prefix, token)
            ) WITHOUT ROWID
            """
        )
        self.connection.execute(
            """
            CREATE TABLE trigrams(
              prefix1 INTEGER NOT NULL,
              prefix2 INTEGER NOT NULL,
              token INTEGER NOT NULL,
              count INTEGER NOT NULL,
              PRIMARY KEY(prefix1, prefix2, token)
            ) WITHOUT ROWID
            """
        )
        if self.max_context_order >= 3:
            self.connection.execute(
                """
                CREATE TABLE fourgrams(
                  prefix1 INTEGER NOT NULL,
                  prefix2 INTEGER NOT NULL,
                  prefix3 INTEGER NOT NULL,
                  token INTEGER NOT NULL,
                  count INTEGER NOT NULL,
                  PRIMARY KEY(prefix1, prefix2, prefix3, token)
                ) WITHOUT ROWID
                """
            )
        self.unigram_batch: Counter[int] = Counter()
        self.bigram_batch: Counter[tuple[int, int]] = Counter()
        self.trigram_batch: Counter[tuple[int, int, int]] = Counter()
        self.fourgram_batch: Counter[tuple[int, int, int, int]] = Counter()
        self.pending = 0

    def _create_metadata(self, metadata: dict[str, str]) -> None:
        self.connection.execute(
            f"CREATE TABLE {SQLITE_COUNT_METADATA_TABLE}(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        self.connection.executemany(
            f"INSERT INTO {SQLITE_COUNT_METADATA_TABLE}(key, value) VALUES(?, ?)",
            sorted(metadata.items()),
        )

    def _verify_existing(self, expected_metadata: dict[str, str]) -> None:
        required = {"unigrams", "bigrams", "trigrams"}
        if self.max_context_order >= 3:
            required.add("fourgrams")
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        existing = {row[0] for row in rows}
        missing = required - existing
        if missing:
            raise ValueError(f"existing SQLite count DB is missing tables: {sorted(missing)}")
        if SQLITE_COUNT_METADATA_TABLE not in existing:
            raise ValueError(
                "existing SQLite count DB has no Obadh metadata; rebuild it without --reuse-sqlite "
                "before using it for profile exports"
            )
        metadata = {
            str(key): str(value)
            for key, value in self.connection.execute(
                f"SELECT key, value FROM {SQLITE_COUNT_METADATA_TABLE}"
            )
        }
        mismatches = {
            key: (expected, metadata.get(key))
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{key}: expected {expected}, found {actual}"
                for key, (expected, actual) in sorted(mismatches.items())
            )
            raise ValueError(f"existing SQLite count DB metadata mismatch: {details}")

    def observe(self, encoded: list[int], weight: int = 1) -> None:
        for index in range(1, len(encoded)):
            target = encoded[index]
            if not is_target_id(target):
                continue
            self.unigram_batch[target] += weight
            self.pending += 1

            previous = encoded[index - 1]
            if is_context_id(previous):
                self.bigram_batch[(previous, target)] += weight
                self.pending += 1
            if index >= 2:
                previous2 = encoded[index - 2]
                if is_context_id(previous2) and is_context_id(previous):
                    self.trigram_batch[(previous2, previous, target)] += weight
                    self.pending += 1
            if self.max_context_order >= 3 and index >= 3:
                previous3 = encoded[index - 3]
                previous2 = encoded[index - 2]
                if (
                    is_context_id(previous3)
                    and is_context_id(previous2)
                    and is_context_id(previous)
                ):
                    self.fourgram_batch[(previous3, previous2, previous, target)] += weight
                    self.pending += 1

            if self.pending >= self.batch_size:
                self.flush()

    def flush(self) -> None:
        if self.unigram_batch:
            self.connection.executemany(
                """
                INSERT INTO unigrams(token, count) VALUES(?, ?)
                ON CONFLICT(token) DO UPDATE SET count = count + excluded.count
                """,
                self.unigram_batch.items(),
            )
            self.unigram_batch.clear()
        if self.bigram_batch:
            self.connection.executemany(
                """
                INSERT INTO bigrams(prefix, token, count) VALUES(?, ?, ?)
                ON CONFLICT(prefix, token) DO UPDATE SET count = count + excluded.count
                """,
                ((prefix, token, count) for (prefix, token), count in self.bigram_batch.items()),
            )
            self.bigram_batch.clear()
        if self.trigram_batch:
            self.connection.executemany(
                """
                INSERT INTO trigrams(prefix1, prefix2, token, count) VALUES(?, ?, ?, ?)
                ON CONFLICT(prefix1, prefix2, token) DO UPDATE SET count = count + excluded.count
                """,
                (
                    (prefix1, prefix2, token, count)
                    for (prefix1, prefix2, token), count in self.trigram_batch.items()
                ),
            )
            self.trigram_batch.clear()
        if self.fourgram_batch:
            self.connection.executemany(
                """
                INSERT INTO fourgrams(prefix1, prefix2, prefix3, token, count)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(prefix1, prefix2, prefix3, token)
                DO UPDATE SET count = count + excluded.count
                """,
                (
                    (prefix1, prefix2, prefix3, token, count)
                    for (prefix1, prefix2, prefix3, token), count in self.fourgram_batch.items()
                ),
            )
            self.fourgram_batch.clear()
        self.connection.commit()
        self.pending = 0

    def finalize(self) -> None:
        self.flush()
        self.connection.execute("CREATE INDEX IF NOT EXISTS bigram_rank ON bigrams(prefix, count DESC, token)")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS trigram_rank ON trigrams(prefix1, prefix2, count DESC, token)"
        )
        if self.max_context_order >= 3:
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS fourgram_rank ON fourgrams(prefix1, prefix2, prefix3, count DESC, token)"
            )
        self.connection.commit()

def build_ngram_lm(
    corpus_dir: Path,
    vocab_path: Path,
    output: Path,
    backend: str,
    sqlite_path: Path,
    sources: set[str] | None,
    source_weights: SourceWeights,
    max_sentences: int | None,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
    reuse_sqlite: bool,
    log_every_sentences: int,
    max_candidates_per_prefix: int,
    unigram_size: int,
    min_count: int,
    bigram_min_count: int | None,
    trigram_min_count: int | None,
    fourgram_min_count: int | None,
    batch_size: int,
    smoothing: float,
    backoff_alpha: float,
    kneser_ney_discount: float,
    score_mode: str,
    max_context_order: int,
    compact_count_records: bool,
) -> dict:
    if reuse_sqlite and backend != "sqlite":
        raise ValueError("--reuse-sqlite requires --backend sqlite")
    if max_context_order not in (2, 3):
        raise ValueError("--max-context-order must be 2 or 3")
    if compact_count_records and score_mode != "count":
        raise ValueError("--compact-count-records requires --score-mode count")
    if not (0.0 < kneser_ney_discount < 1.0):
        raise ValueError("--kneser-ney-discount must be between 0 and 1")
    min_counts = OrderMinCounts.from_cli(
        min_count,
        bigram_min_count=bigram_min_count,
        trigram_min_count=trigram_min_count,
        fourgram_min_count=fourgram_min_count,
    )
    provenance = training_provenance(corpus_dir)
    verify_artifact_training(vocab_path, expected=provenance)
    words, vocab = load_vocab(vocab_path)
    fingerprint = vocab_fingerprint(words)
    sqlite_metadata = {
        "artifact": "obadh-autosuggest-sqlite-counts",
        "metadata_version": SQLITE_COUNT_METADATA_VERSION,
        "vocab_size": str(len(words)),
        "vocab_fingerprint": str(fingerprint),
        "max_context_order": str(max_context_order),
    }
    if provenance is not None:
        # Cached counts depend on corpus membership AND the ingestion policy.
        # Reusing a full-corpus DB with a train-only vocabulary is invalid even
        # when vocabulary IDs happen to match.
        sqlite_metadata["training_selection_sha256"] = digest_json({
            "training_corpus": provenance,
            "sources": sorted(sources) if sources else None,
            "source_weights": source_weights.values,
            "max_sentences": max_sentences,
            "skip_sentences_per_source": skip_sentences_per_source,
            "max_sentences_per_source": max_sentences_per_source,
        })
        sqlite_metadata["vocab_sha256"] = sha256_file(vocab_path)
        sqlite_metadata["counting_complete"] = "1" if reuse_sqlite else "0"
    counts = (
        MemoryCounts(max_context_order)
        if backend == "memory"
        else SqliteCounts(
            sqlite_path,
            batch_size,
            max_context_order=max_context_order,
            metadata=sqlite_metadata,
            reset=not reuse_sqlite,
        )
    )
    try:
        observed_sentences = 0
        observed_tokens = 0
        source_sentences: Counter[str] = Counter()
        weighted_source_tokens: Counter[str] = Counter()
        started_at = time.monotonic()

        if reuse_sqlite and provenance is not None:
            saved = counts.connection.execute(
                f"SELECT value FROM {SQLITE_COUNT_METADATA_TABLE} WHERE key = 'collection_statistics'"
            ).fetchone()
            if saved is None:
                raise ValueError("verified SQLite counts are missing collection statistics; rebuild the cache")
            statistics = json.loads(saved[0])
            observed_sentences = statistics["observed_sentences"]
            observed_tokens = statistics["observed_tokens"]
            source_sentences.update(statistics["source_sentences"])
            weighted_source_tokens.update(statistics["weighted_source_tokens"])

        if not reuse_sqlite:
            for source, tokens in iter_limited_sentence_tokens(
                corpus_dir,
                sources=sources,
                max_sentences=max_sentences,
                skip_sentences_per_source=skip_sentences_per_source,
                max_sentences_per_source=max_sentences_per_source,
            ):
                encoded = encode_tokens(tokens, vocab)
                if len(encoded) < 2:
                    continue
                weight = source_weights.weight_for(source)
                counts.observe(encoded, weight=weight)
                observed_sentences += 1
                observed_tokens += len(encoded) - 1
                source_sentences[source] += 1
                weighted_source_tokens[source] += (len(encoded) - 1) * weight
                if log_every_sentences > 0 and observed_sentences % log_every_sentences == 0:
                    elapsed = time.monotonic() - started_at
                    print(
                        json.dumps(
                            {
                                "event": "autosuggest_ngram_build_progress",
                                "sentences": observed_sentences,
                                "tokens": observed_tokens,
                                "weighted_tokens": sum(weighted_source_tokens.values()),
                                "elapsed_seconds": round(elapsed, 3),
                                "source_sentences": dict(source_sentences),
                                "weighted_source_tokens": dict(weighted_source_tokens),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

        if provenance is not None and training_provenance(corpus_dir) != provenance:
            raise ValueError("training corpus changed during n-gram counting")
        counts.finalize()
        if provenance is not None and isinstance(counts, SqliteCounts) and not reuse_sqlite:
            counts.connection.execute(
                f"INSERT OR REPLACE INTO {SQLITE_COUNT_METADATA_TABLE} VALUES ('collection_statistics', ?)",
                (json.dumps({"observed_sentences": observed_sentences, "observed_tokens": observed_tokens,
                             "source_sentences": dict(source_sentences),
                             "weighted_source_tokens": dict(weighted_source_tokens)}, sort_keys=True),),
            )
            counts.connection.execute(
                f"UPDATE {SQLITE_COUNT_METADATA_TABLE} SET value = '1' WHERE key = 'counting_complete'"
            )
            counts.connection.commit()
        scorer = (
            sqlite_scorer(
                counts.connection,
                smoothing,
                backoff_alpha,
                kneser_ney_discount,
                score_mode,
                max_context_order,
            )
            if isinstance(counts, SqliteCounts)
            else NgramScorer.from_memory_counts(
                counts,
                smoothing,
                backoff_alpha,
                kneser_ney_discount,
                score_mode,
                max_context_order,
            )
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(counts, SqliteCounts):
            export_report = encode_sqlite_artifact(
                words=words,
                counts=counts,
                output=output,
                max_candidates_per_prefix=max_candidates_per_prefix,
                min_counts=min_counts,
                unigram_size=unigram_size,
                scorer=scorer,
                max_context_order=max_context_order,
                compact_count_records=compact_count_records,
            )
        else:
            unigrams, bigrams, trigrams, fourgrams = counts.rows(
                max_candidates_per_prefix,
                min_counts,
                scorer,
            )
            unigrams = unigrams[:unigram_size]
            artifact = encode_artifact(
                words,
                unigrams,
                bigrams,
                trigrams,
                fourgrams,
                fingerprint,
                max_context_order=max_context_order,
                compact_count_records=compact_count_records,
            )
            output.write_bytes(artifact)
            export_report = {
                "unigram_count": len(unigrams),
                "bigram_rows": len(bigrams),
                "trigram_rows": len(trigrams),
                "fourgram_rows": len(fourgrams),
                "candidate_rows": sum(len(row[2]) for row in bigrams)
                + sum(len(row[3]) for row in trigrams)
                + sum(len(row[4]) for row in fourgrams),
                "artifact_bytes": len(artifact),
                "artifact_fingerprint": artifact_fingerprint(artifact),
                "vocab_fingerprint": fingerprint,
                "candidate_record_len": COUNT_CANDIDATE_RECORD_LEN
                if compact_count_records
                else CANDIDATE_RECORD_LEN,
            }

        artifact_version = (
            VERSION_V3 if compact_count_records else VERSION_V2 if max_context_order >= 3 else VERSION
        )
        report = {
            "artifact": "obadh-autosuggest-ngram",
            "training_corpus": provenance,
            "artifact_sha256": sha256_file(output),
            "version": artifact_version,
            "format": (
                "bounded fourgram/trigram/bigram/unigram binary"
                if max_context_order >= 3
                else "bounded trigram/bigram/unigram binary"
            ),
            "corpus_dir": str(corpus_dir),
            "vocab_path": str(vocab_path),
            "output": str(output),
            "backend": backend,
            "sqlite_path": str(sqlite_path) if backend == "sqlite" else None,
            "reuse_sqlite": reuse_sqlite,
            "sources": sorted(sources) if sources else None,
            "source_weights": source_weights.values,
            "max_sentences": max_sentences,
            "skip_sentences_per_source": skip_sentences_per_source,
            "max_sentences_per_source": max_sentences_per_source,
            "observed_sentences": observed_sentences,
            "observed_tokens": observed_tokens,
            "observed_weighted_tokens": sum(weighted_source_tokens.values()),
            "source_sentences": dict(source_sentences),
            "weighted_source_tokens": dict(weighted_source_tokens),
            "vocab_size": len(words),
            "vocab_fingerprint": export_report["vocab_fingerprint"],
            "unigram_count": export_report["unigram_count"],
            "bigram_rows": export_report["bigram_rows"],
            "trigram_rows": export_report["trigram_rows"],
            "fourgram_rows": export_report["fourgram_rows"],
            "candidate_rows": export_report["candidate_rows"],
            "candidate_record_len": export_report["candidate_record_len"],
            "compact_count_records": compact_count_records,
            "max_context_order": max_context_order,
            "max_candidates_per_prefix": max_candidates_per_prefix,
            "min_count": min_count,
            "bigram_min_count": min_counts.bigram,
            "trigram_min_count": min_counts.trigram,
            "fourgram_min_count": min_counts.fourgram,
            "smoothing": smoothing,
            "backoff_alpha": backoff_alpha,
            "kneser_ney_discount": kneser_ney_discount,
            "score": scorer.score_name,
            "artifact_bytes": export_report["artifact_bytes"],
            "artifact_fingerprint": export_report["artifact_fingerprint"],
        }
        manifest_path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        if isinstance(counts, SqliteCounts):
            counts.connection.close()


def iter_limited_sentence_tokens(
    corpus_dir: Path,
    sources: set[str] | None,
    max_sentences: int | None,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
) -> Iterator[tuple[str, list[str]]]:
    emitted_total = 0
    emitted_by_source: Counter[str] = Counter()
    seen_by_source: Counter[str] = Counter()
    pending_sources = set(sources) if sources else None

    for path in sentence_paths(corpus_dir):
        source_hint = path.stem.split(".")[0]
        if sources is not None and source_hint not in sources:
            continue
        if (
            max_sentences_per_source is not None
            and emitted_by_source[source_hint] >= max_sentences_per_source
        ):
            continue

        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                source = row["source"]
                if sources is not None and source not in sources:
                    continue
                seen_by_source[source] += 1
                if seen_by_source[source] <= skip_sentences_per_source:
                    continue
                if (
                    max_sentences_per_source is not None
                    and emitted_by_source[source] >= max_sentences_per_source
                ):
                    if pending_sources is not None:
                        pending_sources.discard(source)
                    break

                tokens = row["tokens"].split(" ") if row.get("tokens") else []
                if not tokens:
                    continue

                yield source, tokens
                emitted_total += 1
                emitted_by_source[source] += 1

                if max_sentences is not None and emitted_total >= max_sentences:
                    return
                if (
                    pending_sources is not None
                    and max_sentences_per_source is not None
                    and emitted_by_source[source] >= max_sentences_per_source
                ):
                    pending_sources.discard(source)
                    if not pending_sources:
                        return
                    break


def encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [BOS_ID, *(vocab.get(token, UNK_ID) for token in tokens)]


def is_target_id(token_id: int) -> bool:
    return token_id > UNK_ID


def is_context_id(token_id: int) -> bool:
    return token_id != PAD_ID and token_id != UNK_ID


def vocab_fingerprint(words: list[str]) -> int:
    token_bytes = bytearray()
    id_records: list[tuple[int, int]] = []
    for word in words:
        encoded = word.encode("utf-8")
        id_records.append((len(token_bytes), len(encoded)))
        token_bytes.extend(encoded)

    fingerprint = FNV32_OFFSET
    fingerprint = fnv32_update_u32(fingerprint, len(words))
    fingerprint = fnv32_update_u32(fingerprint, len(words))
    fingerprint = fnv32_update_u32(fingerprint, len(token_bytes))
    for offset, length in id_records:
        fingerprint = fnv32_update_u32(fingerprint, offset)
        fingerprint = fnv32_update_u32(fingerprint, length)
    fingerprint = fnv32_update(fingerprint, token_bytes)
    return fingerprint or 1


def fnv32_update_u32(fingerprint: int, value: int) -> int:
    return fnv32_update(fingerprint, U32.pack(value))


def fnv32_update(fingerprint: int, data: bytes | bytearray) -> int:
    for byte in data:
        fingerprint = ((fingerprint ^ byte) * FNV32_PRIME) & 0xFFFFFFFF
    return fingerprint


def artifact_fingerprint(data: bytes | bytearray) -> str:
    fingerprint = fnv64_update(FNV64_OFFSET, data)
    return f"{fingerprint or 1:016x}"


def artifact_fingerprint_file(path: Path) -> str:
    fingerprint = FNV64_OFFSET
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            fingerprint = fnv64_update(fingerprint, chunk)
    return f"{fingerprint or 1:016x}"


def fnv64_update(fingerprint: int, data: bytes | bytearray) -> int:
    for byte in data:
        fingerprint = ((fingerprint ^ byte) * FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return fingerprint


class NgramScorer:
    def __init__(
        self,
        unigram_counts: Counter[int] | dict[int, int],
        smoothing: float,
        backoff_alpha: float,
        kneser_ney_discount: float,
        score_mode: str,
        max_context_order: int,
        bigram_counts: dict[tuple[int, int], int] | None = None,
        trigram_counts: dict[tuple[int, int, int], int] | None = None,
        fourgram_counts: dict[tuple[int, int, int, int], int] | None = None,
        bigram_stats: dict[tuple[int], tuple[int, int]] | None = None,
        trigram_stats: dict[tuple[int, int], tuple[int, int]] | None = None,
        fourgram_stats: dict[tuple[int, int, int], tuple[int, int]] | None = None,
        bigram_modified_stats: dict[tuple[int], tuple[int, int, int, int]] | None = None,
        trigram_modified_stats: dict[tuple[int, int], tuple[int, int, int, int]] | None = None,
        fourgram_modified_stats: dict[
            tuple[int, int, int], tuple[int, int, int, int]
        ]
        | None = None,
        continuation_counts: dict[int, int] | None = None,
        continuation_total: int = 0,
        modified_discounts: dict[int, ModifiedDiscounts] | None = None,
    ) -> None:
        self.unigram_counts = dict(unigram_counts)
        self.unigram_total = sum(self.unigram_counts.values())
        self.smoothing = max(0.0, smoothing)
        self.backoff_alpha = max(0.0, min(1.0, backoff_alpha))
        self.kneser_ney_discount = max(0.0, min(1.0, kneser_ney_discount))
        self.score_mode = score_mode
        self.highest_ngram_order = max(1, max_context_order + 1)
        self.bigram_counts = bigram_counts or {}
        self.trigram_counts = trigram_counts or {}
        self.fourgram_counts = fourgram_counts or {}
        self.bigram_stats = bigram_stats or {}
        self.trigram_stats = trigram_stats or {}
        self.fourgram_stats = fourgram_stats or {}
        self.bigram_modified_stats = bigram_modified_stats or {}
        self.trigram_modified_stats = trigram_modified_stats or {}
        self.fourgram_modified_stats = fourgram_modified_stats or {}
        self.continuation_counts = continuation_counts or {}
        self.continuation_total = continuation_total
        self.modified_discounts = modified_discounts or {
            order: ModifiedDiscounts.fixed(self.kneser_ney_discount)
            for order in (2, 3, 4)
        }
        self.score_name = {
            "count": "raw_count_backoff",
            "smoothed-log": "smoothed_log_probability_x1e6",
            "stupid-backoff": "stupid_backoff_log_probability_x1e6",
            "kneser-ney": "interpolated_kneser_ney_log_probability_x1e6",
            "modified-kneser-ney": "modified_kneser_ney_log_probability_x1e6",
        }[score_mode]

    @classmethod
    def from_memory_counts(
        cls,
        counts: MemoryCounts,
        smoothing: float,
        backoff_alpha: float,
        kneser_ney_discount: float,
        score_mode: str,
        max_context_order: int,
    ) -> "NgramScorer":
        return cls(
            counts.unigrams,
            smoothing,
            backoff_alpha,
            kneser_ney_discount,
            score_mode,
            max_context_order,
            bigram_counts={
                (prefix, token): count
                for prefix, counter in counts.bigrams.items()
                for token, count in counter.items()
            },
            trigram_counts={
                (prefix1, prefix2, token): count
                for (prefix1, prefix2), counter in counts.trigrams.items()
                for token, count in counter.items()
            },
            fourgram_counts={
                (prefix1, prefix2, prefix3, token): count
                for (prefix1, prefix2, prefix3), counter in counts.fourgrams.items()
                for token, count in counter.items()
            },
            bigram_stats={
                (prefix,): (sum(counter.values()), len(counter))
                for prefix, counter in counts.bigrams.items()
            },
            trigram_stats={
                prefix: (sum(counter.values()), len(counter))
                for prefix, counter in counts.trigrams.items()
            },
            fourgram_stats={
                prefix: (sum(counter.values()), len(counter))
                for prefix, counter in counts.fourgrams.items()
            },
            bigram_modified_stats={
                (prefix,): modified_context_stats(counter)
                for prefix, counter in counts.bigrams.items()
            },
            trigram_modified_stats={
                prefix: modified_context_stats(counter)
                for prefix, counter in counts.trigrams.items()
            },
            fourgram_modified_stats={
                prefix: modified_context_stats(counter)
                for prefix, counter in counts.fourgrams.items()
            },
            continuation_counts=distinct_predecessor_counts(counts.bigrams),
            continuation_total=sum(len(counter) for counter in counts.bigrams.values()),
            modified_discounts={
                2: ModifiedDiscounts.from_count_histogram(
                    count_bucket_histogram_from_counters(counts.bigrams.values()),
                    kneser_ney_discount,
                ),
                3: ModifiedDiscounts.from_count_histogram(
                    count_bucket_histogram_from_counters(counts.trigrams.values()),
                    kneser_ney_discount,
                ),
                4: ModifiedDiscounts.from_count_histogram(
                    count_bucket_histogram_from_counters(counts.fourgrams.values()),
                    kneser_ney_discount,
                ),
            },
        )

    def score(
        self,
        token_id: int,
        count: int,
        context_total: int | None,
        order: int,
        context: tuple[int, ...] = (),
    ) -> int:
        if self.score_mode == "count":
            return min(count, 2_147_483_647)
        if self.unigram_total <= 0:
            return 0
        if self.score_mode in ("kneser-ney", "modified-kneser-ney"):
            return score_from_probability(
                self.kneser_ney_probability(token_id, count, order, context)
            )

        unigram_probability = self.unigram_counts.get(token_id, 0) / self.unigram_total
        if self.score_mode == "stupid-backoff":
            if context_total is None or context_total <= 0:
                probability = unigram_probability
            else:
                probability = count / context_total
            backoff_power = max(0, self.highest_ngram_order - max(1, min(self.highest_ngram_order, order)))
            probability *= self.backoff_alpha ** backoff_power
            return score_from_probability(probability)

        if context_total is None or context_total <= 0 or self.smoothing == 0.0:
            probability = count / self.unigram_total if context_total is None else count / max(context_total, 1)
        else:
            probability = (count + self.smoothing * unigram_probability) / (
                context_total + self.smoothing
            )
        return score_from_probability(probability)

    def kneser_ney_probability(
        self,
        token_id: int,
        count: int,
        order: int,
        context: tuple[int, ...],
    ) -> float:
        if order <= 1 or not context:
            return self.kneser_ney_unigram_probability(token_id)
        if order == 2 and len(context) >= 1:
            return self.kneser_ney_bigram_probability(context[-1], token_id, count)
        if order == 3 and len(context) >= 2:
            return self.kneser_ney_trigram_probability(
                context[-2],
                context[-1],
                token_id,
                count,
            )
        if order >= 4 and len(context) >= 3:
            return self.kneser_ney_fourgram_probability(
                context[-3],
                context[-2],
                context[-1],
                token_id,
                count,
            )
        return self.kneser_ney_unigram_probability(token_id)

    def kneser_ney_unigram_probability(self, token_id: int) -> float:
        if self.continuation_total > 0:
            continuation_count = self.continuation_counts.get(token_id, 0)
            if continuation_count > 0:
                return continuation_count / self.continuation_total
        return self.unigram_counts.get(token_id, 0) / max(1, self.unigram_total)

    def kneser_ney_bigram_probability(
        self,
        prefix: int,
        token_id: int,
        count: int | None = None,
    ) -> float:
        count = self.bigram_counts.get((prefix, token_id), 0) if count is None else count
        total, distinct = self.bigram_stats.get((prefix,), (0, 0))
        if total <= 0:
            return self.kneser_ney_unigram_probability(token_id)
        if self.score_mode == "modified-kneser-ney":
            total, singleton_count, doubleton_count, high_count = (
                self.bigram_modified_stats.get((prefix,), (0, 0, 0, 0))
            )
            if total <= 0:
                return self.kneser_ney_unigram_probability(token_id)
            discounts = self.modified_discounts[2]
            lower = self.kneser_ney_unigram_probability(token_id)
            return (
                max(count - discounts.for_count(count), 0.0) / total
                + (discounts.backoff_mass(singleton_count, doubleton_count, high_count) / total)
                * lower
            )
        discount = self.kneser_ney_discount
        lower = self.kneser_ney_unigram_probability(token_id)
        return max(count - discount, 0.0) / total + (discount * distinct / total) * lower

    def kneser_ney_trigram_probability(
        self,
        prefix1: int,
        prefix2: int,
        token_id: int,
        count: int | None = None,
    ) -> float:
        count = (
            self.trigram_counts.get((prefix1, prefix2, token_id), 0)
            if count is None
            else count
        )
        total, distinct = self.trigram_stats.get((prefix1, prefix2), (0, 0))
        if total <= 0:
            return self.kneser_ney_bigram_probability(prefix2, token_id)
        if self.score_mode == "modified-kneser-ney":
            total, singleton_count, doubleton_count, high_count = (
                self.trigram_modified_stats.get((prefix1, prefix2), (0, 0, 0, 0))
            )
            if total <= 0:
                return self.kneser_ney_bigram_probability(prefix2, token_id)
            discounts = self.modified_discounts[3]
            lower = self.kneser_ney_bigram_probability(prefix2, token_id)
            return (
                max(count - discounts.for_count(count), 0.0) / total
                + (discounts.backoff_mass(singleton_count, doubleton_count, high_count) / total)
                * lower
            )
        discount = self.kneser_ney_discount
        lower = self.kneser_ney_bigram_probability(prefix2, token_id)
        return max(count - discount, 0.0) / total + (discount * distinct / total) * lower

    def kneser_ney_fourgram_probability(
        self,
        prefix1: int,
        prefix2: int,
        prefix3: int,
        token_id: int,
        count: int | None = None,
    ) -> float:
        count = (
            self.fourgram_counts.get((prefix1, prefix2, prefix3, token_id), 0)
            if count is None
            else count
        )
        total, distinct = self.fourgram_stats.get((prefix1, prefix2, prefix3), (0, 0))
        if total <= 0:
            return self.kneser_ney_trigram_probability(prefix2, prefix3, token_id)
        if self.score_mode == "modified-kneser-ney":
            total, singleton_count, doubleton_count, high_count = (
                self.fourgram_modified_stats.get(
                    (prefix1, prefix2, prefix3), (0, 0, 0, 0)
                )
            )
            if total <= 0:
                return self.kneser_ney_trigram_probability(prefix2, prefix3, token_id)
            discounts = self.modified_discounts[4]
            lower = self.kneser_ney_trigram_probability(prefix2, prefix3, token_id)
            return (
                max(count - discounts.for_count(count), 0.0) / total
                + (discounts.backoff_mass(singleton_count, doubleton_count, high_count) / total)
                * lower
            )
        discount = self.kneser_ney_discount
        lower = self.kneser_ney_trigram_probability(prefix2, prefix3, token_id)
        return max(count - discount, 0.0) / total + (discount * distinct / total) * lower


class SqliteKneserNeyScorer:
    def __init__(
        self,
        connection: sqlite3.Connection,
        kneser_ney_discount: float,
        max_context_order: int,
        modified: bool = False,
    ) -> None:
        self.connection = connection
        self.kneser_ney_discount = max(0.0, min(1.0, kneser_ney_discount))
        self.highest_ngram_order = max(1, max_context_order + 1)
        self.modified = modified
        self.score_name = (
            "modified_kneser_ney_log_probability_x1e6"
            if modified
            else "interpolated_kneser_ney_log_probability_x1e6"
        )
        self.unigram_counts = dict(connection.execute("SELECT token, count FROM unigrams"))
        self.unigram_total = sum(self.unigram_counts.values())
        self.continuation_counts = {
            token: count
            for token, count in connection.execute(
                "SELECT token, COUNT(*) FROM bigrams GROUP BY token"
            )
        }
        self.continuation_total = connection.execute("SELECT COUNT(*) FROM bigrams").fetchone()[0]
        self.has_fourgrams = sqlite_has_table(connection, "fourgrams")
        self.modified_discounts = {
            2: sqlite_modified_discounts(connection, "bigrams", self.kneser_ney_discount),
            3: sqlite_modified_discounts(connection, "trigrams", self.kneser_ney_discount),
            4: sqlite_modified_discounts(connection, "fourgrams", self.kneser_ney_discount)
            if self.has_fourgrams
            else ModifiedDiscounts.fixed(self.kneser_ney_discount),
        }

    def score(
        self,
        token_id: int,
        count: int,
        context_total: int | None,
        order: int,
        context: tuple[int, ...] = (),
    ) -> int:
        return score_from_probability(
            self.kneser_ney_probability(token_id, count, order, context)
        )

    def kneser_ney_probability(
        self,
        token_id: int,
        count: int,
        order: int,
        context: tuple[int, ...],
    ) -> float:
        if order <= 1 or not context:
            return self.kneser_ney_unigram_probability(token_id)
        if order == 2 and len(context) >= 1:
            return self.kneser_ney_bigram_probability(context[-1], token_id, count)
        if order == 3 and len(context) >= 2:
            return self.kneser_ney_trigram_probability(
                context[-2],
                context[-1],
                token_id,
                count,
            )
        if order >= 4 and len(context) >= 3:
            return self.kneser_ney_fourgram_probability(
                context[-3],
                context[-2],
                context[-1],
                token_id,
                count,
            )
        return self.kneser_ney_unigram_probability(token_id)

    def kneser_ney_unigram_probability(self, token_id: int) -> float:
        if self.continuation_total > 0:
            continuation_count = self.continuation_counts.get(token_id, 0)
            if continuation_count > 0:
                return continuation_count / self.continuation_total
        return self.unigram_counts.get(token_id, 0) / max(1, self.unigram_total)

    def kneser_ney_bigram_probability(
        self,
        prefix: int,
        token_id: int,
        count: int | None = None,
    ) -> float:
        count = self.bigram_count(prefix, token_id) if count is None else count
        total, distinct = self.bigram_stats(prefix)
        if total <= 0:
            return self.kneser_ney_unigram_probability(token_id)
        if self.modified:
            total, singleton_count, doubleton_count, high_count = self.bigram_modified_stats(
                prefix
            )
            if total <= 0:
                return self.kneser_ney_unigram_probability(token_id)
            discounts = self.modified_discounts[2]
            lower = self.kneser_ney_unigram_probability(token_id)
            return (
                max(count - discounts.for_count(count), 0.0) / total
                + (discounts.backoff_mass(singleton_count, doubleton_count, high_count) / total)
                * lower
            )
        discount = self.kneser_ney_discount
        lower = self.kneser_ney_unigram_probability(token_id)
        return max(count - discount, 0.0) / total + (discount * distinct / total) * lower

    def kneser_ney_trigram_probability(
        self,
        prefix1: int,
        prefix2: int,
        token_id: int,
        count: int | None = None,
    ) -> float:
        count = self.trigram_count(prefix1, prefix2, token_id) if count is None else count
        total, distinct = self.trigram_stats(prefix1, prefix2)
        if total <= 0:
            return self.kneser_ney_bigram_probability(prefix2, token_id)
        if self.modified:
            total, singleton_count, doubleton_count, high_count = self.trigram_modified_stats(
                prefix1, prefix2
            )
            if total <= 0:
                return self.kneser_ney_bigram_probability(prefix2, token_id)
            discounts = self.modified_discounts[3]
            lower = self.kneser_ney_bigram_probability(prefix2, token_id)
            return (
                max(count - discounts.for_count(count), 0.0) / total
                + (discounts.backoff_mass(singleton_count, doubleton_count, high_count) / total)
                * lower
            )
        discount = self.kneser_ney_discount
        lower = self.kneser_ney_bigram_probability(prefix2, token_id)
        return max(count - discount, 0.0) / total + (discount * distinct / total) * lower

    def kneser_ney_fourgram_probability(
        self,
        prefix1: int,
        prefix2: int,
        prefix3: int,
        token_id: int,
        count: int | None = None,
    ) -> float:
        if not self.has_fourgrams:
            return self.kneser_ney_trigram_probability(prefix2, prefix3, token_id, count)
        count = (
            self.fourgram_count(prefix1, prefix2, prefix3, token_id)
            if count is None
            else count
        )
        total, distinct = self.fourgram_stats(prefix1, prefix2, prefix3)
        if total <= 0:
            return self.kneser_ney_trigram_probability(prefix2, prefix3, token_id)
        if self.modified:
            total, singleton_count, doubleton_count, high_count = self.fourgram_modified_stats(
                prefix1, prefix2, prefix3
            )
            if total <= 0:
                return self.kneser_ney_trigram_probability(prefix2, prefix3, token_id)
            discounts = self.modified_discounts[4]
            lower = self.kneser_ney_trigram_probability(prefix2, prefix3, token_id)
            return (
                max(count - discounts.for_count(count), 0.0) / total
                + (discounts.backoff_mass(singleton_count, doubleton_count, high_count) / total)
                * lower
            )
        discount = self.kneser_ney_discount
        lower = self.kneser_ney_trigram_probability(prefix2, prefix3, token_id)
        return max(count - discount, 0.0) / total + (discount * distinct / total) * lower

    @lru_cache(maxsize=131_072)
    def bigram_count(self, prefix: int, token_id: int) -> int:
        return sqlite_scalar_int(
            self.connection,
            "SELECT count FROM bigrams WHERE prefix = ? AND token = ?",
            (prefix, token_id),
        )

    @lru_cache(maxsize=262_144)
    def trigram_count(self, prefix1: int, prefix2: int, token_id: int) -> int:
        return sqlite_scalar_int(
            self.connection,
            """
            SELECT count FROM trigrams
            WHERE prefix1 = ? AND prefix2 = ? AND token = ?
            """,
            (prefix1, prefix2, token_id),
        )

    @lru_cache(maxsize=262_144)
    def fourgram_count(
        self,
        prefix1: int,
        prefix2: int,
        prefix3: int,
        token_id: int,
    ) -> int:
        if not self.has_fourgrams:
            return 0
        return sqlite_scalar_int(
            self.connection,
            """
            SELECT count FROM fourgrams
            WHERE prefix1 = ? AND prefix2 = ? AND prefix3 = ? AND token = ?
            """,
            (prefix1, prefix2, prefix3, token_id),
        )

    @lru_cache(maxsize=65_536)
    def bigram_stats(self, prefix: int) -> tuple[int, int]:
        return sqlite_stat_pair(
            self.connection,
            "SELECT SUM(count), COUNT(*) FROM bigrams WHERE prefix = ?",
            (prefix,),
        )

    @lru_cache(maxsize=65_536)
    def bigram_modified_stats(self, prefix: int) -> tuple[int, int, int, int]:
        return sqlite_modified_stat(
            self.connection,
            "SELECT SUM(count), SUM(count = 1), SUM(count = 2), SUM(count >= 3) FROM bigrams WHERE prefix = ?",
            (prefix,),
        )

    @lru_cache(maxsize=262_144)
    def trigram_stats(self, prefix1: int, prefix2: int) -> tuple[int, int]:
        return sqlite_stat_pair(
            self.connection,
            """
            SELECT SUM(count), COUNT(*)
            FROM trigrams
            WHERE prefix1 = ? AND prefix2 = ?
            """,
            (prefix1, prefix2),
        )

    @lru_cache(maxsize=262_144)
    def trigram_modified_stats(self, prefix1: int, prefix2: int) -> tuple[int, int, int, int]:
        return sqlite_modified_stat(
            self.connection,
            """
            SELECT SUM(count), SUM(count = 1), SUM(count = 2), SUM(count >= 3)
            FROM trigrams
            WHERE prefix1 = ? AND prefix2 = ?
            """,
            (prefix1, prefix2),
        )

    @lru_cache(maxsize=262_144)
    def fourgram_stats(self, prefix1: int, prefix2: int, prefix3: int) -> tuple[int, int]:
        if not self.has_fourgrams:
            return (0, 0)
        return sqlite_stat_pair(
            self.connection,
            """
            SELECT SUM(count), COUNT(*)
            FROM fourgrams
            WHERE prefix1 = ? AND prefix2 = ? AND prefix3 = ?
            """,
            (prefix1, prefix2, prefix3),
        )

    @lru_cache(maxsize=262_144)
    def fourgram_modified_stats(
        self, prefix1: int, prefix2: int, prefix3: int
    ) -> tuple[int, int, int, int]:
        if not self.has_fourgrams:
            return (0, 0, 0, 0)
        return sqlite_modified_stat(
            self.connection,
            """
            SELECT SUM(count), SUM(count = 1), SUM(count = 2), SUM(count >= 3)
            FROM fourgrams
            WHERE prefix1 = ? AND prefix2 = ? AND prefix3 = ?
            """,
            (prefix1, prefix2, prefix3),
        )


def distinct_predecessor_counts(
    bigrams: dict[int, Counter[int]],
) -> dict[int, int]:
    predecessors: defaultdict[int, int] = defaultdict(int)
    for counter in bigrams.values():
        for token in counter:
            predecessors[token] += 1
    return dict(predecessors)


def clamp_discount(value: float, upper_bound: float, fallback: float) -> float:
    if not math.isfinite(value) or value <= 0.0 or value >= upper_bound:
        return fallback
    return value


def count_bucket_histogram_from_counters(
    counters: Iterable[Counter[int]],
) -> dict[int, int]:
    histogram: Counter[int] = Counter()
    for counter in counters:
        for count in counter.values():
            histogram[min(count, 4)] += 1
    return dict(histogram)


def modified_context_stats(counter: Counter[int]) -> tuple[int, int, int, int]:
    total = 0
    singleton_count = 0
    doubleton_count = 0
    high_count = 0
    for count in counter.values():
        total += count
        if count == 1:
            singleton_count += 1
        elif count == 2:
            doubleton_count += 1
        elif count >= 3:
            high_count += 1
    return (total, singleton_count, doubleton_count, high_count)


def sqlite_scorer(
    connection: sqlite3.Connection,
    smoothing: float,
    backoff_alpha: float,
    kneser_ney_discount: float,
    score_mode: str,
    max_context_order: int,
) -> CandidateScorer:
    unigram_counts = dict(connection.execute("SELECT token, count FROM unigrams"))
    if score_mode not in ("kneser-ney", "modified-kneser-ney"):
        return NgramScorer(
            unigram_counts,
            smoothing,
            backoff_alpha,
            kneser_ney_discount,
            score_mode,
            max_context_order,
        )
    return SqliteKneserNeyScorer(
        connection,
        kneser_ney_discount,
        max_context_order,
        modified=score_mode == "modified-kneser-ney",
    )


def sqlite_has_table(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def sqlite_scalar_int(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[int, ...],
) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def sqlite_stat_pair(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[int, ...],
) -> tuple[int, int]:
    row = connection.execute(query, params).fetchone()
    if not row or row[0] is None:
        return (0, 0)
    return (int(row[0]), int(row[1]))


def sqlite_modified_stat(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[int, ...],
) -> tuple[int, int, int, int]:
    row = connection.execute(query, params).fetchone()
    if not row or row[0] is None:
        return (0, 0, 0, 0)
    return (
        int(row[0] or 0),
        int(row[1] or 0),
        int(row[2] or 0),
        int(row[3] or 0),
    )


def sqlite_modified_discounts(
    connection: sqlite3.Connection,
    table: str,
    fallback_discount: float,
) -> ModifiedDiscounts:
    if not sqlite_has_table(connection, table):
        return ModifiedDiscounts.fixed(fallback_discount)
    histogram = {
        int(bucket): int(total)
        for bucket, total in connection.execute(
            f"""
            SELECT CASE WHEN count >= 4 THEN 4 ELSE count END AS bucket, COUNT(*)
            FROM {table}
            GROUP BY bucket
            """
        )
    }
    return ModifiedDiscounts.from_count_histogram(histogram, fallback_discount)


def score_from_probability(probability: float) -> int:
    probability = max(MIN_PROBABILITY, min(1.0, probability))
    return int(round(math.log(probability) * SCORE_SCALE))


def candidate_sort_key(candidate: tuple[int, int, int]) -> tuple[int, int, int]:
    token_id, count, score = candidate
    return (-score, -count, token_id)


def top_candidates(
    counter: Counter[int],
    limit: int,
    min_count: int,
    scorer: CandidateScorer,
    order: int,
    context: tuple[int, ...],
) -> list[tuple[int, int, int]]:
    context_total = sum(counter.values())
    candidates = [
        (
            token,
            count,
            scorer.score(token, count, context_total, order=order, context=context),
        )
        for token, count in counter.items()
        if count >= min_count
    ]
    candidates.sort(key=candidate_sort_key)
    return candidates[:limit]


def encode_artifact(
    words: list[str],
    unigrams: list[tuple[int, int, int]],
    bigrams: list[tuple[int, int, list[tuple[int, int, int]]]],
    trigrams: list[tuple[int, int, int, list[tuple[int, int, int]]]],
    fourgrams: list[tuple[int, int, int, int, list[tuple[int, int, int]]]],
    fingerprint: int,
    max_context_order: int,
    compact_count_records: bool,
) -> bytes:
    token_bytes = bytearray()
    id_records: list[tuple[int, int]] = []
    for word in words:
        encoded = word.encode("utf-8")
        id_records.append((len(token_bytes), len(encoded)))
        token_bytes.extend(encoded)

    token_index = [
        (token_bytes[offset : offset + length], offset, length, token_id)
        for token_id, (offset, length) in enumerate(id_records)
    ]
    token_index.sort(key=lambda item: bytes(item[0]))

    candidate_records: list[tuple[int, int, int]] = []
    bigram_rows: list[tuple[int, int, int]] = []
    for prefix, _, candidates in bigrams:
        start = len(candidate_records)
        for token_id, count, score in candidates:
            candidate_records.append((token_id, count, score))
        bigram_rows.append((prefix, start, len(candidates)))

    trigram_rows: list[tuple[int, int, int, int]] = []
    for prefix1, prefix2, _, candidates in trigrams:
        start = len(candidate_records)
        for token_id, count, score in candidates:
            candidate_records.append((token_id, count, score))
        trigram_rows.append((prefix1, prefix2, start, len(candidates)))

    fourgram_rows: list[tuple[int, int, int, int, int]] = []
    for prefix1, prefix2, prefix3, _, candidates in fourgrams:
        start = len(candidate_records)
        for token_id, count, score in candidates:
            candidate_records.append((token_id, count, score))
        fourgram_rows.append((prefix1, prefix2, prefix3, start, len(candidates)))

    artifact_version = (
        VERSION_V3 if compact_count_records else VERSION_V2 if max_context_order >= 3 else VERSION
    )
    artifact = bytearray()
    artifact.extend(
        MAGIC_V3 if artifact_version == VERSION_V3 else MAGIC_V2 if artifact_version == VERSION_V2 else MAGIC
    )
    write_u32(artifact, artifact_version)
    write_u32(artifact, len(words))
    write_u32(artifact, len(token_index))
    write_u32(artifact, len(unigrams))
    write_u32(artifact, len(bigram_rows))
    write_u32(artifact, len(trigram_rows))
    if artifact_version >= VERSION_V2:
        write_u32(artifact, len(fourgram_rows))
    write_u32(artifact, len(candidate_records))
    write_u32(artifact, len(token_bytes))
    write_u32(artifact, fingerprint)
    if artifact_version >= VERSION_V3:
        write_u32(
            artifact,
            COUNT_CANDIDATE_RECORD_LEN if compact_count_records else CANDIDATE_RECORD_LEN,
        )

    for offset, length in id_records:
        write_u32(artifact, offset)
        write_u32(artifact, length)
    for _, offset, length, token_id in token_index:
        write_u32(artifact, offset)
        write_u32(artifact, length)
        write_u32(artifact, token_id)
    for token_id, count, score in unigrams:
        write_candidate_record_to_buffer(artifact, token_id, count, score, compact_count_records)
    for prefix, start, length in bigram_rows:
        write_u32(artifact, prefix)
        write_u32(artifact, start)
        write_u32(artifact, length)
    for prefix1, prefix2, start, length in trigram_rows:
        write_u32(artifact, prefix1)
        write_u32(artifact, prefix2)
        write_u32(artifact, start)
        write_u32(artifact, length)
    for prefix1, prefix2, prefix3, start, length in fourgram_rows:
        write_u32(artifact, prefix1)
        write_u32(artifact, prefix2)
        write_u32(artifact, prefix3)
        write_u32(artifact, start)
        write_u32(artifact, length)
    for token_id, count, score in candidate_records:
        write_candidate_record_to_buffer(artifact, token_id, count, score, compact_count_records)
    artifact.extend(token_bytes)
    return bytes(artifact)


def encode_sqlite_artifact(
    words: list[str],
    counts: SqliteCounts,
    output: Path,
    max_candidates_per_prefix: int,
    min_counts: OrderMinCounts,
    unigram_size: int,
    scorer: CandidateScorer,
    max_context_order: int,
    compact_count_records: bool,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"{output.name}.", dir=output.parent) as temp_dir:
        temp_path = Path(temp_dir)
        id_tokens_path = temp_path / "id_tokens.bin"
        token_index_path = temp_path / "token_index.bin"
        unigrams_path = temp_path / "unigrams.bin"
        bigram_rows_path = temp_path / "bigram_rows.bin"
        trigram_rows_path = temp_path / "trigram_rows.bin"
        fourgram_rows_path = temp_path / "fourgram_rows.bin"
        candidates_path = temp_path / "candidates.bin"
        token_bytes_path = temp_path / "token_bytes.bin"

        token_bytes_len = write_token_sections(
            words,
            id_tokens_path,
            token_index_path,
            token_bytes_path,
        )
        unigram_count = write_unigram_section(
            counts.connection,
            unigrams_path,
            unigram_size,
            scorer,
            compact_count_records,
        )
        candidate_count = 0
        bigram_rows, candidate_count = write_bigram_sections(
            counts.connection,
            bigram_rows_path,
            candidates_path,
            max_candidates_per_prefix,
            min_counts.bigram,
            scorer,
            candidate_count,
            compact_count_records,
        )
        trigram_rows, candidate_count = write_trigram_sections(
            counts.connection,
            trigram_rows_path,
            candidates_path,
            max_candidates_per_prefix,
            min_counts.trigram,
            scorer,
            candidate_count,
            compact_count_records,
        )
        if max_context_order >= 3:
            fourgram_rows, candidate_count = write_fourgram_sections(
                counts.connection,
                fourgram_rows_path,
                candidates_path,
                max_candidates_per_prefix,
                min_counts.fourgram,
                scorer,
                candidate_count,
                compact_count_records,
            )
        else:
            fourgram_rows = 0
            fourgram_rows_path.write_bytes(b"")
        fingerprint = vocab_fingerprint(words)

        with output.open("wb") as handle:
            write_header(
                handle,
                version=VERSION_V3
                if compact_count_records
                else VERSION_V2 if max_context_order >= 3 else VERSION,
                vocab_size=len(words),
                token_index_count=len(words),
                unigram_count=unigram_count,
                bigram_rows=bigram_rows,
                trigram_rows=trigram_rows,
                fourgram_rows=fourgram_rows,
                candidate_count=candidate_count,
                token_bytes_len=token_bytes_len,
                vocab_fingerprint=fingerprint,
                compact_count_records=compact_count_records,
            )
            append_file(handle, id_tokens_path)
            append_file(handle, token_index_path)
            append_file(handle, unigrams_path)
            append_file(handle, bigram_rows_path)
            append_file(handle, trigram_rows_path)
            if max_context_order >= 3:
                append_file(handle, fourgram_rows_path)
            append_file(handle, candidates_path)
            append_file(handle, token_bytes_path)

    return {
        "unigram_count": unigram_count,
        "bigram_rows": bigram_rows,
        "trigram_rows": trigram_rows,
        "fourgram_rows": fourgram_rows,
        "candidate_rows": candidate_count,
        "candidate_record_len": COUNT_CANDIDATE_RECORD_LEN
        if compact_count_records
        else CANDIDATE_RECORD_LEN,
        "artifact_bytes": output.stat().st_size,
        "artifact_fingerprint": artifact_fingerprint_file(output),
        "vocab_fingerprint": fingerprint,
    }


def write_token_sections(
    words: list[str],
    id_tokens_path: Path,
    token_index_path: Path,
    token_bytes_path: Path,
) -> int:
    id_records: list[tuple[int, int]] = []
    token_index: list[tuple[bytes, int, int, int]] = []
    token_bytes_len = 0

    with token_bytes_path.open("wb") as token_bytes_handle:
        for token_id, word in enumerate(words):
            encoded = word.encode("utf-8")
            offset = token_bytes_len
            length = len(encoded)
            id_records.append((offset, length))
            token_index.append((encoded, offset, length, token_id))
            token_bytes_handle.write(encoded)
            token_bytes_len += length

    with id_tokens_path.open("wb") as id_tokens_handle:
        for offset, length in id_records:
            write_u32_to_file(id_tokens_handle, offset)
            write_u32_to_file(id_tokens_handle, length)

    token_index.sort(key=lambda item: item[0])
    with token_index_path.open("wb") as token_index_handle:
        for _, offset, length, token_id in token_index:
            write_u32_to_file(token_index_handle, offset)
            write_u32_to_file(token_index_handle, length)
            write_u32_to_file(token_index_handle, token_id)

    return token_bytes_len


def write_unigram_section(
    connection: sqlite3.Connection,
    output: Path,
    unigram_size: int,
    scorer: CandidateScorer,
    compact_count_records: bool,
) -> int:
    candidates = [
        (
            token_id,
            frequency,
            scorer.score(token_id, frequency, None, order=1),
        )
        for token_id, frequency in connection.execute("SELECT token, count FROM unigrams")
    ]
    candidates.sort(key=candidate_sort_key)
    candidates = candidates[:unigram_size]

    with output.open("wb") as handle:
        for token_id, frequency, score in candidates:
            write_candidate_record(
                handle,
                token_id,
                frequency,
                score,
                compact_count_records,
            )
    return len(candidates)


def write_bigram_sections(
    connection: sqlite3.Connection,
    row_output: Path,
    candidate_output: Path,
    max_candidates_per_prefix: int,
    min_count: int,
    scorer: CandidateScorer,
    candidate_start: int,
    compact_count_records: bool,
) -> tuple[int, int]:
    query = """
        SELECT prefix, token, count, total
        FROM bigrams
        JOIN (
          SELECT prefix, SUM(count) AS total
          FROM bigrams
          GROUP BY prefix
        ) USING(prefix)
        WHERE count >= ?
        ORDER BY prefix ASC, count DESC, token ASC
    """
    rows = connection.execute(query, (min_count,))
    with row_output.open("wb") as row_handle, candidate_output.open("ab") as candidate_handle:
        return write_grouped_bigram_stream(
            rows,
            row_handle,
            candidate_handle,
            scorer,
            max_candidates_per_prefix,
            candidate_start,
            compact_count_records,
        )


def write_trigram_sections(
    connection: sqlite3.Connection,
    row_output: Path,
    candidate_output: Path,
    max_candidates_per_prefix: int,
    min_count: int,
    scorer: CandidateScorer,
    candidate_start: int,
    compact_count_records: bool,
) -> tuple[int, int]:
    query = """
        SELECT prefix1, prefix2, token, count, total
        FROM trigrams
        JOIN (
          SELECT prefix1, prefix2, SUM(count) AS total
          FROM trigrams
          GROUP BY prefix1, prefix2
        ) USING(prefix1, prefix2)
        WHERE count >= ?
        ORDER BY prefix1 ASC, prefix2 ASC, count DESC, token ASC
    """
    rows = connection.execute(query, (min_count,))
    with row_output.open("wb") as row_handle, candidate_output.open("ab") as candidate_handle:
        return write_grouped_trigram_stream(
            rows,
            row_handle,
            candidate_handle,
            scorer,
            max_candidates_per_prefix,
            candidate_start,
            compact_count_records,
        )


def write_fourgram_sections(
    connection: sqlite3.Connection,
    row_output: Path,
    candidate_output: Path,
    max_candidates_per_prefix: int,
    min_count: int,
    scorer: CandidateScorer,
    candidate_start: int,
    compact_count_records: bool,
) -> tuple[int, int]:
    query = """
        SELECT prefix1, prefix2, prefix3, token, count, total
        FROM fourgrams
        JOIN (
          SELECT prefix1, prefix2, prefix3, SUM(count) AS total
          FROM fourgrams
          GROUP BY prefix1, prefix2, prefix3
        ) USING(prefix1, prefix2, prefix3)
        WHERE count >= ?
        ORDER BY prefix1 ASC, prefix2 ASC, prefix3 ASC, count DESC, token ASC
    """
    rows = connection.execute(query, (min_count,))
    with row_output.open("wb") as row_handle, candidate_output.open("ab") as candidate_handle:
        return write_grouped_fourgram_stream(
            rows,
            row_handle,
            candidate_handle,
            scorer,
            max_candidates_per_prefix,
            candidate_start,
            compact_count_records,
        )


def write_grouped_bigram_stream(
    rows: Iterable[tuple[int, int, int, int]],
    row_handle,
    candidate_handle,
    scorer: CandidateScorer,
    limit: int,
    candidate_count: int,
    compact_count_records: bool,
) -> tuple[int, int]:
    row_count = 0
    current_prefix: int | None = None
    row_start = candidate_count
    candidates: list[tuple[int, int, int]] = []

    for prefix, token_id, count, total in rows:
        if current_prefix is not None and prefix != current_prefix:
            row_len = write_candidate_group(
                candidate_handle,
                candidates,
                limit,
                compact_count_records,
            )
            candidate_count += row_len
            write_bigram_row(row_handle, current_prefix, row_start, row_len)
            row_count += 1
            row_start = candidate_count
            candidates = []
        current_prefix = prefix
        candidates.append(
            (
                token_id,
                count,
                scorer.score(token_id, count, total, order=2, context=(prefix,)),
            )
        )

    if current_prefix is not None:
        row_len = write_candidate_group(candidate_handle, candidates, limit, compact_count_records)
        candidate_count += row_len
        write_bigram_row(row_handle, current_prefix, row_start, row_len)
        row_count += 1

    return row_count, candidate_count


def write_grouped_trigram_stream(
    rows: Iterable[tuple[int, int, int, int, int]],
    row_handle,
    candidate_handle,
    scorer: CandidateScorer,
    limit: int,
    candidate_count: int,
    compact_count_records: bool,
) -> tuple[int, int]:
    row_count = 0
    current_prefix: tuple[int, int] | None = None
    row_start = candidate_count
    candidates: list[tuple[int, int, int]] = []

    for prefix1, prefix2, token_id, count, total in rows:
        prefix = (prefix1, prefix2)
        if current_prefix is not None and prefix != current_prefix:
            row_len = write_candidate_group(
                candidate_handle,
                candidates,
                limit,
                compact_count_records,
            )
            candidate_count += row_len
            write_trigram_row(row_handle, current_prefix[0], current_prefix[1], row_start, row_len)
            row_count += 1
            row_start = candidate_count
            candidates = []
        current_prefix = prefix
        candidates.append(
            (
                token_id,
                count,
                scorer.score(
                    token_id,
                    count,
                    total,
                    order=3,
                    context=(prefix1, prefix2),
                ),
            )
        )

    if current_prefix is not None:
        row_len = write_candidate_group(candidate_handle, candidates, limit, compact_count_records)
        candidate_count += row_len
        write_trigram_row(row_handle, current_prefix[0], current_prefix[1], row_start, row_len)
        row_count += 1

    return row_count, candidate_count


def write_grouped_fourgram_stream(
    rows: Iterable[tuple[int, int, int, int, int, int]],
    row_handle,
    candidate_handle,
    scorer: CandidateScorer,
    limit: int,
    candidate_count: int,
    compact_count_records: bool,
) -> tuple[int, int]:
    row_count = 0
    current_prefix: tuple[int, int, int] | None = None
    row_start = candidate_count
    candidates: list[tuple[int, int, int]] = []

    for prefix1, prefix2, prefix3, token_id, count, total in rows:
        prefix = (prefix1, prefix2, prefix3)
        if current_prefix is not None and prefix != current_prefix:
            row_len = write_candidate_group(
                candidate_handle,
                candidates,
                limit,
                compact_count_records,
            )
            candidate_count += row_len
            write_fourgram_row(
                row_handle,
                current_prefix[0],
                current_prefix[1],
                current_prefix[2],
                row_start,
                row_len,
            )
            row_count += 1
            row_start = candidate_count
            candidates = []
        current_prefix = prefix
        candidates.append(
            (
                token_id,
                count,
                scorer.score(
                    token_id,
                    count,
                    total,
                    order=4,
                    context=(prefix1, prefix2, prefix3),
                ),
            )
        )

    if current_prefix is not None:
        row_len = write_candidate_group(candidate_handle, candidates, limit, compact_count_records)
        candidate_count += row_len
        write_fourgram_row(
            row_handle,
            current_prefix[0],
            current_prefix[1],
            current_prefix[2],
            row_start,
            row_len,
        )
        row_count += 1

    return row_count, candidate_count


def write_candidate_group(
    handle,
    candidates: list[tuple[int, int, int]],
    limit: int,
    compact_count_records: bool,
) -> int:
    candidates.sort(key=candidate_sort_key)
    candidates = candidates[:limit]
    for token_id, count, score in candidates:
        write_candidate_record(handle, token_id, count, score, compact_count_records)
    return len(candidates)


def write_header(
    handle,
    version: int,
    vocab_size: int,
    token_index_count: int,
    unigram_count: int,
    bigram_rows: int,
    trigram_rows: int,
    fourgram_rows: int,
    candidate_count: int,
    token_bytes_len: int,
    vocab_fingerprint: int,
    compact_count_records: bool,
) -> None:
    is_v2_or_newer = version >= VERSION_V2
    handle.write(MAGIC_V3 if version >= VERSION_V3 else MAGIC_V2 if is_v2_or_newer else MAGIC)
    write_u32_to_file(handle, VERSION_V3 if version >= VERSION_V3 else VERSION_V2 if is_v2_or_newer else VERSION)
    write_u32_to_file(handle, vocab_size)
    write_u32_to_file(handle, token_index_count)
    write_u32_to_file(handle, unigram_count)
    write_u32_to_file(handle, bigram_rows)
    write_u32_to_file(handle, trigram_rows)
    if is_v2_or_newer:
        write_u32_to_file(handle, fourgram_rows)
    write_u32_to_file(handle, candidate_count)
    write_u32_to_file(handle, token_bytes_len)
    write_u32_to_file(handle, vocab_fingerprint)
    if version >= VERSION_V3:
        write_u32_to_file(
            handle,
            COUNT_CANDIDATE_RECORD_LEN if compact_count_records else CANDIDATE_RECORD_LEN,
        )


def write_bigram_row(handle, prefix: int, start: int, length: int) -> None:
    write_u32_to_file(handle, prefix)
    write_u32_to_file(handle, start)
    write_u32_to_file(handle, length)


def write_trigram_row(handle, prefix1: int, prefix2: int, start: int, length: int) -> None:
    write_u32_to_file(handle, prefix1)
    write_u32_to_file(handle, prefix2)
    write_u32_to_file(handle, start)
    write_u32_to_file(handle, length)


def write_fourgram_row(
    handle,
    prefix1: int,
    prefix2: int,
    prefix3: int,
    start: int,
    length: int,
) -> None:
    write_u32_to_file(handle, prefix1)
    write_u32_to_file(handle, prefix2)
    write_u32_to_file(handle, prefix3)
    write_u32_to_file(handle, start)
    write_u32_to_file(handle, length)


def write_candidate_record(
    handle,
    token_id: int,
    count: int,
    score: int,
    compact_count_records: bool,
) -> None:
    write_u32_to_file(handle, token_id)
    write_u32_to_file(handle, count)
    if not compact_count_records:
        write_i32_to_file(handle, score)


def append_file(output_handle, path: Path) -> None:
    with path.open("rb") as input_handle:
        while chunk := input_handle.read(1024 * 1024):
            output_handle.write(chunk)


def write_u32(output: bytearray, value: int) -> None:
    output.extend(U32.pack(value))


def write_i32(output: bytearray, value: int) -> None:
    output.extend(I32.pack(value))


def write_candidate_record_to_buffer(
    output: bytearray,
    token_id: int,
    count: int,
    score: int,
    compact_count_records: bool,
) -> None:
    write_u32(output, token_id)
    write_u32(output, count)
    if not compact_count_records:
        write_i32(output, score)


def write_u32_to_file(output, value: int) -> None:
    output.write(U32.pack(value))


def write_i32_to_file(output, value: int) -> None:
    output.write(I32.pack(value))


def manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--vocab", type=Path, default=Path("data/autosuggest/models/ngram/vocab.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/autosuggest/models/ngram/autosuggest-ngram.bin"))
    parser.add_argument("--backend", choices=("sqlite", "memory"), default="sqlite")
    parser.add_argument("--sqlite-path", type=Path, default=Path("data/autosuggest/models/ngram/autosuggest-ngram.sqlite"))
    parser.add_argument("--reuse-sqlite", action="store_true")
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument(
        "--source-weight",
        action="append",
        dest="source_weights",
        metavar="SOURCE=WEIGHT",
        help="Positive integer source count weight, repeatable. Unlisted sources use weight 1.",
    )
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--skip-sentences-per-source", type=int, default=0)
    parser.add_argument("--max-sentences-per-source", type=int)
    parser.add_argument("--log-every-sentences", type=int, default=250_000)
    parser.add_argument("--max-candidates-per-prefix", type=int, default=8)
    parser.add_argument("--unigram-size", type=int, default=2048)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--bigram-min-count", type=int)
    parser.add_argument("--trigram-min-count", type=int)
    parser.add_argument("--fourgram-min-count", type=int)
    parser.add_argument(
        "--max-context-order",
        type=int,
        choices=(2, 3),
        default=2,
        help="Maximum previous-token context to encode: 2 emits v1 trigram artifacts, 3 emits v2 fourgram artifacts.",
    )
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--smoothing", type=float, default=64.0)
    parser.add_argument("--backoff-alpha", type=float, default=0.4)
    parser.add_argument("--kneser-ney-discount", type=float, default=0.75)
    parser.add_argument(
        "--score-mode",
        choices=(
            "count",
            "smoothed-log",
            "stupid-backoff",
            "kneser-ney",
            "modified-kneser-ney",
        ),
        default="count",
    )
    parser.add_argument(
        "--compact-count-records",
        action="store_true",
        help="Emit v3 count-only 8-byte candidate records; valid only with --score-mode count.",
    )
    args = parser.parse_args()

    report = build_ngram_lm(
        corpus_dir=args.corpus_dir,
        vocab_path=args.vocab,
        output=args.output,
        backend=args.backend,
        sqlite_path=args.sqlite_path,
        sources=set(args.sources) if args.sources else None,
        source_weights=SourceWeights.from_cli(args.source_weights),
        max_sentences=args.max_sentences,
        skip_sentences_per_source=args.skip_sentences_per_source,
        max_sentences_per_source=args.max_sentences_per_source,
        reuse_sqlite=args.reuse_sqlite,
        log_every_sentences=args.log_every_sentences,
        max_candidates_per_prefix=args.max_candidates_per_prefix,
        unigram_size=args.unigram_size,
        min_count=args.min_count,
        bigram_min_count=args.bigram_min_count,
        trigram_min_count=args.trigram_min_count,
        fourgram_min_count=args.fourgram_min_count,
        batch_size=args.batch_size,
        smoothing=args.smoothing,
        backoff_alpha=args.backoff_alpha,
        kneser_ney_discount=args.kneser_ney_discount,
        score_mode=args.score_mode,
        max_context_order=args.max_context_order,
        compact_count_records=args.compact_count_records,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
