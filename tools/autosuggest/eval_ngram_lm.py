#!/usr/bin/env python3
"""Evaluate Obadh n-gram autosuggest artifacts on held-out corpus rows."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tools.autosuggest.common import BOS_ID, PAD_ID, UNK_ID, sentence_paths
from tools.corpus.provenance import evaluation_provenance


MAGIC = b"OBAUTOSUGLM_V1\0\0"
MAGIC_V2 = b"OBAUTOSUGLM_V2\0\0"
MAGIC_V3 = b"OBAUTOSUGLM_V3\0\0"
VERSION = 1
VERSION_V2 = 2
VERSION_V3 = 3
HEADER = struct.Struct("<16sIIIIIIIII")
HEADER_V2 = struct.Struct("<16sIIIIIIIIII")
HEADER_V3 = struct.Struct("<16sIIIIIIIIIII")
U32 = struct.Struct("<I")
I32 = struct.Struct("<i")
ID_TOKEN_RECORD_LEN = 8
TOKEN_INDEX_RECORD_LEN = 12
CANDIDATE_RECORD_LEN = 12
COUNT_CANDIDATE_RECORD_LEN = 8
BIGRAM_ROW_LEN = 12
TRIGRAM_ROW_LEN = 16
FOURGRAM_ROW_LEN = 20


@dataclass(frozen=True)
class Candidate:
    token_id: int
    count: int
    score: int


def candidate_sort_key(candidate: Candidate) -> tuple[int, int, int]:
    return (-candidate.score, -candidate.count, candidate.token_id)


class NgramLm:
    def __init__(self, path: Path) -> None:
        self.bytes = path.read_bytes()
        magic = self.bytes[:16]
        if magic == MAGIC:
            (
                _,
                self.version,
                self.vocab_size,
                self.token_index_count,
                self.unigram_count,
                self.bigram_row_count,
                self.trigram_row_count,
                self.candidate_count,
                self.token_bytes_len,
                self.vocab_fingerprint,
            ) = HEADER.unpack_from(self.bytes, 0)
            self.fourgram_row_count = 0
            offset = HEADER.size
        elif magic == MAGIC_V2:
            (
                _,
                self.version,
                self.vocab_size,
                self.token_index_count,
                self.unigram_count,
                self.bigram_row_count,
                self.trigram_row_count,
                self.fourgram_row_count,
                self.candidate_count,
                self.token_bytes_len,
                self.vocab_fingerprint,
            ) = HEADER_V2.unpack_from(self.bytes, 0)
            offset = HEADER_V2.size
        elif magic == MAGIC_V3:
            (
                _,
                self.version,
                self.vocab_size,
                self.token_index_count,
                self.unigram_count,
                self.bigram_row_count,
                self.trigram_row_count,
                self.fourgram_row_count,
                self.candidate_count,
                self.token_bytes_len,
                self.vocab_fingerprint,
                self.candidate_record_len,
            ) = HEADER_V3.unpack_from(self.bytes, 0)
            offset = HEADER_V3.size
        else:
            raise ValueError(f"invalid artifact magic in {path}")
        if self.version not in (VERSION, VERSION_V2, VERSION_V3):
            raise ValueError(f"unsupported artifact version {self.version}")
        if self.version < VERSION_V3:
            self.candidate_record_len = CANDIDATE_RECORD_LEN
        if self.candidate_record_len not in (
            CANDIDATE_RECORD_LEN,
            COUNT_CANDIDATE_RECORD_LEN,
        ):
            raise ValueError(f"invalid candidate record length {self.candidate_record_len}")
        self.max_context_order = 3 if self.fourgram_row_count > 0 else 2

        self.id_tokens_offset = offset
        offset += self.vocab_size * ID_TOKEN_RECORD_LEN
        self.token_index_offset = offset
        offset += self.token_index_count * TOKEN_INDEX_RECORD_LEN
        self.unigrams_offset = offset
        offset += self.unigram_count * self.candidate_record_len
        self.bigram_rows_offset = offset
        offset += self.bigram_row_count * BIGRAM_ROW_LEN
        self.trigram_rows_offset = offset
        offset += self.trigram_row_count * TRIGRAM_ROW_LEN
        self.fourgram_rows_offset = offset
        offset += self.fourgram_row_count * FOURGRAM_ROW_LEN
        self.candidates_offset = offset
        offset += self.candidate_count * self.candidate_record_len
        self.token_bytes_offset = offset
        offset += self.token_bytes_len
        if offset != len(self.bytes):
            raise ValueError("artifact section sizes do not match file length")

        self.token_to_id = self._load_token_lookup()
        self.score_mode = self._detect_score_mode()

    def _load_token_lookup(self) -> dict[str, int]:
        lookup: dict[str, int] = {}
        for index in range(self.token_index_count):
            offset = self.token_index_offset + index * TOKEN_INDEX_RECORD_LEN
            token_offset = read_u32(self.bytes, offset)
            token_len = read_u32(self.bytes, offset + 4)
            token_id = read_u32(self.bytes, offset + 8)
            lookup[self._token_text(token_offset, token_len)] = token_id
        return lookup

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, UNK_ID)

    def token_text(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token id {token_id} is outside vocab")
        offset = self.id_tokens_offset + token_id * ID_TOKEN_RECORD_LEN
        token_offset = read_u32(self.bytes, offset)
        token_len = read_u32(self.bytes, offset + 4)
        return self._token_text(token_offset, token_len)

    def suggest_ids(
        self,
        context_ids: list[int],
        limit: int,
        backoff_policy: str = "full",
    ) -> list[int]:
        recent = model_recent_context(context_ids, max_context=self.max_context_order)

        if self.score_mode == "backoff":
            if backoff_policy == "reserved" and limit >= 16:
                return self._suggest_ids_reserved_backoff(recent, limit)
            return self._suggest_ids_backoff(recent, limit)

        output: list[Candidate] = []

        if len(recent) == 3:
            row = self._find_fourgram_row(recent[0], recent[1], recent[2])
            if row:
                self._merge(row[0], row[1], limit, output)
        if len(recent) >= 2:
            row = self._find_trigram_row(recent[-2], recent[-1])
            if row:
                self._merge(row[0], row[1], limit, output)
        if recent:
            row = self._find_bigram_row(recent[-1])
            if row:
                self._merge(row[0], row[1], limit, output)
        for index in range(self.unigram_count):
            offset = self.unigrams_offset + index * self.candidate_record_len
            if self._merge_candidate(self._candidate_at(offset), limit, output):
                break
        return [candidate.token_id for candidate in output]

    def _suggest_ids_backoff(self, recent: list[int], limit: int) -> list[int]:
        output: list[int] = []
        seen: set[int] = set()

        if len(recent) == 3:
            row = self._find_fourgram_row(recent[0], recent[1], recent[2])
            if row:
                self._append(row[0], row[1], limit, seen, output)
        if len(recent) >= 2:
            row = self._find_trigram_row(recent[-2], recent[-1])
            if row:
                self._append(row[0], row[1], limit, seen, output)
        if len(output) < limit and recent:
            row = self._find_bigram_row(recent[-1])
            if row:
                self._append(row[0], row[1], limit, seen, output)
        if len(output) < limit:
            for index in range(self.unigram_count):
                if len(output) >= limit:
                    break
                offset = self.unigrams_offset + index * self.candidate_record_len
                token_id = read_u32(self.bytes, offset)
                if token_id > UNK_ID and token_id not in seen:
                    seen.add(token_id)
                    output.append(token_id)
        return output

    def _suggest_ids_reserved_backoff(self, recent: list[int], limit: int) -> list[int]:
        output: list[int] = []
        seen: set[int] = set()
        caps = reserved_backoff_caps(len(recent), limit)
        cap_index = 0

        if len(recent) == 3:
            row = self._find_fourgram_row(recent[0], recent[1], recent[2])
            if row:
                self._append(row[0], row[1], caps[cap_index], seen, output)
            cap_index += 1
        if len(recent) >= 2:
            row = self._find_trigram_row(recent[-2], recent[-1])
            if row:
                self._append(row[0], row[1], caps[cap_index], seen, output)
            cap_index += 1
        if recent:
            row = self._find_bigram_row(recent[-1])
            if row:
                self._append(row[0], row[1], caps[cap_index], seen, output)
            cap_index += 1
        if len(output) < limit:
            for index in range(self.unigram_count):
                if len(output) >= limit:
                    break
                offset = self.unigrams_offset + index * self.candidate_record_len
                token_id = read_u32(self.bytes, offset)
                if token_id > UNK_ID and token_id not in seen:
                    seen.add(token_id)
                    output.append(token_id)
        return output

    def _find_bigram_row(self, prefix: int) -> tuple[int, int] | None:
        low = 0
        high = self.bigram_row_count
        while low < high:
            mid = low + (high - low) // 2
            offset = self.bigram_rows_offset + mid * BIGRAM_ROW_LEN
            row_prefix = read_u32(self.bytes, offset)
            if row_prefix < prefix:
                low = mid + 1
            elif row_prefix > prefix:
                high = mid
            else:
                return read_u32(self.bytes, offset + 4), read_u32(self.bytes, offset + 8)
        return None

    def _find_trigram_row(self, prefix1: int, prefix2: int) -> tuple[int, int] | None:
        low = 0
        high = self.trigram_row_count
        target = (prefix1, prefix2)
        while low < high:
            mid = low + (high - low) // 2
            offset = self.trigram_rows_offset + mid * TRIGRAM_ROW_LEN
            row = (read_u32(self.bytes, offset), read_u32(self.bytes, offset + 4))
            if row < target:
                low = mid + 1
            elif row > target:
                high = mid
            else:
                return read_u32(self.bytes, offset + 8), read_u32(self.bytes, offset + 12)
        return None

    def _find_fourgram_row(
        self,
        prefix1: int,
        prefix2: int,
        prefix3: int,
    ) -> tuple[int, int] | None:
        low = 0
        high = self.fourgram_row_count
        target = (prefix1, prefix2, prefix3)
        while low < high:
            mid = low + (high - low) // 2
            offset = self.fourgram_rows_offset + mid * FOURGRAM_ROW_LEN
            row = (
                read_u32(self.bytes, offset),
                read_u32(self.bytes, offset + 4),
                read_u32(self.bytes, offset + 8),
            )
            if row < target:
                low = mid + 1
            elif row > target:
                high = mid
            else:
                return read_u32(self.bytes, offset + 12), read_u32(self.bytes, offset + 16)
        return None

    def _merge(
        self,
        start: int,
        length: int,
        limit: int,
        output: list[Candidate],
    ) -> None:
        for index in range(start, start + length):
            offset = self.candidates_offset + index * self.candidate_record_len
            if self._merge_candidate(self._candidate_at(offset), limit, output):
                break

    def _append(
        self,
        start: int,
        length: int,
        limit: int,
        seen: set[int],
        output: list[int],
    ) -> None:
        for index in range(start, start + length):
            if len(output) >= limit:
                break
            offset = self.candidates_offset + index * self.candidate_record_len
            token_id = read_u32(self.bytes, offset)
            if token_id > UNK_ID and token_id not in seen:
                seen.add(token_id)
                output.append(token_id)

    def _merge_candidate(
        self,
        candidate: Candidate,
        limit: int,
        output: list[Candidate],
    ) -> bool:
        if candidate.token_id <= UNK_ID:
            return False

        existing_index = next(
            (
                index
                for index, item in enumerate(output)
                if item.token_id == candidate.token_id
            ),
            None,
        )
        if existing_index is not None:
            if candidate_sort_key(candidate) < candidate_sort_key(output[existing_index]):
                output.pop(existing_index)
            elif len(output) >= limit and candidate_sort_key(candidate) >= candidate_sort_key(output[-1]):
                return True
            else:
                return False
        elif len(output) >= limit and candidate_sort_key(candidate) >= candidate_sort_key(output[-1]):
            return True
        elif len(output) >= limit:
            output.pop()

        insert_at = len(output)
        for index, item in enumerate(output):
            if candidate_sort_key(candidate) < candidate_sort_key(item):
                insert_at = index
                break
        output.insert(insert_at, candidate)
        return False

    def _candidate_at(self, offset: int) -> Candidate:
        count = read_u32(self.bytes, offset + 4)
        return Candidate(
            token_id=read_u32(self.bytes, offset),
            count=count,
            score=count
            if self.candidate_record_len == COUNT_CANDIDATE_RECORD_LEN
            else read_i32(self.bytes, offset + 8),
        )

    def _detect_score_mode(self) -> str:
        if self.candidate_record_len == COUNT_CANDIDATE_RECORD_LEN:
            return "backoff"
        sample_len = min(self.unigram_count, 8)
        if sample_len == 0:
            return "interpolated"
        for index in range(sample_len):
            offset = self.unigrams_offset + index * self.candidate_record_len
            count = read_u32(self.bytes, offset + 4)
            score = read_i32(self.bytes, offset + 8)
            if score < 0 or score != count:
                return "interpolated"
        return "backoff"

    def _token_text(self, offset: int, length: int) -> str:
        start = self.token_bytes_offset + offset
        return self.bytes[start : start + length].decode("utf-8")


def model_recent_context(context_ids: list[int], max_context: int) -> list[int]:
    recent: list[int] = []
    at_sentence_start = True
    for token_id in context_ids:
        if token_id == BOS_ID:
            recent.clear()
            at_sentence_start = True
        elif token_id > UNK_ID:
            at_sentence_start = False
            recent.append(token_id)
            if len(recent) > max_context:
                recent.pop(0)
        elif token_id in (PAD_ID, UNK_ID):
            recent.clear()
            at_sentence_start = False

    if at_sentence_start and not recent:
        return [BOS_ID]
    return recent


def reserved_backoff_caps(context_len: int, limit: int) -> list[int]:
    """Cumulative caps that keep lower-order priors in scorer-sized pools."""

    if context_len >= 3:
        reserves = (max(8, limit // 4), max(4, limit // 8), max(2, limit // 16))
    elif context_len == 2:
        reserves = (max(8, limit // 4), max(2, limit // 16))
    elif context_len == 1:
        reserves = (max(2, limit // 8),)
    else:
        return [limit]

    caps: list[int] = []
    remaining_reserve = sum(reserves)
    for reserve in reserves:
        caps.append(max(1, limit - remaining_reserve))
        remaining_reserve -= reserve
    caps.append(limit)
    return caps


def evaluate(
    model: Path,
    corpus_dir: Path,
    sources: set[str] | None,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
    max_targets: int | None,
    top_k: int,
    backoff_policy: str,
    miss_samples: int = 0,
) -> dict:
    provenance = evaluation_provenance(model, corpus_dir)
    lm = NgramLm(model)
    report_ks = report_cutoffs(top_k)
    total_targets = 0
    total = 0
    skipped_unknown_target = 0
    hits = Counter()
    reciprocal_rank_sum = 0.0
    per_source_targets = Counter()
    per_source_total = Counter()
    per_source_hits: dict[int, Counter] = {k: Counter() for k in report_ks}
    per_source_reciprocal_rank_sum = Counter()
    per_context_targets = Counter()
    per_context_total = Counter()
    per_context_hits: dict[int, Counter] = {k: Counter() for k in report_ks}
    per_context_reciprocal_rank_sum = Counter()
    missed_targets = Counter()

    for source, tokens in iter_eval_sentence_tokens(
        corpus_dir,
        sources=sources,
        skip_sentences_per_source=skip_sentences_per_source,
        max_sentences_per_source=max_sentences_per_source,
    ):
        encoded = [BOS_ID, *(lm.token_id(token) for token in tokens)]
        for index in range(1, len(encoded)):
            recent = model_recent_context(encoded[:index], max_context=lm.max_context_order)
            context_bucket = context_bucket_name(recent)
            total_targets += 1
            per_source_targets[source] += 1
            per_context_targets[context_bucket] += 1
            target = encoded[index]
            if target <= UNK_ID:
                skipped_unknown_target += 1
                continue
            candidates = lm.suggest_ids(encoded[:index], top_k, backoff_policy)
            total += 1
            per_source_total[source] += 1
            per_context_total[context_bucket] += 1
            try:
                rank = candidates.index(target) + 1
            except ValueError:
                rank = 0
            if rank:
                reciprocal_rank_sum += 1.0 / rank
                per_source_reciprocal_rank_sum[source] += 1.0 / rank
                per_context_reciprocal_rank_sum[context_bucket] += 1.0 / rank
            else:
                missed_targets[target] += 1
            for k in report_ks:
                if rank and rank <= min(k, top_k):
                    hits[k] += 1
                    per_source_hits[k][source] += 1
                    per_context_hits[k][context_bucket] += 1
            if max_targets is not None and total >= max_targets:
                return report(
                    lm,
                    total_targets,
                    total,
                    skipped_unknown_target,
                    hits,
                    reciprocal_rank_sum,
                    per_source_targets,
                    per_source_total,
                    per_source_hits,
                    per_source_reciprocal_rank_sum,
                    per_context_targets,
                    per_context_total,
                    per_context_hits,
                    per_context_reciprocal_rank_sum,
                    missed_targets,
                    top_k,
                    report_ks,
                    backoff_policy,
                    miss_samples,
                ) | {"evaluation_provenance": provenance}

    return report(
        lm,
        total_targets,
        total,
        skipped_unknown_target,
        hits,
        reciprocal_rank_sum,
        per_source_targets,
        per_source_total,
        per_source_hits,
        per_source_reciprocal_rank_sum,
        per_context_targets,
        per_context_total,
        per_context_hits,
        per_context_reciprocal_rank_sum,
        missed_targets,
        top_k,
        report_ks,
        backoff_policy,
        miss_samples,
    ) | {"evaluation_provenance": provenance}


def report(
    lm: NgramLm,
    total_targets: int,
    total: int,
    skipped_unknown_target: int,
    hits: Counter,
    reciprocal_rank_sum: float,
    per_source_targets: Counter,
    per_source_total: Counter,
    per_source_hits: dict[int, Counter],
    per_source_reciprocal_rank_sum: Counter,
    per_context_targets: Counter,
    per_context_total: Counter,
    per_context_hits: dict[int, Counter],
    per_context_reciprocal_rank_sum: Counter,
    missed_targets: Counter,
    top_k: int,
    report_ks: list[int],
    backoff_policy: str,
    miss_samples: int,
) -> dict:
    def ratio(value: int, denominator: int = total) -> float:
        return value / denominator if denominator else 0.0

    def all_ratio(value: int, denominator: int = total_targets) -> float:
        return value / denominator if denominator else 0.0

    pool_k = max(report_ks) if report_ks else 0
    pool_hits = hits[pool_k] if pool_k else 0
    result = {
        "artifact": {
            "version": lm.version,
            "bytes": len(lm.bytes),
            "vocab_size": lm.vocab_size,
            "vocab_fingerprint": lm.vocab_fingerprint,
            "unigram_count": lm.unigram_count,
            "bigram_rows": lm.bigram_row_count,
            "trigram_rows": lm.trigram_row_count,
            "fourgram_rows": lm.fourgram_row_count,
            "candidate_record_len": lm.candidate_record_len,
            "candidate_rows": lm.candidate_count,
        },
        "max_context_order": lm.max_context_order,
        "context_semantics": "bos_sentence_start_unknown_fallback",
        "backoff_policy": backoff_policy,
        "total_targets": total_targets,
        "eligible_targets": total,
        "skipped_unknown_targets": skipped_unknown_target,
        "skipped_unknown_ratio": all_ratio(skipped_unknown_target),
        "reported_k": report_ks,
        "candidate_pool_k": pool_k,
        "candidate_pool_hit_ratio_all_targets": all_ratio(pool_hits),
        "candidate_pool_hit_ratio": ratio(pool_hits),
        "candidate_pool_miss_ratio": ratio(max(0, total - pool_hits)),
        "mrr_all_targets": all_ratio(reciprocal_rank_sum),
        "mrr": reciprocal_rank_sum / total if total else 0.0,
        "per_source": {
            source: {
                "total_targets": per_source_targets[source],
                "eligible_targets": per_source_total[source],
                "skipped_unknown_targets": per_source_targets[source]
                - per_source_total[source],
                "mrr_all_targets": ratio(
                    per_source_reciprocal_rank_sum[source],
                    per_source_targets[source],
                ),
                "mrr": ratio(
                    per_source_reciprocal_rank_sum[source],
                    per_source_total[source],
                ),
                **{
                    f"top{k}_all_targets": ratio(
                        per_source_hits[k][source],
                        per_source_targets[source],
                    )
                    for k in report_ks
                },
                **{
                    f"top{k}": ratio(
                        per_source_hits[k][source],
                        per_source_total[source],
                    )
                    for k in report_ks
                },
            }
            for source in sorted(per_source_targets)
        },
        "per_context": {
            bucket: {
                "total_targets": per_context_targets[bucket],
                "eligible_targets": per_context_total[bucket],
                "skipped_unknown_targets": per_context_targets[bucket]
                - per_context_total[bucket],
                "mrr_all_targets": ratio(
                    per_context_reciprocal_rank_sum[bucket],
                    per_context_targets[bucket],
                ),
                "mrr": ratio(
                    per_context_reciprocal_rank_sum[bucket],
                    per_context_total[bucket],
                ),
                **{
                    f"top{k}_all_targets": ratio(
                        per_context_hits[k][bucket],
                        per_context_targets[bucket],
                    )
                    for k in report_ks
                },
                **{
                    f"top{k}": ratio(
                        per_context_hits[k][bucket],
                        per_context_total[bucket],
                    )
                    for k in report_ks
                },
            }
            for bucket in sorted(per_context_targets, key=context_bucket_sort_key)
        },
    }
    for k in report_ks:
        result[f"top{k}_all_targets"] = all_ratio(hits[k])
        result[f"top{k}"] = ratio(hits[k])
    if pool_k:
        for k in report_ks:
            if k == pool_k:
                continue
            headroom = max(0, pool_hits - hits[k])
            result[f"top{k}_to_top{pool_k}_headroom_all_targets"] = all_ratio(headroom)
            result[f"top{k}_to_top{pool_k}_headroom"] = ratio(headroom)
    if miss_samples > 0:
        result["top_missed_targets"] = [
            {
                "token_id": token_id,
                "token": lm.token_text(token_id),
                "misses": count,
            }
            for token_id, count in missed_targets.most_common(miss_samples)
        ]
    return result


def report_cutoffs(top_k: int) -> list[int]:
    return sorted({k for k in (1, 3, 5, 10, 20, top_k) if 1 <= k <= top_k})


def context_bucket_name(recent: list[int]) -> str:
    if recent == [BOS_ID]:
        return "sentence_start"
    if not recent:
        return "no_context"
    return f"context_{len(recent)}"


def context_bucket_sort_key(bucket: str) -> tuple[int, str]:
    if bucket == "sentence_start":
        return (0, bucket)
    if bucket == "no_context":
        return (1, bucket)
    if bucket.startswith("context_"):
        try:
            return (2 + int(bucket.removeprefix("context_")), bucket)
        except ValueError:
            pass
    return (99, bucket)


def iter_eval_sentence_tokens(
    corpus_dir: Path,
    sources: set[str] | None,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
) -> Iterator[tuple[str, list[str]]]:
    emitted_by_source: Counter[str] = Counter()
    seen_by_source: Counter[str] = Counter()
    pending_sources = set(sources) if sources else None

    for path in sentence_paths(corpus_dir):
        source_hint = path.stem.split(".")[0]
        if sources is not None and source_hint not in sources:
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
                emitted_by_source[source] += 1
                yield source, tokens
                if (
                    pending_sources is not None
                    and max_sentences_per_source is not None
                    and emitted_by_source[source] >= max_sentences_per_source
                ):
                    pending_sources.discard(source)
                    if not pending_sources:
                        return
                    break


def read_u32(bytes_: bytes, offset: int) -> int:
    return U32.unpack_from(bytes_, offset)[0]


def read_i32(bytes_: bytes, offset: int) -> int:
    return I32.unpack_from(bytes_, offset)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--skip-sentences-per-source", type=int, default=0)
    parser.add_argument("--max-sentences-per-source", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--backoff-policy",
        choices=("full", "reserved"),
        default="full",
        help="Candidate collection policy for count/backoff artifacts.",
    )
    parser.add_argument(
        "--miss-samples",
        type=int,
        default=0,
        help="Include the N most frequent in-vocabulary targets missing from the candidate pool.",
    )
    args = parser.parse_args()

    result = evaluate(
        model=args.model,
        corpus_dir=args.corpus_dir,
        sources=set(args.sources) if args.sources else None,
        skip_sentences_per_source=args.skip_sentences_per_source,
        max_sentences_per_source=args.max_sentences_per_source,
        max_targets=args.max_targets,
        top_k=args.top_k,
        backoff_policy=args.backoff_policy,
        miss_samples=args.miss_samples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
