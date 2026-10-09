#!/usr/bin/env python3
"""Train a compact candidate-pool reranker for Obadh autosuggest.

The runtime contract is intentionally narrow: the static n-gram artifact
retrieves a small candidate pool, and this model may only rerank that pool. It
never performs a full-vocabulary softmax and cannot invent words outside the
retrieved candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from tools.autosuggest.common import BOS_ID, PAD_ID, UNK_ID
from tools.corpus.provenance import neural_training_provenance, sha256_file
from tools.autosuggest.eval_ngram_lm import (
    Candidate,
    NgramLm,
    candidate_sort_key,
    iter_eval_sentence_tokens,
    model_recent_context,
)


SOURCE_PAD = 0
SOURCE_UNIGRAM = 1
SOURCE_BIGRAM = 2
SOURCE_TRIGRAM = 3
SOURCE_FOURGRAM = 4
SOURCE_COUNT = 5
DEFAULT_ALPHA_SWEEP = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class RankedCandidate:
    token_id: int
    count: int
    score: int
    source: int

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return candidate_sort_key(Candidate(self.token_id, self.count, self.score))


@dataclass
class ExampleSet:
    contexts: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_sources: torch.Tensor
    candidate_counts: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_ranks: torch.Tensor
    labels: torch.Tensor
    total_targets: int
    eligible_targets: int
    candidate_hits: int
    examples_by_source: dict[str, int]
    scanned_sentences_by_source: dict[str, int]

    @property
    def size(self) -> int:
        return int(self.labels.numel())


class CandidatePoolReranker(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pool_size: int,
        context_len: int,
        embedding_dim: int,
        source_dim: int,
        hidden_dim: int,
        dropout: float,
        architecture: str,
        transformer_layers: int,
        transformer_heads: int,
    ) -> None:
        super().__init__()
        if architecture not in ("mlp", "gru", "transformer"):
            raise ValueError(f"unsupported reranker architecture: {architecture}")
        if architecture == "transformer" and embedding_dim % transformer_heads != 0:
            raise ValueError("--embedding-dim must be divisible by --transformer-heads")
        self.pool_size = pool_size
        self.context_len = context_len
        self.architecture = architecture
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(context_len, embedding_dim)
        self.source_embedding = nn.Embedding(SOURCE_COUNT, source_dim, padding_idx=SOURCE_PAD)
        if architecture == "mlp":
            self.context_encoder = nn.Sequential(
                nn.Linear(context_len * embedding_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
        elif architecture == "gru":
            self.context_encoder = nn.GRU(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
            )
            self.context_norm = nn.LayerNorm(embedding_dim)
        numeric_dim = 3
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 3 + source_dim + numeric_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        contexts: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_sources: torch.Tensor,
        candidate_counts: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_ranks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, pool_size = candidate_ids.shape
        context_vector = self.encode_context(contexts)
        candidate_embeddings = self.token_embedding(candidate_ids)
        source_embeddings = self.source_embedding(candidate_sources)
        context_expanded = context_vector.unsqueeze(1).expand(batch_size, pool_size, -1)
        interaction = context_expanded * candidate_embeddings
        numeric = torch.stack(
            (
                normalize_per_row(candidate_scores),
                torch.log1p(candidate_counts.clamp_min(0.0)) / 16.0,
                1.0 / (candidate_ranks + 1.0),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                context_expanded,
                candidate_embeddings,
                interaction,
                source_embeddings,
                numeric,
            ),
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)

    def encode_context(self, contexts: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(contexts)
        if self.architecture == "mlp":
            positions = torch.arange(self.context_len, device=contexts.device)
            context_embeddings = token_embeddings + self.position_embedding(positions)
            return self.context_encoder(context_embeddings.flatten(start_dim=1))
        if self.architecture == "gru":
            _, hidden = self.context_encoder(token_embeddings)
            return hidden[-1]

        positions = torch.arange(self.context_len, device=contexts.device)
        context_embeddings = token_embeddings + self.position_embedding(positions)
        encoded = self.context_encoder(context_embeddings)
        non_pad = (contexts != PAD_ID).unsqueeze(-1)
        lengths = non_pad.sum(dim=1).clamp_min(1)
        pooled = (encoded * non_pad).sum(dim=1) / lengths
        return self.context_norm(pooled)


def normalize_per_row(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True).clamp_min(1.0)
    return (values - mean) / std


def static_base_logits(pool_size: int, device: torch.device) -> torch.Tensor:
    return -torch.arange(pool_size, device=device, dtype=torch.float32)


def suggest_ranked_candidates(
    lm: NgramLm,
    context_ids: list[int],
    limit: int,
) -> list[RankedCandidate]:
    recent = model_recent_context(context_ids, max_context=lm.max_context_order)
    if lm.score_mode == "backoff":
        return suggest_ranked_candidates_backoff(lm, recent, limit)

    output: list[RankedCandidate] = []
    if len(recent) == 3:
        row = lm._find_fourgram_row(recent[0], recent[1], recent[2])
        if row:
            merge_ranked_candidates(lm, row[0], row[1], SOURCE_FOURGRAM, limit, output)
    if len(recent) >= 2:
        row = lm._find_trigram_row(recent[-2], recent[-1])
        if row:
            merge_ranked_candidates(lm, row[0], row[1], SOURCE_TRIGRAM, limit, output)
    if recent:
        row = lm._find_bigram_row(recent[-1])
        if row:
            merge_ranked_candidates(lm, row[0], row[1], SOURCE_BIGRAM, limit, output)
    for index in range(lm.unigram_count):
        offset = lm.unigrams_offset + index * lm.candidate_record_len
        if merge_ranked_candidate(
            RankedCandidate(*candidate_tuple(lm._candidate_at(offset)), SOURCE_UNIGRAM),
            limit,
            output,
        ):
            break
    return output


def suggest_ranked_candidates_backoff(
    lm: NgramLm,
    recent: list[int],
    limit: int,
) -> list[RankedCandidate]:
    output: list[RankedCandidate] = []
    seen: set[int] = set()
    if len(recent) == 3:
        row = lm._find_fourgram_row(recent[0], recent[1], recent[2])
        if row:
            append_ranked_candidates(lm, row[0], row[1], SOURCE_FOURGRAM, limit, seen, output)
    if len(recent) >= 2:
        row = lm._find_trigram_row(recent[-2], recent[-1])
        if row:
            append_ranked_candidates(lm, row[0], row[1], SOURCE_TRIGRAM, limit, seen, output)
    if len(output) < limit and recent:
        row = lm._find_bigram_row(recent[-1])
        if row:
            append_ranked_candidates(lm, row[0], row[1], SOURCE_BIGRAM, limit, seen, output)
    if len(output) < limit:
        for index in range(lm.unigram_count):
            if len(output) >= limit:
                break
            offset = lm.unigrams_offset + index * lm.candidate_record_len
            candidate = lm._candidate_at(offset)
            if candidate.token_id > UNK_ID and candidate.token_id not in seen:
                seen.add(candidate.token_id)
                output.append(RankedCandidate(*candidate_tuple(candidate), SOURCE_UNIGRAM))
    return output


def append_ranked_candidates(
    lm: NgramLm,
    start: int,
    length: int,
    source: int,
    limit: int,
    seen: set[int],
    output: list[RankedCandidate],
) -> None:
    for index in range(start, start + length):
        if len(output) >= limit:
            break
        offset = lm.candidates_offset + index * lm.candidate_record_len
        candidate = lm._candidate_at(offset)
        if candidate.token_id > UNK_ID and candidate.token_id not in seen:
            seen.add(candidate.token_id)
            output.append(RankedCandidate(*candidate_tuple(candidate), source))


def merge_ranked_candidates(
    lm: NgramLm,
    start: int,
    length: int,
    source: int,
    limit: int,
    output: list[RankedCandidate],
) -> None:
    for index in range(start, start + length):
        offset = lm.candidates_offset + index * lm.candidate_record_len
        if merge_ranked_candidate(
            RankedCandidate(*candidate_tuple(lm._candidate_at(offset)), source),
            limit,
            output,
        ):
            break


def merge_ranked_candidate(
    candidate: RankedCandidate,
    limit: int,
    output: list[RankedCandidate],
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
        if candidate.sort_key < output[existing_index].sort_key:
            output.pop(existing_index)
        elif len(output) >= limit and candidate.sort_key >= output[-1].sort_key:
            return True
        else:
            return False
    elif len(output) >= limit and candidate.sort_key >= output[-1].sort_key:
        return True
    elif len(output) >= limit:
        output.pop()

    insert_at = len(output)
    for index, item in enumerate(output):
        if candidate.sort_key < item.sort_key:
            insert_at = index
            break
    output.insert(insert_at, candidate)
    return False


def candidate_tuple(candidate: Candidate) -> tuple[int, int, int]:
    return (candidate.token_id, candidate.count, candidate.score)


def collect_examples(
    lm: NgramLm,
    corpus_dir: Path,
    sources: set[str] | None,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
    max_examples_per_source: int | None,
    pool_size: int,
    context_window: int,
    seed: int,
    log_every_targets: int,
) -> ExampleSet:
    random.seed(seed)
    contexts: list[list[int]] = []
    candidate_ids: list[list[int]] = []
    candidate_sources: list[list[int]] = []
    candidate_counts: list[list[float]] = []
    candidate_scores: list[list[float]] = []
    candidate_ranks: list[list[float]] = []
    labels: list[int] = []
    total_targets = 0
    eligible_targets = 0
    examples_by_source: Counter[str] = Counter()
    scanned_sentences_by_source: Counter[str] = Counter()
    started_at = time.time()

    for source, tokens in iter_eval_sentence_tokens(
        corpus_dir,
        sources=sources,
        skip_sentences_per_source=skip_sentences_per_source,
        max_sentences_per_source=max_sentences_per_source,
    ):
        if (
            max_examples_per_source is not None
            and examples_by_source[source] >= max_examples_per_source
        ):
            continue
        scanned_sentences_by_source[source] += 1
        encoded = [BOS_ID, *(lm.token_id(token) for token in tokens)]
        for index in range(1, len(encoded)):
            total_targets += 1
            target = encoded[index]
            if target <= UNK_ID:
                continue
            eligible_targets += 1
            candidates = suggest_ranked_candidates(lm, encoded[:index], pool_size)
            label = next(
                (
                    candidate_index
                    for candidate_index, candidate in enumerate(candidates)
                    if candidate.token_id == target
                ),
                -1,
            )
            if label < 0:
                continue
            append_example(
                lm,
                encoded[:index],
                candidates,
                label,
                pool_size,
                context_window,
                contexts,
                candidate_ids,
                candidate_sources,
                candidate_counts,
                candidate_scores,
                candidate_ranks,
                labels,
            )
            examples_by_source[source] += 1
            if (
                max_examples_per_source is not None
                and examples_by_source[source] >= max_examples_per_source
            ):
                break
        if log_every_targets > 0 and total_targets % log_every_targets < len(tokens):
            print(
                json.dumps(
                    {
                        "event": "reranker_collect_progress",
                        "targets": total_targets,
                        "eligible": eligible_targets,
                        "examples": len(labels),
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "examples_by_source": dict(sorted(examples_by_source.items())),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if max_examples_per_source is not None and sources:
            if all(examples_by_source[source_name] >= max_examples_per_source for source_name in sources):
                break

    return ExampleSet(
        contexts=torch.tensor(contexts, dtype=torch.long),
        candidate_ids=torch.tensor(candidate_ids, dtype=torch.long),
        candidate_sources=torch.tensor(candidate_sources, dtype=torch.long),
        candidate_counts=torch.tensor(candidate_counts, dtype=torch.float32),
        candidate_scores=torch.tensor(candidate_scores, dtype=torch.float32),
        candidate_ranks=torch.tensor(candidate_ranks, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        total_targets=total_targets,
        eligible_targets=eligible_targets,
        candidate_hits=len(labels),
        examples_by_source=dict(sorted(examples_by_source.items())),
        scanned_sentences_by_source=dict(sorted(scanned_sentences_by_source.items())),
    )


def append_example(
    lm: NgramLm,
    context_ids: list[int],
    candidates: list[RankedCandidate],
    label: int,
    pool_size: int,
    context_window: int,
    contexts: list[list[int]],
    candidate_ids: list[list[int]],
    candidate_sources: list[list[int]],
    candidate_counts: list[list[float]],
    candidate_scores: list[list[float]],
    candidate_ranks: list[list[float]],
    labels: list[int],
) -> None:
    recent = model_recent_context(context_ids, max_context=context_window)
    padded_context = ([PAD_ID] * context_window + recent)[-context_window :]
    contexts.append(padded_context)
    ids = [candidate.token_id for candidate in candidates[:pool_size]]
    sources = [candidate.source for candidate in candidates[:pool_size]]
    counts = [float(candidate.count) for candidate in candidates[:pool_size]]
    scores = [float(candidate.score) for candidate in candidates[:pool_size]]
    ranks = [float(index + 1) for index in range(len(ids))]
    pad_len = pool_size - len(ids)
    if pad_len > 0:
        ids.extend([PAD_ID] * pad_len)
        sources.extend([SOURCE_PAD] * pad_len)
        counts.extend([0.0] * pad_len)
        scores.extend([min(scores) if scores else 0.0] * pad_len)
        ranks.extend([float(pool_size + 1)] * pad_len)
    candidate_ids.append(ids)
    candidate_sources.append(sources)
    candidate_counts.append(counts)
    candidate_scores.append(scores)
    candidate_ranks.append(ranks)
    labels.append(label)


def train(
    model: CandidatePoolReranker,
    train_set: ExampleSet,
    eval_set: ExampleSet,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    delta_l2: float,
    train_locked_prefix: int,
    seed: int,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        started_at = time.time()
        model.train()
        permutation = torch.randperm(train_set.size, generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, train_set.size, batch_size):
            indexes = permutation[start : start + batch_size]
            batch = batch_to_device(train_set, indexes, device)
            optimizer.zero_grad(set_to_none=True)
            delta = model(
                batch["contexts"],
                batch["candidate_ids"],
                batch["candidate_sources"],
                batch["candidate_counts"],
                batch["candidate_scores"],
                batch["candidate_ranks"],
            )
            logits = static_base_logits(model.pool_size, device).unsqueeze(0) + delta
            loss = ranking_loss(logits, batch["labels"], train_locked_prefix)
            if delta_l2 > 0.0:
                loss = loss + delta_l2 * delta.square().mean()
            loss.backward()
            optimizer.step()
            batch_size_seen = int(indexes.numel())
            seen += batch_size_seen
            loss_sum += float(loss.detach().cpu()) * batch_size_seen
        epoch_report = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, seen),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "eval": evaluate_model(model, eval_set, device),
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, ensure_ascii=False), flush=True)
    return history


def ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    locked_prefix: int,
) -> torch.Tensor:
    if locked_prefix <= 0:
        return F.cross_entropy(logits, labels)
    if locked_prefix >= logits.shape[1]:
        raise ValueError("--train-locked-prefix must be smaller than --pool-size")

    tail_mask = labels >= locked_prefix
    if not bool(tail_mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits[tail_mask, locked_prefix:],
        labels[tail_mask] - locked_prefix,
    )


def batch_to_device(example_set: ExampleSet, indexes: torch.Tensor, device: torch.device) -> dict:
    return {
        "contexts": example_set.contexts[indexes].to(device),
        "candidate_ids": example_set.candidate_ids[indexes].to(device),
        "candidate_sources": example_set.candidate_sources[indexes].to(device),
        "candidate_counts": example_set.candidate_counts[indexes].to(device),
        "candidate_scores": example_set.candidate_scores[indexes].to(device),
        "candidate_ranks": example_set.candidate_ranks[indexes].to(device),
        "labels": example_set.labels[indexes].to(device),
    }


@torch.no_grad()
def evaluate_model(
    model: CandidatePoolReranker,
    example_set: ExampleSet,
    device: torch.device,
    batch_size: int = 8192,
    alphas: tuple[float, ...] = DEFAULT_ALPHA_SWEEP,
) -> dict:
    model.eval()
    alpha_hits = {
        alpha: {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "mrr": 0.0}
        for alpha in alphas
    }
    locked_top1_hits = {
        alpha: {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "mrr": 0.0}
        for alpha in alphas
    }
    baseline_hits = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "mrr": 0.0}
    base = static_base_logits(model.pool_size, device).unsqueeze(0)
    for start in range(0, example_set.size, batch_size):
        end = min(example_set.size, start + batch_size)
        indexes = torch.arange(start, end)
        batch = batch_to_device(example_set, indexes, device)
        labels = batch["labels"]
        baseline_ranks = labels + 1
        update_rank_metrics(baseline_hits, baseline_ranks)
        delta = model(
            batch["contexts"],
            batch["candidate_ids"],
            batch["candidate_sources"],
            batch["candidate_counts"],
            batch["candidate_scores"],
            batch["candidate_ranks"],
        )
        for alpha in alphas:
            logits = base + delta * alpha
            ranks = label_ranks(logits, labels)
            update_rank_metrics(alpha_hits[alpha], ranks)
            locked_ranks = label_ranks(lock_static_prefix(logits, prefix_len=1), labels)
            update_rank_metrics(locked_top1_hits[alpha], locked_ranks)

    baseline = finalize_rank_metrics(baseline_hits, example_set)
    scored = {
        str(alpha): finalize_rank_metrics(metrics, example_set)
        for alpha, metrics in alpha_hits.items()
    }
    locked_top1 = {
        str(alpha): finalize_rank_metrics(metrics, example_set)
        for alpha, metrics in locked_top1_hits.items()
    }
    best_alpha, best_metrics = best_profile(scored)
    best_locked_alpha, best_locked_metrics = best_profile(locked_top1)
    return {
        "baseline": baseline,
        "alphas": scored,
        "locked_top1_alphas": locked_top1,
        "best_alpha": float(best_alpha),
        "best": best_metrics,
        "best_locked_top1_alpha": float(best_locked_alpha),
        "best_locked_top1": best_locked_metrics,
        "top5_all_target_gain": best_metrics["top5_all_targets"] - baseline["top5_all_targets"],
        "mrr_all_target_gain": best_metrics["mrr_all_targets"] - baseline["mrr_all_targets"],
        "locked_top1_top5_all_target_gain": best_locked_metrics["top5_all_targets"]
        - baseline["top5_all_targets"],
        "locked_top1_mrr_all_target_gain": best_locked_metrics["mrr_all_targets"]
        - baseline["mrr_all_targets"],
    }


def label_ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    label_scores = logits.gather(1, labels.unsqueeze(1))
    return (logits > label_scores).sum(dim=1) + 1


def lock_static_prefix(logits: torch.Tensor, prefix_len: int) -> torch.Tensor:
    if prefix_len <= 0:
        return logits
    output = logits.clone()
    locked = torch.arange(prefix_len, device=logits.device, dtype=logits.dtype)
    output[:, :prefix_len] = 1_000_000.0 - locked
    return output


def best_profile(profiles: dict[str, dict]) -> tuple[str, dict]:
    return max(
        profiles.items(),
        key=lambda item: (
            item[1]["top5_all_targets"],
            item[1]["mrr_all_targets"],
            item[1]["top1_all_targets"],
        ),
    )


def update_rank_metrics(metrics: dict, ranks: torch.Tensor) -> None:
    ranks_cpu = ranks.detach().cpu()
    metrics["top1"] += int((ranks_cpu <= 1).sum())
    metrics["top3"] += int((ranks_cpu <= 3).sum())
    metrics["top5"] += int((ranks_cpu <= 5).sum())
    metrics["top10"] += int((ranks_cpu <= 10).sum())
    metrics["mrr"] += float((1.0 / ranks_cpu.float()).sum())


def finalize_rank_metrics(metrics: dict, example_set: ExampleSet) -> dict:
    size = max(1, example_set.size)
    total_targets = max(1, example_set.total_targets)
    result = {
        "top1_in_pool": metrics["top1"] / size,
        "top3_in_pool": metrics["top3"] / size,
        "top5_in_pool": metrics["top5"] / size,
        "top10_in_pool": metrics["top10"] / size,
        "mrr_in_pool": metrics["mrr"] / size,
        "top1_all_targets": metrics["top1"] / total_targets,
        "top3_all_targets": metrics["top3"] / total_targets,
        "top5_all_targets": metrics["top5"] / total_targets,
        "top10_all_targets": metrics["top10"] / total_targets,
        "mrr_all_targets": metrics["mrr"] / total_targets,
    }
    return result


def example_set_report(example_set: ExampleSet) -> dict:
    return {
        "total_targets": example_set.total_targets,
        "eligible_targets": example_set.eligible_targets,
        "candidate_hits": example_set.candidate_hits,
        "candidate_hit_ratio_all_targets": example_set.candidate_hits
        / max(1, example_set.total_targets),
        "candidate_hit_ratio_eligible": example_set.candidate_hits
        / max(1, example_set.eligible_targets),
        "examples_by_source": example_set.examples_by_source,
        "scanned_sentences_by_source": example_set.scanned_sentences_by_source,
    }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--validation-corpus-dir", type=Path)
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--context-window", type=int, default=16)
    parser.add_argument(
        "--architecture",
        choices=("mlp", "gru", "transformer"),
        default="gru",
    )
    parser.add_argument("--train-skip-sentences-per-source", type=int, default=0)
    parser.add_argument("--train-max-sentences-per-source", type=int, default=100_000)
    parser.add_argument("--train-max-examples-per-source", type=int, default=80_000)
    parser.add_argument("--eval-skip-sentences-per-source", type=int)
    parser.add_argument("--eval-max-sentences-per-source", type=int, default=25_000)
    parser.add_argument("--eval-max-examples-per-source", type=int, default=30_000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--source-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--delta-l2", type=float, default=1e-4)
    parser.add_argument("--train-locked-prefix", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-report", type=Path, default=Path("target/autosuggest-reranker-report.json"))
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--log-every-targets", type=int, default=250_000)
    args = parser.parse_args()

    data_provenance = neural_training_provenance(args.model, args.corpus_dir, args.validation_corpus_dir)
    if args.eval_skip_sentences_per_source is None:
        args.eval_skip_sentences_per_source = 0 if args.validation_corpus_dir else 100_000
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    lm = NgramLm(args.model)
    if args.context_window < lm.max_context_order:
        raise SystemExit("--context-window must be at least the artifact context order")
    if args.pool_size < 1:
        raise SystemExit("--pool-size must be at least 1")
    if args.train_locked_prefix < 0 or args.train_locked_prefix >= args.pool_size:
        raise SystemExit("--train-locked-prefix must be in [0, pool-size)")
    device = choose_device(args.device)
    sources = set(args.sources) if args.sources else None
    train_started = time.time()
    train_set = collect_examples(
        lm,
        args.corpus_dir,
        sources,
        args.train_skip_sentences_per_source,
        args.train_max_sentences_per_source,
        args.train_max_examples_per_source,
        args.pool_size,
        args.context_window,
        args.seed,
        args.log_every_targets,
    )
    eval_set = collect_examples(
        lm,
        args.validation_corpus_dir or args.corpus_dir,
        sources,
        args.eval_skip_sentences_per_source,
        args.eval_max_sentences_per_source,
        args.eval_max_examples_per_source,
        args.pool_size,
        args.context_window,
        args.seed + 1,
        args.log_every_targets,
    )
    model = CandidatePoolReranker(
        vocab_size=lm.vocab_size,
        pool_size=args.pool_size,
        context_len=args.context_window,
        embedding_dim=args.embedding_dim,
        source_dim=args.source_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        architecture=args.architecture,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
    )
    history = train(
        model,
        train_set,
        eval_set,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.delta_l2,
        args.train_locked_prefix,
        args.seed,
    )
    final_eval = evaluate_model(model, eval_set, device)
    report = {
        "data_provenance": data_provenance,
        "artifact": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "bytes": len(lm.bytes),
            "vocab_size": lm.vocab_size,
            "max_context_order": lm.max_context_order,
            "pool_size": args.pool_size,
            "reranker_context_window": args.context_window,
            "candidate_record_len": lm.candidate_record_len,
        },
        "architecture": args.architecture,
        "device": str(device),
        "parameter_count": parameter_count(model),
        "fp32_parameter_bytes": parameter_count(model) * 4,
        "train_locked_prefix": args.train_locked_prefix,
        "train_collection": example_set_report(train_set),
        "eval_collection": example_set_report(eval_set),
        "history": history,
        "final_eval": final_eval,
        "elapsed_seconds": round(time.time() - train_started, 3),
    }
    if args.output_checkpoint:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "config": {
                    "vocab_size": lm.vocab_size,
                    "pool_size": args.pool_size,
                    "context_len": args.context_window,
                    "embedding_dim": args.embedding_dim,
                    "source_dim": args.source_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "architecture": args.architecture,
                    "transformer_layers": args.transformer_layers,
                    "transformer_heads": args.transformer_heads,
                    "train_locked_prefix": args.train_locked_prefix,
                },
                "report": report,
            },
            args.output_checkpoint,
        )
        report["checkpoint"] = str(args.output_checkpoint)
    save_report(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
