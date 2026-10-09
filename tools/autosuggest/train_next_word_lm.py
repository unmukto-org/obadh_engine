#!/usr/bin/env python3
"""Train a compact full-vocabulary next-word LM for Obadh autosuggest.

This is the neural gate for the next-word layer. Unlike the candidate reranker,
this model predicts directly over the bounded Obadh autosuggest vocabulary. The
deployment intent is still mobile-first: a short context window, tied input and
output embeddings, and a small recurrent/attention encoder that can later be
exported to an on-device runtime only if it clears the measured quality bar.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tools.autosuggest.common import BOS_ID, PAD_ID, UNK_ID
from tools.autosuggest.checkpoints import atomic_torch_save, capture_rng, restore_rng
from tools.corpus.provenance import (
    neural_training_provenance, sha256_file, verify_checkpoint_provenance,
)
from tools.autosuggest.eval_ngram_lm import (
    NgramLm,
    iter_eval_sentence_tokens,
    model_recent_context,
)


DEFAULT_REPORT_CUTOFFS = (1, 3, 5, 10)


@dataclass
class ExampleSet:
    contexts: torch.Tensor
    labels: torch.Tensor
    source_ids: torch.Tensor
    source_names: tuple[str, ...]
    total_targets: int
    eligible_targets: int
    examples_by_source: dict[str, int]
    scanned_sentences_by_source: dict[str, int]
    total_targets_by_source: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.labels.numel())


@dataclass
class CandidatePoolSet:
    rows: list[list[int]]
    ids: torch.Tensor
    hit_count: int


def empty_example_set() -> ExampleSet:
    return ExampleSet(
        contexts=torch.empty((0, 0), dtype=torch.long),
        labels=torch.empty((0,), dtype=torch.long),
        source_ids=torch.empty((0,), dtype=torch.long),
        source_names=(),
        total_targets=0,
        eligible_targets=0,
        examples_by_source={},
        scanned_sentences_by_source={},
    )


class NextWordLm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        embedding_dim: int,
        hidden_dim: int,
        architecture: str,
        dropout: float,
        transformer_layers: int,
        transformer_heads: int,
        transformer_padding_mode: str = "unmasked-v0",
        transformer_initialization: str = "legacy-v0",
    ) -> None:
        super().__init__()
        if architecture not in ("gru", "transformer"):
            raise ValueError(f"unsupported architecture: {architecture}")
        if architecture == "transformer" and embedding_dim % transformer_heads != 0:
            raise ValueError("--embedding-dim must be divisible by --transformer-heads")

        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embedding_dim = embedding_dim
        self.architecture = architecture
        if transformer_padding_mode not in ("unmasked-v0", "masked-v1"):
            raise ValueError("unsupported transformer padding contract")
        self.transformer_padding_mode = transformer_padding_mode
        if transformer_initialization not in ("legacy-v0", "independent-v1"):
            raise ValueError("unsupported transformer initialization contract")
        self.transformer_initialization = transformer_initialization
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(context_len, embedding_dim)
        if architecture == "gru":
            self.encoder = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            self.output_projection = (
                nn.Identity()
                if hidden_dim == embedding_dim
                else nn.Linear(hidden_dim, embedding_dim, bias=False)
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
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False,
            )
            self.output_projection = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()
        if architecture == "transformer" and transformer_initialization == "independent-v1":
            # Keep word/position scales comparable and independently initialize
            # cloned encoder layers. Checkpoints retain their initialization ID.
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
            for layer in self.encoder.layers:
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                nn.init.normal_(layer.self_attn.in_proj_weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.self_attn.in_proj_bias)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_context(contexts)
        return hidden @ self.token_embedding.weight.T + self.output_bias

    def encode_context(self, contexts: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(contexts)
        if self.architecture == "gru":
            _, hidden = self.encoder(self.dropout(embedded))
            return self.output_projection(hidden[-1])

        positions = torch.arange(self.context_len, device=contexts.device)
        non_pad = contexts != PAD_ID
        # An unknown preceding word can leave no known context. Keep the final
        # PAD slot visible in that case: fully masked attention produces NaNs.
        # Real context always masks left padding, independent of the host adapter.
        empty = torch.logical_not(non_pad.any(dim=1, keepdim=True))
        visible = torch.logical_or(non_pad, torch.logical_and(empty, positions == self.context_len - 1))
        encoded = self.encoder(
            self.dropout(embedded + self.position_embedding(positions)),
            src_key_padding_mask=torch.logical_not(visible) if self.transformer_padding_mode == "masked-v1" else None,
        )
        last_positions = contexts.shape[1] - 1 - non_pad.flip(dims=(1,)).int().argmax(dim=1)
        batch_positions = torch.arange(contexts.shape[0], device=contexts.device)
        return self.output_projection(encoded[batch_positions, last_positions])


def collect_examples(
    lm: NgramLm,
    corpus_dir: Path,
    sources: set[str] | None,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
    max_examples_per_source: int | None,
    context_window: int,
    log_every_targets: int,
) -> ExampleSet:
    contexts: list[list[int]] = []
    labels: list[int] = []
    source_ids: list[int] = []
    source_index: dict[str, int] = {}
    total_targets = 0
    eligible_targets = 0
    examples_by_source: Counter[str] = Counter()
    scanned_sentences_by_source: Counter[str] = Counter()
    total_targets_by_source: Counter[str] = Counter()
    started_at = time.time()

    for source, tokens in iter_eval_sentence_tokens(
        corpus_dir,
        sources=sources,
        skip_sentences_per_source=skip_sentences_per_source,
        max_sentences_per_source=max_sentences_per_source,
    ):
        source_id = source_index.setdefault(source, len(source_index))
        if (
            max_examples_per_source is not None
            and examples_by_source[source] >= max_examples_per_source
        ):
            continue
        scanned_sentences_by_source[source] += 1
        encoded = [BOS_ID, *(lm.token_id(token) for token in tokens)]
        for index in range(1, len(encoded)):
            total_targets += 1
            total_targets_by_source[source] += 1
            target = encoded[index]
            if target <= UNK_ID:
                continue
            eligible_targets += 1
            recent = model_recent_context(encoded[:index], max_context=context_window)
            contexts.append(([PAD_ID] * context_window + recent)[-context_window:])
            labels.append(target)
            source_ids.append(source_id)
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
                        "event": "next_word_lm_collect_progress",
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
        labels=torch.tensor(labels, dtype=torch.long),
        source_ids=torch.tensor(source_ids, dtype=torch.long),
        source_names=tuple(
            source for source, _ in sorted(source_index.items(), key=lambda item: item[1])
        ),
        total_targets=total_targets,
        eligible_targets=eligible_targets,
        examples_by_source=dict(sorted(examples_by_source.items())),
        scanned_sentences_by_source=dict(sorted(scanned_sentences_by_source.items())),
        total_targets_by_source=dict(sorted(total_targets_by_source.items())),
    )


def initialize_output_bias_from_unigrams(
    model: NextWordLm,
    lm: NgramLm,
    floor_count: float = 0.1,
) -> None:
    counts = torch.full((lm.vocab_size,), floor_count, dtype=torch.float32)
    counts[: UNK_ID + 1] = floor_count
    for index in range(lm.unigram_count):
        offset = lm.unigrams_offset + index * lm.candidate_record_len
        candidate = lm._candidate_at(offset)
        if candidate.token_id > UNK_ID:
            counts[candidate.token_id] = max(float(candidate.count), floor_count)
    probabilities = counts / counts.sum()
    with torch.no_grad():
        model.output_bias.copy_(probabilities.log())


def train(
    model: NextWordLm,
    optimizer: torch.optim.Optimizer,
    train_set: ExampleSet,
    eval_set: ExampleSet,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    start_epoch: int,
    loss_mode: str,
    teacher_model: NextWordLm | None,
    distill_alpha: float,
    distill_temperature: float,
    distill_top_k: int,
    batch_sampling: str,
    *,
    progress: dict | None = None,
    checkpoint_callback=None,
    epoch_callback=None,
    checkpoint_seconds: float = 300,
    max_steps: int | None = None,
    max_training_seconds: float | None = None,
    stop_at: float | None = None,
    max_grad_norm: float | None = None,
) -> list[dict]:
    model.to(device)
    if teacher_model is not None:
        teacher_model.to(device)
        teacher_model.eval()
    if train_set.size == 0:
        raise ValueError("training collection is empty")
    progress = {} if progress is None else progress
    progress.setdefault("completed_epochs", start_epoch)
    progress.setdefault("next_batch", 0)
    progress.setdefault("optimizer_steps", 0)
    progress["status"] = "running"
    run_started = time.monotonic()
    last_checkpoint = run_started
    invocation_steps = 0
    history = []
    if checkpoint_callback:
        checkpoint_callback(progress)
    for epoch in range(1, epochs + 1):
        absolute_epoch = start_epoch + epoch
        # Reconstruct exactly the same batch order when resuming within an epoch.
        # This generator is independent of global dropout/model RNG state.
        generator = torch.Generator().manual_seed(seed + absolute_epoch)
        skip_batches = progress["next_batch"] if epoch == 1 else 0
        started_at = time.time()
        model.train()
        loss_sum = progress.get("loss_sum", 0.0) if skip_batches else 0.0
        seen = progress.get("seen", 0) if skip_batches else 0
        for batch_index, indexes in enumerate(iter_epoch_batches(
            train_set,
            batch_size,
            generator,
            batch_sampling,
        )):
            if batch_index < skip_batches:
                continue
            if stop_at is not None and time.time() >= stop_at:
                progress["status"] = "deadline"
                if checkpoint_callback:
                    checkpoint_callback(progress)
                return history
            contexts = train_set.contexts[indexes].to(device)
            labels = train_set.labels[indexes].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(contexts)
            teacher_logits = None
            teacher_top_values = None
            teacher_top_indices = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(contexts)
                    if distill_top_k > 0:
                        teacher_top_values, teacher_top_indices = torch.topk(
                            teacher_logits,
                            k=min(distill_top_k, teacher_logits.shape[1]),
                            dim=1,
                        )
                        teacher_logits = None
            loss = batch_loss(
                logits,
                labels,
                train_set.source_ids[indexes].to(device),
                loss_mode,
                teacher_logits,
                teacher_top_values,
                teacher_top_indices,
                distill_alpha,
                distill_temperature,
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss; last valid checkpoint retained")
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            batch_seen = int(indexes.numel())
            seen += batch_seen
            loss_sum += float(loss.detach().cpu()) * batch_seen
            invocation_steps += 1
            progress.update(completed_epochs=absolute_epoch - 1, next_batch=batch_index + 1,
                            optimizer_steps=progress["optimizer_steps"] + 1, loss_sum=loss_sum, seen=seen)
            now = time.monotonic()
            if max_steps is not None and invocation_steps >= max_steps:
                progress["status"] = "step_limit"
            elif max_training_seconds is not None and now - run_started >= max_training_seconds:
                progress["status"] = "time_limit"
            elif stop_at is not None and time.time() >= stop_at:
                progress["status"] = "deadline"
            if checkpoint_callback and (now - last_checkpoint >= checkpoint_seconds or progress["status"] != "running"):
                checkpoint_callback(progress)
                last_checkpoint = time.monotonic()
            if progress["optimizer_steps"] % 100 == 0:
                print(json.dumps({"event": "training_progress", "epoch": absolute_epoch,
                                  "step": progress["optimizer_steps"], "loss": loss_sum / max(1, seen),
                                  "elapsed_seconds": round(now - run_started, 3)}), flush=True)
            if progress["status"] != "running":
                return history
        progress.update(completed_epochs=absolute_epoch, next_batch=0, loss_sum=0.0, seen=0)
        if checkpoint_callback:
            checkpoint_callback(progress)  # Survive preemption during validation too.
            last_checkpoint = time.monotonic()
        epoch_report = {
            "epoch": absolute_epoch,
            "train_loss": loss_sum / max(1, seen),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "eval": evaluate_model(model, eval_set, device),
            "eval_by_source": evaluate_model_by_source(model, eval_set, device),
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, ensure_ascii=False), flush=True)
        if epoch_callback:
            epoch_callback(progress, epoch_report)
    progress["status"] = "completed"
    if checkpoint_callback:
        checkpoint_callback(progress)
    return history


def iter_epoch_batches(
    train_set: ExampleSet,
    batch_size: int,
    generator: torch.Generator,
    sampling_mode: str,
):
    if sampling_mode == "token":
        permutation = torch.randperm(train_set.size, generator=generator)
        for start in range(0, train_set.size, batch_size):
            yield permutation[start : start + batch_size]
        return
    if sampling_mode != "source-balanced":
        raise ValueError(f"unsupported batch sampling mode: {sampling_mode}")

    source_values = torch.unique(train_set.source_ids).tolist()
    source_values = [int(source_id) for source_id in source_values]
    if not source_values:
        return
    source_indexes = {
        source_id: torch.nonzero(
            train_set.source_ids == source_id,
            as_tuple=False,
        ).flatten()
        for source_id in source_values
    }
    source_indexes = {
        source_id: indexes
        for source_id, indexes in source_indexes.items()
        if indexes.numel() > 0
    }
    if not source_indexes:
        return

    source_count = len(source_indexes)
    if batch_size < source_count:
        raise ValueError(
            "source-balanced batch sampling requires batch_size >= source count"
        )
    per_source = max(1, batch_size // source_count)
    remainder = max(0, batch_size - per_source * source_count)
    largest_source = max(indexes.numel() for indexes in source_indexes.values())
    batches_per_epoch = max(1, (largest_source + per_source - 1) // per_source)
    shuffled = {
        source_id: indexes[torch.randperm(indexes.numel(), generator=generator)]
        for source_id, indexes in source_indexes.items()
    }
    cursors = {source_id: 0 for source_id in source_indexes}
    ordered_sources = sorted(source_indexes)

    for _ in range(batches_per_epoch):
        pieces = []
        for offset, source_id in enumerate(ordered_sources):
            take = per_source + (1 if offset < remainder else 0)
            pieces.append(
                take_source_indexes(
                    source_id,
                    take,
                    shuffled,
                    cursors,
                    generator,
                )
            )
        batch = torch.cat(pieces)
        yield batch[torch.randperm(batch.numel(), generator=generator)]


def take_source_indexes(
    source_id: int,
    take: int,
    shuffled: dict[int, torch.Tensor],
    cursors: dict[int, int],
    generator: torch.Generator,
) -> torch.Tensor:
    pieces = []
    remaining = take
    while remaining > 0:
        indexes = shuffled[source_id]
        cursor = cursors[source_id]
        if cursor >= indexes.numel():
            shuffled[source_id] = indexes[torch.randperm(indexes.numel(), generator=generator)]
            indexes = shuffled[source_id]
            cursor = 0
        chunk_len = min(remaining, int(indexes.numel()) - cursor)
        pieces.append(indexes[cursor : cursor + chunk_len])
        cursor += chunk_len
        remaining -= chunk_len
        cursors[source_id] = cursor
    return torch.cat(pieces)


def batch_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    source_ids: torch.Tensor,
    loss_mode: str,
    teacher_logits: torch.Tensor | None = None,
    teacher_top_values: torch.Tensor | None = None,
    teacher_top_indices: torch.Tensor | None = None,
    distill_alpha: float = 1.0,
    distill_temperature: float = 1.0,
) -> torch.Tensor:
    per_example_loss = F.cross_entropy(logits, labels, reduction="none")
    if teacher_logits is not None and distill_alpha < 1.0:
        temperature = max(float(distill_temperature), 1.0e-6)
        distill_loss = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="none",
        ).sum(dim=1) * temperature * temperature
        per_example_loss = distill_alpha * per_example_loss + (
            1.0 - distill_alpha
        ) * distill_loss
    elif teacher_top_values is not None and teacher_top_indices is not None and distill_alpha < 1.0:
        temperature = max(float(distill_temperature), 1.0e-6)
        student_top_logits = logits.gather(1, teacher_top_indices)
        distill_loss = F.kl_div(
            F.log_softmax(student_top_logits / temperature, dim=1),
            F.softmax(teacher_top_values / temperature, dim=1),
            reduction="none",
        ).sum(dim=1) * temperature * temperature
        per_example_loss = distill_alpha * per_example_loss + (
            1.0 - distill_alpha
        ) * distill_loss
    if loss_mode == "token":
        return per_example_loss.mean()
    if loss_mode != "source-balanced":
        raise ValueError(f"unsupported loss mode: {loss_mode}")
    source_means = [
        per_example_loss[source_ids == source_id].mean()
        for source_id in torch.unique(source_ids)
    ]
    return torch.stack(source_means).mean()


@torch.no_grad()
def evaluate_model(
    model: NextWordLm,
    example_set: ExampleSet,
    device: torch.device,
    batch_size: int = 2048,
    cutoffs: tuple[int, ...] = DEFAULT_REPORT_CUTOFFS,
) -> dict:
    model.eval()
    hits = Counter()
    reciprocal_rank_sum = 0.0
    for start in range(0, example_set.size, batch_size):
        end = min(example_set.size, start + batch_size)
        contexts = example_set.contexts[start:end].to(device)
        labels = example_set.labels[start:end].to(device)
        logits = model(contexts)
        ranks = label_ranks(logits, labels)
        ranks_cpu = ranks.detach().cpu()
        reciprocal_rank_sum += float((1.0 / ranks_cpu.float()).sum())
        for cutoff in cutoffs:
            hits[cutoff] += int((ranks_cpu <= cutoff).sum())
    return rank_report(hits, reciprocal_rank_sum, example_set, cutoffs)


@torch.no_grad()
def evaluate_model_by_source(
    model: NextWordLm,
    example_set: ExampleSet,
    device: torch.device,
    batch_size: int = 2048,
    cutoffs: tuple[int, ...] = DEFAULT_REPORT_CUTOFFS,
) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for source_id, source_name in enumerate(example_set.source_names):
        indexes = torch.nonzero(example_set.source_ids == source_id, as_tuple=False).flatten()
        if indexes.numel() == 0 and example_set.total_targets_by_source.get(source_name, 0) == 0:
            continue
        reports[source_name] = evaluate_model(
            model,
            subset_example_set(example_set, indexes, source_name=source_name),
            device,
            batch_size=batch_size,
            cutoffs=cutoffs,
        )
    return reports


@torch.no_grad()
def evaluate_ngram_baseline(
    lm: NgramLm,
    example_set: ExampleSet,
    top_k: int,
    cutoffs: tuple[int, ...] = DEFAULT_REPORT_CUTOFFS,
) -> dict:
    hits = Counter()
    reciprocal_rank_sum = 0.0
    max_context = lm.max_context_order
    for context_row, label in zip(example_set.contexts.tolist(), example_set.labels.tolist()):
        context_ids = [token_id for token_id in context_row if token_id != PAD_ID]
        candidates = lm.suggest_ids(context_ids[-max_context:], top_k)
        try:
            rank = candidates.index(label) + 1
        except ValueError:
            rank = 0
        if rank:
            reciprocal_rank_sum += 1.0 / rank
        for cutoff in cutoffs:
            if rank and rank <= min(cutoff, top_k):
                hits[cutoff] += 1
    return rank_report(hits, reciprocal_rank_sum, example_set, cutoffs)


def evaluate_ngram_baseline_by_source(
    lm: NgramLm,
    example_set: ExampleSet,
    top_k: int,
    cutoffs: tuple[int, ...] = DEFAULT_REPORT_CUTOFFS,
) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for source_id, source_name in enumerate(example_set.source_names):
        indexes = torch.nonzero(example_set.source_ids == source_id, as_tuple=False).flatten()
        if indexes.numel() == 0 and example_set.total_targets_by_source.get(source_name, 0) == 0:
            continue
        reports[source_name] = evaluate_ngram_baseline(
            lm,
            subset_example_set(example_set, indexes, source_name=source_name),
            top_k,
            cutoffs=cutoffs,
        )
    return reports


def collect_candidate_pool(lm: NgramLm, example_set: ExampleSet, pool_k: int) -> CandidatePoolSet:
    max_context = lm.max_context_order
    rows: list[list[int]] = []
    hit_count = 0
    for context_row, label in zip(example_set.contexts.tolist(), example_set.labels.tolist()):
        context_ids = [token_id for token_id in context_row if token_id != PAD_ID]
        candidates = lm.suggest_ids(context_ids[-max_context:], pool_k)
        if label in candidates:
            hit_count += 1
        rows.append(candidates)
    padded_rows = [
        (row[:pool_k] + [PAD_ID] * max(0, pool_k - len(row)))[:pool_k]
        for row in rows
    ]
    return CandidatePoolSet(
        rows=rows,
        ids=torch.tensor(padded_rows, dtype=torch.long),
        hit_count=hit_count,
    )


def score_candidate_pool(
    model: NextWordLm,
    contexts: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    hidden = model.encode_context(contexts)
    candidate_embeddings = model.token_embedding(candidate_ids)
    scores = torch.einsum("bd,bkd->bk", hidden, candidate_embeddings)
    return scores + model.output_bias[candidate_ids]


@torch.no_grad()
def evaluate_hybrid_rerank(
    model: NextWordLm,
    lm: NgramLm,
    example_set: ExampleSet,
    device: torch.device,
    pool_k: int,
    rank_penalties: tuple[float, ...],
    lock_first: bool,
    batch_size: int = 1024,
    cutoffs: tuple[int, ...] = DEFAULT_REPORT_CUTOFFS,
) -> list[dict]:
    """Evaluate neural reranking over the bounded static candidate pool.

    The runtime design should not ask a neural model to invent arbitrary words.
    This probe keeps retrieval in the mmap-friendly n-gram artifact, then uses
    the neural LM only to reorder the small candidate set. `rank_penalty` is the
    cost of moving one slot down from the static rank; high values converge back
    toward the static model, while zero is neural-only within the pool.
    """

    if pool_k < 1:
        raise ValueError("pool_k must be at least 1")
    model.eval()
    candidate_pool = collect_candidate_pool(lm, example_set, pool_k)

    reports: list[dict] = []
    counters = {
        penalty: {
            "hits": Counter(),
            "reciprocal_rank_sum": 0.0,
        }
        for penalty in rank_penalties
    }
    for start in range(0, example_set.size, batch_size):
        end = min(example_set.size, start + batch_size)
        contexts = example_set.contexts[start:end].to(device)
        candidate_scores = score_candidate_pool(
            model,
            contexts,
            candidate_pool.ids[start:end].to(device),
        ).detach().cpu()
        labels = example_set.labels[start:end].tolist()
        for row_index, label in enumerate(labels, start=start):
            candidates = candidate_pool.rows[row_index]
            if not candidates:
                continue
            for penalty, counter in counters.items():
                row_scores = candidate_scores[row_index - start]
                indexed_candidates = list(enumerate(candidates))
                scored_tail = sorted(
                    indexed_candidates[1:] if lock_first else indexed_candidates,
                    key=lambda item: (
                        -float(row_scores[item[0]]) + penalty * item[0],
                        item[0],
                    ),
                )
                if lock_first:
                    ranked = [indexed_candidates[0], *scored_tail]
                else:
                    ranked = scored_tail
                rank = 0
                for sorted_index, (_, token_id) in enumerate(ranked, start=1):
                    if token_id == label:
                        rank = sorted_index
                        break
                if rank:
                    counter["reciprocal_rank_sum"] += 1.0 / rank
                for cutoff in cutoffs:
                    if rank and rank <= min(cutoff, pool_k):
                        counter["hits"][cutoff] += 1

    for penalty in rank_penalties:
        counter = counters[penalty]
        report = rank_report(
            counter["hits"],
            counter["reciprocal_rank_sum"],
            example_set,
            cutoffs,
        )
        report["rank_penalty"] = penalty
        report["pool_k"] = pool_k
        report["lock_first"] = lock_first
        report["pool_recall"] = candidate_pool.hit_count / max(1, example_set.size)
        report["pool_recall_all_targets"] = candidate_pool.hit_count / max(
            1,
            example_set.total_targets,
        )
        reports.append(report)
    return reports


@torch.no_grad()
def evaluate_neural_augmented_pool(
    model: NextWordLm,
    lm: NgramLm,
    example_set: ExampleSet,
    device: torch.device,
    ngram_pool_k: int,
    neural_top_ks: tuple[int, ...],
    batch_size: int = 1024,
) -> list[dict]:
    """Measure whether full-vocabulary neural candidates raise pool recall.

    The deployed c64 scorer can only rerank candidates already retrieved from
    the n-gram artifact. This probe answers a different architecture question:
    if a platform can afford one full-vocabulary neural pass, how much candidate
    recall is gained by unioning the neural top-N IDs with the n-gram pool?
    """

    top_ks = tuple(sorted({top_k for top_k in neural_top_ks if top_k > 0}))
    if not top_ks:
        return []
    if ngram_pool_k < 1:
        raise ValueError("ngram_pool_k must be at least 1")

    model.eval()
    candidate_pool = collect_candidate_pool(lm, example_set, ngram_pool_k)
    ngram_hit_mask = [
        label in candidates
        for label, candidates in zip(example_set.labels.tolist(), candidate_pool.rows)
    ]
    ngram_hit_count = sum(1 for hit in ngram_hit_mask if hit)
    neural_hits = Counter()
    union_hits = Counter()

    for start in range(0, example_set.size, batch_size):
        end = min(example_set.size, start + batch_size)
        contexts = example_set.contexts[start:end].to(device)
        labels = example_set.labels[start:end].to(device)
        ranks = label_ranks(model(contexts), labels).detach().cpu().tolist()
        for offset, rank in enumerate(ranks):
            row_index = start + offset
            for top_k in top_ks:
                neural_hit = rank <= top_k
                if neural_hit:
                    neural_hits[top_k] += 1
                if ngram_hit_mask[row_index] or neural_hit:
                    union_hits[top_k] += 1

    reports: list[dict] = []
    for top_k in top_ks:
        report = {
            "ngram_pool_k": ngram_pool_k,
            "neural_top_k": top_k,
            "eligible_targets": example_set.size,
            "total_targets": example_set.total_targets,
            "ngram_pool_recall": ngram_hit_count / max(1, example_set.size),
            "ngram_pool_recall_all_targets": ngram_hit_count
            / max(1, example_set.total_targets),
            "neural_recall": neural_hits[top_k] / max(1, example_set.size),
            "neural_recall_all_targets": neural_hits[top_k]
            / max(1, example_set.total_targets),
            "union_recall": union_hits[top_k] / max(1, example_set.size),
            "union_recall_all_targets": union_hits[top_k]
            / max(1, example_set.total_targets),
        }
        report["absolute_union_gain_all_targets"] = (
            report["union_recall_all_targets"] - report["ngram_pool_recall_all_targets"]
        )
        report["absolute_union_gain"] = report["union_recall"] - report["ngram_pool_recall"]
        reports.append(report)
    return reports


@torch.no_grad()
def benchmark_candidate_pool(
    model: NextWordLm,
    lm: NgramLm,
    example_set: ExampleSet,
    device: torch.device,
    pool_k: int,
    iterations: int,
    batch_size: int,
) -> dict | None:
    if iterations <= 0:
        return None
    if batch_size <= 0:
        raise ValueError("candidate benchmark batch size must be positive")
    model.eval()
    sample_size = min(example_set.size, max(batch_size, min(iterations * batch_size, 4096)))
    if sample_size == 0:
        return None
    sample = ExampleSet(
        contexts=example_set.contexts[:sample_size],
        labels=example_set.labels[:sample_size],
        source_ids=example_set.source_ids[:sample_size],
        source_names=example_set.source_names,
        total_targets=sample_size,
        eligible_targets=sample_size,
        examples_by_source={},
        scanned_sentences_by_source={},
    )
    candidate_pool = collect_candidate_pool(lm, sample, pool_k)
    contexts = sample.contexts.to(device)
    candidate_ids = candidate_pool.ids.to(device)
    warmup = min(100, iterations)
    for index in range(warmup):
        start = (index * batch_size) % sample_size
        end = min(start + batch_size, sample_size)
        if end - start < batch_size:
            start = 0
            end = batch_size
        _ = score_candidate_pool(model, contexts[start:end], candidate_ids[start:end])
    synchronize_device(device)
    started_at = time.perf_counter()
    for index in range(iterations):
        start = (index * batch_size) % sample_size
        end = min(start + batch_size, sample_size)
        if end - start < batch_size:
            start = 0
            end = batch_size
        _ = score_candidate_pool(model, contexts[start:end], candidate_ids[start:end])
    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "pool_k": pool_k,
        "iterations": iterations,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "mean_us_per_batch": elapsed_seconds * 1_000_000.0 / iterations,
        "mean_us_per_item": elapsed_seconds * 1_000_000.0 / (iterations * batch_size),
        "sample_size": sample_size,
    }


@torch.no_grad()
def benchmark_full_vocab(
    model: NextWordLm,
    example_set: ExampleSet,
    device: torch.device,
    top_k: int,
    iterations: int,
    batch_size: int,
) -> dict | None:
    if iterations <= 0:
        return None
    if batch_size <= 0:
        raise ValueError("full-vocab benchmark batch size must be positive")
    if top_k <= 0:
        raise ValueError("full-vocab benchmark top-k must be positive")
    model.eval()
    sample_size = min(example_set.size, max(batch_size, min(iterations * batch_size, 4096)))
    if sample_size == 0:
        return None
    contexts = example_set.contexts[:sample_size].to(device)
    top_k = min(top_k, model.vocab_size)
    warmup = min(100, iterations)
    for index in range(warmup):
        start = (index * batch_size) % sample_size
        end = min(start + batch_size, sample_size)
        if end - start < batch_size:
            start = 0
            end = batch_size
        logits = model(contexts[start:end])
        _ = torch.topk(logits, k=top_k, dim=1)
    synchronize_device(device)
    started_at = time.perf_counter()
    for index in range(iterations):
        start = (index * batch_size) % sample_size
        end = min(start + batch_size, sample_size)
        if end - start < batch_size:
            start = 0
            end = batch_size
        logits = model(contexts[start:end])
        _ = torch.topk(logits, k=top_k, dim=1)
    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "vocab_size": model.vocab_size,
        "top_k": top_k,
        "iterations": iterations,
        "batch_size": batch_size,
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "mean_us_per_batch": elapsed_seconds * 1_000_000.0 / iterations,
        "mean_us_per_item": elapsed_seconds * 1_000_000.0 / (iterations * batch_size),
        "sample_size": sample_size,
    }


def label_ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    label_scores = logits.gather(1, labels.unsqueeze(1))
    return (logits > label_scores).sum(dim=1) + 1


def rank_report(
    hits: Counter,
    reciprocal_rank_sum: float,
    example_set: ExampleSet,
    cutoffs: tuple[int, ...],
) -> dict:
    total_targets = max(1, example_set.total_targets)
    eligible = max(1, example_set.size)
    report = {
        "eligible_targets": example_set.size,
        "total_targets": example_set.total_targets,
        "mrr": reciprocal_rank_sum / eligible,
        "mrr_all_targets": reciprocal_rank_sum / total_targets,
    }
    for cutoff in cutoffs:
        report[f"top{cutoff}"] = hits[cutoff] / eligible
        report[f"top{cutoff}_all_targets"] = hits[cutoff] / total_targets
    return report


def example_set_report(example_set: ExampleSet) -> dict:
    return {
        "total_targets": example_set.total_targets,
        "eligible_targets": example_set.eligible_targets,
        "examples": example_set.size,
        "eligible_ratio": example_set.eligible_targets / max(1, example_set.total_targets),
        "examples_by_source": example_set.examples_by_source,
        "scanned_sentences_by_source": example_set.scanned_sentences_by_source,
        "total_targets_by_source": example_set.total_targets_by_source,
    }


def subset_example_set(example_set: ExampleSet, indexes: torch.Tensor, *, source_name: str | None = None) -> ExampleSet:
    source_ids = example_set.source_ids[indexes]
    examples_by_source = {
        source_name: int((source_ids == source_id).sum())
        for source_id, source_name in enumerate(example_set.source_names)
        if int((source_ids == source_id).sum()) > 0
    }
    total_targets_by_source = {
        name: example_set.total_targets_by_source.get(name, count)
        if count == example_set.examples_by_source.get(name) else count
        for name, count in examples_by_source.items()
    }
    if source_name is not None:
        total_targets_by_source = {source_name: example_set.total_targets_by_source.get(source_name, int(indexes.numel()))}
    return ExampleSet(
        contexts=example_set.contexts[indexes],
        labels=example_set.labels[indexes],
        source_ids=source_ids,
        source_names=example_set.source_names,
        total_targets=sum(total_targets_by_source.values()),
        eligible_targets=int(indexes.numel()),
        examples_by_source=examples_by_source,
        scanned_sentences_by_source={},
        total_targets_by_source=total_targets_by_source,
    )


def source_gain_report(model_by_source: dict[str, dict], baseline_by_source: dict[str, dict]) -> dict:
    report: dict[str, dict] = {}
    for source_name, model_report in model_by_source.items():
        baseline_report = baseline_by_source.get(source_name)
        if baseline_report is None:
            continue
        source_report = {
            "eligible_targets": model_report["eligible_targets"],
            "total_targets": model_report["total_targets"],
        }
        for cutoff in DEFAULT_REPORT_CUTOFFS:
            key = f"top{cutoff}_all_targets"
            source_report[key] = model_report[key]
            source_report[f"{key}_gain_vs_ngram"] = model_report[key] - baseline_report[key]
        source_report["mrr_all_targets"] = model_report["mrr_all_targets"]
        source_report["mrr_all_target_gain_vs_ngram"] = (
            model_report["mrr_all_targets"] - baseline_report["mrr_all_targets"]
        )
        report[source_name] = source_report
    return report


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tensor_tree_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_tree_to_cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [tensor_tree_to_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree_to_cpu(child) for child in value)
    return value


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint_model(
    checkpoint_path: Path,
    expected_vocab_size: int,
    expected_context_window: int,
    *,
    provenance: dict,
    retrieval_model: Path,
) -> tuple[NextWordLm, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    verify_checkpoint_provenance(checkpoint, provenance, retrieval_model)
    config = checkpoint["config"]
    if int(config["vocab_size"]) != expected_vocab_size:
        raise SystemExit(
            f"checkpoint vocab mismatch: checkpoint={config['vocab_size']} "
            f"artifact={expected_vocab_size}"
        )
    if int(config["context_window"]) != expected_context_window:
        raise SystemExit(
            f"checkpoint context mismatch: checkpoint={config['context_window']} "
            f"requested={expected_context_window}"
        )
    model = NextWordLm(
        vocab_size=expected_vocab_size,
        context_len=expected_context_window,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        architecture=str(config["architecture"]),
        dropout=0.0,
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        transformer_padding_mode=config.get("transformer_padding_mode", "unmasked-v0"),
        transformer_initialization=config.get("transformer_initialization", "legacy-v0"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--validation-corpus-dir", type=Path,
                        help="Independent validation partition; required for versioned corpora. Test data is forbidden here.")
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--context-window", type=int, default=16)
    parser.add_argument("--architecture", choices=("gru", "transformer"), default="gru")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-padding-mode", choices=("masked-v1", "unmasked-v0"), default="masked-v1",
                        help="Versioned padding behavior; unmasked-v0 preserves historical transformer checkpoints")
    parser.add_argument("--transformer-initialization", choices=("independent-v1", "legacy-v0"), default="independent-v1",
                        help="Versioned initialization; legacy-v0 preserves historical training profiles")
    parser.add_argument("--train-skip-sentences-per-source", type=int, default=0)
    parser.add_argument("--train-max-sentences-per-source", type=int, default=100_000)
    parser.add_argument("--train-max-examples-per-source", type=int, default=80_000)
    parser.add_argument("--eval-skip-sentences-per-source", type=int,
                        help="Defaults to 0 for separate validation data, 100000 for legacy corpus slicing.")
    parser.add_argument("--eval-max-sentences-per-source", type=int, default=25_000)
    parser.add_argument("--eval-max-examples-per-source", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float,
                        help="Optional gradient norm clipping; non-finite gradients fail before the update")
    parser.add_argument(
        "--loss-mode",
        choices=("token", "source-balanced"),
        default="token",
        help="Training loss reduction. source-balanced gives each source present in a batch equal weight.",
    )
    parser.add_argument(
        "--batch-sampling",
        choices=("token", "source-balanced"),
        default="token",
        help=(
            "Epoch batch construction. source-balanced samples each batch evenly "
            "across available sources and oversamples smaller sources when needed."
        ),
    )
    parser.add_argument(
        "--no-unigram-prior",
        action="store_true",
        help="Disable output-bias initialization from the artifact unigram counts.",
    )
    parser.add_argument(
        "--no-resume-optimizer",
        action="store_true",
        help="Do not restore AdamW state from --input-checkpoint even if it is present.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline-top-k", type=int, default=10)
    parser.add_argument("--hybrid-pool-k", type=int, default=16)
    parser.add_argument(
        "--neural-augment-top-ks",
        default="",
        help="Comma-separated neural top-N candidate budgets to union with the n-gram pool for recall probes.",
    )
    parser.add_argument(
        "--hybrid-rank-penalties",
        default="0,0.25,0.5,0.75,1,1.5,2,3,4,6,8",
        help="Comma-separated static-rank penalties for neural pool reranking.",
    )
    parser.add_argument("--candidate-benchmark-iterations", type=int, default=0)
    parser.add_argument("--candidate-benchmark-batch-size", type=int, default=1)
    parser.add_argument("--full-vocab-benchmark-iterations", type=int, default=0)
    parser.add_argument("--full-vocab-benchmark-batch-size", type=int, default=1)
    parser.add_argument("--full-vocab-benchmark-top-k", type=int, default=128)
    parser.add_argument("--input-checkpoint", type=Path)
    parser.add_argument(
        "--distill-teacher-checkpoint",
        type=Path,
        help="Optional teacher checkpoint for hard-label plus KL distillation.",
    )
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=1.0,
        help="Hard-label CE weight when distilling. 1.0 disables teacher loss.",
    )
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--distill-top-k",
        type=int,
        default=0,
        help="Use only the teacher's top-K logits for distillation. 0 uses full-vocab KL.",
    )
    parser.add_argument("--output-report", type=Path, default=Path("target/autosuggest-next-word-lm-report.json"))
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--best-checkpoint", type=Path,
                        help="Save best validation macro top-5 accuracy including unknown targets")
    parser.add_argument("--checkpoint-seconds", type=float, default=300,
                        help="Atomic resumable checkpoint interval, at most 900 seconds")
    parser.add_argument("--max-steps", type=int, help="Stop cleanly after this many optimizer steps in this invocation")
    parser.add_argument("--max-training-seconds", type=float)
    parser.add_argument("--stop-at-utc", help="Absolute ISO-8601 cutoff with timezone, checked between optimizer steps")
    parser.add_argument("--log-every-targets", type=int, default=250_000)
    args = parser.parse_args()

    data_provenance = neural_training_provenance(args.model, args.corpus_dir, args.validation_corpus_dir)
    if args.eval_skip_sentences_per_source is None:
        args.eval_skip_sentences_per_source = 0 if args.validation_corpus_dir else 100_000

    if args.context_window < 1:
        raise SystemExit("--context-window must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.epochs < 0:
        raise SystemExit("--epochs must be nonnegative")
    if args.epochs == 0 and (args.output_checkpoint or args.best_checkpoint):
        raise SystemExit("evaluation-only runs cannot write training checkpoints; use eval_next_word_lm")
    if args.max_grad_norm is not None and (not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0):
        raise SystemExit("--max-grad-norm must be positive and finite")
    if not 0 < args.checkpoint_seconds <= 900:
        raise SystemExit("--checkpoint-seconds must be in (0, 900]")
    if args.best_checkpoint and args.output_checkpoint and args.best_checkpoint.resolve() == args.output_checkpoint.resolve():
        raise SystemExit("--best-checkpoint must differ from --output-checkpoint")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if args.max_training_seconds is not None and (not math.isfinite(args.max_training_seconds) or args.max_training_seconds <= 0):
        raise SystemExit("--max-training-seconds must be positive and finite")
    stop_at = None
    if args.stop_at_utc:
        cutoff = datetime.fromisoformat(args.stop_at_utc.replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            raise SystemExit("--stop-at-utc requires a timezone")
        stop_at = cutoff.timestamp()
    rank_penalties = tuple(
        float(value)
        for value in args.hybrid_rank_penalties.split(",")
        if value.strip()
    )
    if not rank_penalties:
        raise SystemExit("--hybrid-rank-penalties must contain at least one value")
    neural_augment_top_ks = tuple(
        int(value)
        for value in args.neural_augment_top_ks.split(",")
        if value.strip()
    )
    if any(value < 1 for value in neural_augment_top_ks):
        raise SystemExit("--neural-augment-top-ks must contain positive integers")
    if not (0.0 <= args.distill_alpha <= 1.0):
        raise SystemExit("--distill-alpha must be in [0, 1]")
    if args.distill_temperature <= 0.0:
        raise SystemExit("--distill-temperature must be positive")
    if args.distill_top_k < 0:
        raise SystemExit("--distill-top-k must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started_at = time.time()
    device = choose_device(args.device)
    lm = NgramLm(args.model)
    print(json.dumps({"event": "training_device", "device": str(device), "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}), flush=True)
    sources = set(args.sources) if args.sources else None

    eval_set = collect_examples(
        lm,
        args.validation_corpus_dir or args.corpus_dir,
        sources,
        args.eval_skip_sentences_per_source,
        args.eval_max_sentences_per_source,
        args.eval_max_examples_per_source,
        args.context_window,
        args.log_every_targets,
    )
    train_set = (
        empty_example_set()
        if args.epochs == 0
        else collect_examples(
            lm,
            args.corpus_dir,
            sources,
            args.train_skip_sentences_per_source,
            args.train_max_sentences_per_source,
            args.train_max_examples_per_source,
            args.context_window,
            args.log_every_targets,
        )
    )
    model = NextWordLm(
        vocab_size=lm.vocab_size,
        context_len=args.context_window,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        architecture=args.architecture,
        dropout=args.dropout,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_padding_mode=args.transformer_padding_mode,
        transformer_initialization=args.transformer_initialization,
    )
    if not args.no_unigram_prior:
        initialize_output_bias_from_unigrams(model, lm)
    checkpoint = None
    start_epoch = 0
    optimizer_state_loaded = False
    if args.input_checkpoint:
        checkpoint = torch.load(args.input_checkpoint, map_location="cpu", weights_only=False)
        verify_checkpoint_provenance(checkpoint, data_provenance, args.model)
        if args.architecture == "transformer" and checkpoint["config"].get("transformer_padding_mode", "unmasked-v0") != args.transformer_padding_mode:
            raise ValueError("checkpoint transformer padding contract differs; preserve its recorded mode")
        if args.architecture == "transformer" and checkpoint["config"].get("transformer_initialization", "legacy-v0") != args.transformer_initialization:
            raise ValueError("checkpoint transformer initialization contract differs; preserve its recorded mode")
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
    teacher_model = None
    teacher_config = None
    if args.distill_teacher_checkpoint:
        teacher_model, teacher_config = load_checkpoint_model(
            args.distill_teacher_checkpoint,
            lm.vocab_size,
            args.context_window,
            provenance=data_provenance,
            retrieval_model=args.model,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if (
        checkpoint is not None
        and not args.no_resume_optimizer
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state(optimizer, device)
        optimizer_state_loaded = True
    artifact_record = {"path": str(args.model), "sha256": sha256_file(args.model), "bytes": len(lm.bytes),
                       "vocab_size": lm.vocab_size, "candidate_record_len": lm.candidate_record_len,
                       "max_context_order": lm.max_context_order}
    model_config = {"architecture": args.architecture, "context_window": args.context_window,
                    "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
                    "transformer_layers": args.transformer_layers, "transformer_heads": args.transformer_heads,
                    "transformer_padding_mode": args.transformer_padding_mode,
                    "transformer_initialization": args.transformer_initialization,
                    "dropout": args.dropout, "vocab_size": lm.vocab_size}
    contract_fields = ("seed", "batch_size", "learning_rate", "weight_decay", "max_grad_norm", "loss_mode", "batch_sampling",
                       "train_skip_sentences_per_source", "train_max_sentences_per_source", "train_max_examples_per_source",
                       "eval_skip_sentences_per_source", "eval_max_sentences_per_source", "eval_max_examples_per_source",
                       "no_unigram_prior", "distill_alpha", "distill_temperature", "distill_top_k")
    training_contract = {"version": 1, "model": model_config, "options": {key: getattr(args, key) for key in contract_fields},
                         "sources": sorted(sources) if sources else None, "trainer_sha256": sha256_file(Path(__file__)),
                         "teacher_sha256": sha256_file(args.distill_teacher_checkpoint) if args.distill_teacher_checkpoint else None}
    progress = {"completed_epochs": start_epoch, "next_batch": 0, "optimizer_steps": 0, "status": "evaluation_only"}
    if checkpoint is not None and checkpoint.get("training_state") is not None and args.epochs > 0:
        if checkpoint.get("training_contract") != training_contract:
            raise ValueError("checkpoint training contract differs; sampling, architecture, and optimizer settings must match")
        if not optimizer_state_loaded:
            raise ValueError("mid-run resume requires the saved optimizer state")
        progress = dict(checkpoint["training_state"])
        restore_rng(checkpoint["rng_state"], device)

    def save_progress(state: dict, full_report: dict | None = None, destination: Path | None = None) -> None:
        destination = destination or args.output_checkpoint
        if destination is None:
            return
        atomic_torch_save({"state_dict": tensor_tree_to_cpu(model.state_dict()),
                           "optimizer_state_dict": tensor_tree_to_cpu(optimizer.state_dict()),
                           "epoch": state["completed_epochs"], "training_state": dict(state),
                           "training_contract": training_contract, "rng_state": capture_rng(device),
                           "config": model_config,
                           "report": full_report or {"data_provenance": data_provenance, "artifact": artifact_record,
                                                      "model": model_config, "training_state": dict(state)}},
                          destination)
        print(json.dumps({"event": "checkpoint_saved", "path": str(destination),
                          "completed_epochs": state["completed_epochs"], "next_batch": state["next_batch"],
                          "optimizer_steps": state["optimizer_steps"], "status": state["status"]}), flush=True)

    def save_best(state: dict, epoch_report: dict) -> None:
        reports = list(epoch_report["eval_by_source"].values())
        score = sum(item["top5_all_targets"] for item in reports) / max(1, len(reports))
        if reports and score > state.get("best_validation_macro_top5_all_targets", -1.0):
            state["best_validation_macro_top5_all_targets"] = score
            state["best_epoch"] = state["completed_epochs"]
            if args.best_checkpoint:
                save_progress(state, {"data_provenance": data_provenance, "artifact": artifact_record,
                                      "model": model_config, "epoch_validation": epoch_report,
                                      "selection_metric": "validation_macro_top5_all_targets"}, args.best_checkpoint)
        save_progress(state)

    history = []
    if args.epochs > 0:
        history = train(
            model,
            optimizer,
            train_set,
            eval_set,
            device,
            args.epochs,
            args.batch_size,
            args.seed,
            start_epoch,
            args.loss_mode,
            teacher_model,
            args.distill_alpha,
            args.distill_temperature,
            args.distill_top_k,
            args.batch_sampling,
            progress=progress,
            checkpoint_callback=save_progress if args.output_checkpoint else None,
            epoch_callback=save_best,
            checkpoint_seconds=args.checkpoint_seconds,
            max_steps=args.max_steps,
            max_training_seconds=args.max_training_seconds,
            stop_at=stop_at,
            max_grad_norm=args.max_grad_norm,
        )
    final_eval = evaluate_model(model, eval_set, device)
    final_eval_by_source = evaluate_model_by_source(model, eval_set, device)
    baseline = evaluate_ngram_baseline(lm, eval_set, args.baseline_top_k)
    baseline_by_source = evaluate_ngram_baseline_by_source(lm, eval_set, args.baseline_top_k)
    hybrid_rerank = evaluate_hybrid_rerank(
        model,
        lm,
        eval_set,
        device,
        args.hybrid_pool_k,
        rank_penalties,
        lock_first=False,
    )
    hybrid_rerank_locked_first = evaluate_hybrid_rerank(
        model,
        lm,
        eval_set,
        device,
        args.hybrid_pool_k,
        rank_penalties,
        lock_first=True,
    )
    neural_augmented_pool = evaluate_neural_augmented_pool(
        model,
        lm,
        eval_set,
        device,
        args.hybrid_pool_k,
        neural_augment_top_ks,
    )
    candidate_benchmark = benchmark_candidate_pool(
        model,
        lm,
        eval_set,
        device,
        args.hybrid_pool_k,
        args.candidate_benchmark_iterations,
        args.candidate_benchmark_batch_size,
    )
    full_vocab_benchmark = benchmark_full_vocab(
        model,
        eval_set,
        device,
        args.full_vocab_benchmark_top_k,
        args.full_vocab_benchmark_iterations,
        args.full_vocab_benchmark_batch_size,
    )
    report = {
        "data_provenance": data_provenance,
        "training_state": progress,
        "training_contract": training_contract,
        "artifact": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "bytes": len(lm.bytes),
            "vocab_size": lm.vocab_size,
            "candidate_record_len": lm.candidate_record_len,
            "max_context_order": lm.max_context_order,
        },
        "model": {
            "architecture": args.architecture,
            "context_window": args.context_window,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "parameter_count": parameter_count(model),
            "fp32_parameter_bytes": parameter_count(model) * 4,
            "unigram_prior": not args.no_unigram_prior,
            "input_checkpoint": str(args.input_checkpoint) if args.input_checkpoint else None,
            "start_epoch": start_epoch,
        },
        "device": str(device),
        "optimizer": {
            "type": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "state_loaded": optimizer_state_loaded,
            "loss_mode": args.loss_mode,
            "batch_sampling": args.batch_sampling,
        },
        "distillation": {
            "teacher_checkpoint": (
                str(args.distill_teacher_checkpoint)
                if args.distill_teacher_checkpoint
                else None
            ),
            "teacher_model": teacher_config,
            "alpha": args.distill_alpha,
            "temperature": args.distill_temperature,
            "top_k": args.distill_top_k,
            "enabled": teacher_model is not None and args.distill_alpha < 1.0,
        },
        "train_collection": example_set_report(train_set),
        "eval_collection": example_set_report(eval_set),
        "ngram_baseline": baseline,
        "ngram_baseline_by_source": baseline_by_source,
        "hybrid_rerank": hybrid_rerank,
        "hybrid_rerank_locked_first": hybrid_rerank_locked_first,
        "neural_augmented_pool": neural_augmented_pool,
        "candidate_pool_benchmark": candidate_benchmark,
        "full_vocab_benchmark": full_vocab_benchmark,
        "history": history,
        "final_eval": final_eval,
        "final_eval_by_source": final_eval_by_source,
        "final_eval_source_gains_vs_ngram": source_gain_report(
            final_eval_by_source,
            baseline_by_source,
        ),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    if args.output_checkpoint:
        save_progress(progress, report)
        report["checkpoint"] = str(args.output_checkpoint)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
