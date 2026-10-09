#!/usr/bin/env python3
"""Evaluate autosuggest scorer policies across multiple held-out offsets.

This is an offline production gate for the next-word layer. It evaluates a
trained full-vocabulary checkpoint only as a bounded candidate scorer over the
n-gram pool and treats out-of-pool targets as misses. That last detail matters:
candidate-pool recall is the hard ceiling for any deployed reranker.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from tools.autosuggest.common import PAD_ID, write_json
from tools.autosuggest.eval_ngram_lm import NgramLm
from tools.autosuggest.train_candidate_reranker import suggest_ranked_candidates
from tools.autosuggest.train_next_word_lm import NextWordLm, collect_examples, score_candidate_pool


REPORT_CUTOFFS = (1, 3, 5, 10, 20, 64)


def load_checkpoint(path: Path) -> NextWordLm:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = NextWordLm(
        vocab_size=int(config["vocab_size"]),
        context_len=int(config["context_window"]),
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
    return model


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def evaluate_offset(
    model: NextWordLm,
    lm: NgramLm,
    corpus_dir: Path,
    skip_sentences_per_source: int,
    max_sentences_per_source: int | None,
    max_examples_per_source: int | None,
    context_window: int,
    pool_k: int,
    rank_penalties: tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> dict:
    examples = collect_examples(
        lm,
        corpus_dir,
        sources=None,
        skip_sentences_per_source=skip_sentences_per_source,
        max_sentences_per_source=max_sentences_per_source,
        max_examples_per_source=max_examples_per_source,
        context_window=context_window,
        log_every_targets=0,
    )
    candidate_ids, label_positions, valid_mask = collect_candidate_matrix(lm, examples, pool_k)
    neural_scores = score_candidates(
        model,
        examples.contexts,
        candidate_ids,
        batch_size,
        device,
    )
    total_targets = max(1, examples.total_targets)
    rank = np.arange(pool_k, dtype=np.float32)[None, :]
    tie_break = rank * 1e-6

    profiles: dict[str, dict] = {
        "static": metrics_for_scores(
            -rank - tie_break,
            label_positions,
            valid_mask,
            total_targets,
        )
    }
    for penalty in rank_penalties:
        scores = neural_scores - penalty * rank - tie_break
        scores[:, 0] = 1e9
        profiles[f"{penalty:g}"] = metrics_for_scores(
            scores,
            label_positions,
            valid_mask,
            total_targets,
        )

    return {
        "skip_sentences_per_source": skip_sentences_per_source,
        "total_targets": examples.total_targets,
        "eligible_targets": examples.eligible_targets,
        "examples": examples.size,
        "candidate_hit_ratio_all_targets": float(
            np.count_nonzero(label_positions >= 0) / total_targets
        ),
        "profiles": profiles,
        "best_by_top5": best_profile(profiles, "top5_all_targets"),
        "best_by_mrr": best_profile(profiles, "mrr_all_targets"),
    }


def collect_candidate_matrix(
    lm: NgramLm,
    examples,
    pool_k: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    candidate_ids = np.full((examples.size, pool_k), PAD_ID, dtype=np.int64)
    label_positions = np.full(examples.size, -1, dtype=np.int64)
    valid_mask = np.zeros((examples.size, pool_k), dtype=bool)

    for row_index, context_row in enumerate(examples.contexts.tolist()):
        context_ids = [token_id for token_id in context_row if token_id != PAD_ID]
        candidates = suggest_ranked_candidates(
            lm,
            context_ids[-lm.max_context_order :],
            pool_k,
        )
        label = int(examples.labels[row_index])
        for candidate_index, candidate in enumerate(candidates[:pool_k]):
            candidate_ids[row_index, candidate_index] = candidate.token_id
            valid_mask[row_index, candidate_index] = True
            if candidate.token_id == label:
                label_positions[row_index] = candidate_index

    return torch.from_numpy(candidate_ids), label_positions, valid_mask


@torch.no_grad()
def score_candidates(
    model: NextWordLm,
    contexts: torch.Tensor,
    candidate_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.to(device)
    for start in range(0, int(contexts.shape[0]), batch_size):
        end = min(int(contexts.shape[0]), start + batch_size)
        scores = score_candidate_pool(
            model,
            contexts[start:end].to(device),
            candidate_ids[start:end].to(device),
        )
        chunks.append(scores.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def metrics_for_scores(
    scores: np.ndarray,
    label_positions: np.ndarray,
    valid_mask: np.ndarray,
    total_targets: int,
) -> dict:
    valid_rows = label_positions >= 0
    ranks = np.zeros(label_positions.shape[0], dtype=np.int32)

    if np.any(valid_rows):
        masked_scores = np.where(valid_mask, scores, -1e9)
        row_indexes = np.nonzero(valid_rows)[0]
        label_scores = masked_scores[row_indexes, label_positions[valid_rows]]
        ranks[valid_rows] = (
            masked_scores[row_indexes] > label_scores[:, None]
        ).sum(axis=1) + 1

    metrics = {
        f"top{cutoff}_all_targets": float(
            np.count_nonzero((ranks > 0) & (ranks <= cutoff)) / total_targets
        )
        for cutoff in REPORT_CUTOFFS
    }
    metrics["mrr_all_targets"] = float(
        np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0).sum() / total_targets
    )
    return metrics


def best_profile(profiles: dict[str, dict], metric: str) -> dict:
    name, metrics = max(
        profiles.items(),
        key=lambda item: (
            item[1][metric],
            item[1]["mrr_all_targets"],
            item[1]["top3_all_targets"],
        ),
    )
    return {"profile": name, **metrics}


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--context-window", type=int, default=16)
    parser.add_argument("--pool-k", type=int, default=64)
    parser.add_argument(
        "--skip-sentences-per-source",
        type=parse_int_tuple,
        default=(100_000, 125_000, 150_000, 175_000),
        help="Comma-separated held-out source offsets.",
    )
    parser.add_argument("--max-sentences-per-source", type=int, default=25_000)
    parser.add_argument("--max-examples-per-source", type=int, default=30_000)
    parser.add_argument(
        "--rank-penalties",
        type=parse_float_tuple,
        default=(0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()

    if args.pool_k < 1:
        raise SystemExit("--pool-k must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    device = choose_device(args.device)
    lm = NgramLm(args.model)
    model = load_checkpoint(args.checkpoint)
    reports = [
        evaluate_offset(
            model,
            lm,
            args.corpus_dir,
            skip,
            args.max_sentences_per_source,
            args.max_examples_per_source,
            args.context_window,
            args.pool_k,
            args.rank_penalties,
            args.batch_size,
            device,
        )
        for skip in args.skip_sentences_per_source
    ]
    output = {
        "model": str(args.model),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "context_window": args.context_window,
        "pool_k": args.pool_k,
        "rank_penalties": list(args.rank_penalties),
        "offsets": reports,
        "mean_best_top5_all_targets": mean_metric(reports, "best_by_top5", "top5_all_targets"),
        "mean_best_mrr_all_targets": mean_metric(reports, "best_by_mrr", "mrr_all_targets"),
    }
    if args.output_report:
        write_json(args.output_report, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def mean_metric(reports: Iterable[dict], profile_key: str, metric: str) -> float:
    values = [report[profile_key][metric] for report in reports]
    return float(sum(values) / max(1, len(values)))


if __name__ == "__main__":
    main()
