"""Evaluate one frozen checkpoint and ranking policy on validation or test.

No optimizer, model selection, or parameter sweep is permitted here. All-target
metrics include unknown targets as misses. Test results must not feed tuning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tools.corpus.provenance import (
    evaluation_provenance,
    neural_training_provenance,
    sha256_file,
    verify_checkpoint_provenance,
    write_json,
)
from tools.autosuggest.eval_ngram_lm import NgramLm
from tools.autosuggest.train_next_word_lm import (
    NextWordLm,
    choose_device,
    collect_examples,
    example_set_report,
    evaluate_model,
    evaluate_ngram_baseline,
    evaluate_hybrid_rerank,
    subset_example_set,
)


def evaluate(args: argparse.Namespace) -> dict:
    provenance = evaluation_provenance(args.model, args.corpus_dir)
    if provenance["status"] != "partition_verified":
        raise ValueError("frozen neural evaluation requires a verified partition set")
    training = neural_training_provenance(
        args.model,
        args.corpus_dir.parent / "train",
        args.corpus_dir.parent / "validation",
    )
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    verify_checkpoint_provenance(checkpoint, training, args.model)
    lm = NgramLm(args.model)
    config = checkpoint["config"]
    if config["vocab_size"] != lm.vocab_size:
        raise ValueError("checkpoint vocabulary size differs from retrieval artifact")
    model = NextWordLm(
        vocab_size=config["vocab_size"],
        context_len=config["context_window"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        architecture=config["architecture"],
        dropout=0,
        transformer_layers=config["transformer_layers"],
        transformer_heads=config["transformer_heads"],
        transformer_padding_mode=config.get("transformer_padding_mode", "unmasked-v0"),
        transformer_initialization=config.get(
            "transformer_initialization", "legacy-v0"
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = choose_device(args.device)
    model.to(device).eval()
    examples = collect_examples(
        lm,
        args.corpus_dir,
        None,
        0,
        args.max_sentences_per_source,
        args.max_examples_per_source,
        config["context_window"],
        0,
    )

    def metrics(sample):
        return {
            "ngram": evaluate_ngram_baseline(lm, sample, 10),
            "neural": evaluate_model(model, sample, device),
            "hybrid": evaluate_hybrid_rerank(
                model,
                lm,
                sample,
                device,
                args.pool_k,
                (args.rank_penalty,),
                args.lock_first,
            )[0],
        }

    by_source = {}
    for source_id, source in enumerate(examples.source_names):
        if examples.total_targets_by_source.get(source, 0) == 0:
            continue
        indexes = torch.nonzero(
            examples.source_ids == source_id, as_tuple=False
        ).flatten()
        by_source[source] = metrics(
            subset_example_set(examples, indexes, source_name=source)
        )
    if not by_source:
        raise ValueError("evaluation collection has no targets")
    macro = {
        method: {
            metric: sum(source[method][metric] for source in by_source.values())
            / len(by_source)
            for metric in (
                "top1_all_targets",
                "top3_all_targets",
                "top5_all_targets",
                "mrr_all_targets",
            )
        }
        for method in ("ngram", "neural", "hybrid")
    }
    if checkpoint_hash != sha256_file(args.checkpoint):
        raise ValueError(
            "checkpoint changed during evaluation; use an immutable snapshot"
        )
    return {
        "data_provenance": provenance,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": str(args.checkpoint),
        "training_state": checkpoint.get("training_state"),
        "model": config,
        "device": str(device),
        "torch": torch.__version__,
        "policy": {
            "pool_k": args.pool_k,
            "rank_penalty": args.rank_penalty,
            "lock_first": args.lock_first,
        },
        "selection": {
            "skip_sentences_per_source": 0,
            "max_sentences_per_source": args.max_sentences_per_source,
            "max_examples_per_source": args.max_examples_per_source,
        },
        "metric_scope": "teacher-forced Bangla word prediction; unknown targets count as misses; not correction accuracy",
        "collection": example_set_report(examples),
        "metrics": metrics(examples),
        "by_source": by_source,
        "macro_by_source": macro,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-sentences-per-source", type=int, default=6000)
    parser.add_argument("--max-examples-per-source", type=int, default=40000)
    parser.add_argument("--pool-k", type=int, default=64)
    parser.add_argument("--rank-penalty", type=float, default=0.5)
    parser.add_argument("--lock-first", action="store_true")
    args = parser.parse_args()
    if (
        min(args.max_sentences_per_source, args.max_examples_per_source, args.pool_k)
        < 1
    ):
        parser.error("collection limits and pool size must be positive")
    if not math.isfinite(args.rank_penalty) or args.rank_penalty < 0:
        parser.error("rank penalty must be finite and nonnegative")
    result = evaluate(args)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
