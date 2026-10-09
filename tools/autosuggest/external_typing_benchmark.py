"""Prepare a fresh external comment benchmark, then evaluate a frozen selection.

This is next-word candidate ranking on noisy references, not correction gold.
The external text is never admitted to training. A selection receipt must bind
both neural checkpoints before evaluation is allowed.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import unicodedata
from tools.corpus.provenance import sha256_file, write_json, load_partition, digest_json
from tools.autosuggest.eval_ngram_lm import (
    iter_eval_sentence_tokens,
    NgramLm,
    model_recent_context,
)
from tools.autocorrect.bangla_lexicon_utils import iter_bangla_tokens


def verify_selection(selection, paths):
    if selection.get("selected_on") != "internal-validation-only":
        raise ValueError("selection must be frozen before external testing")
    for name, path in paths.items():
        if selection.get(name + "_sha256") != sha256_file(path):
            raise ValueError(f"selection does not match {name}")


def words(text):
    result = []
    current = []
    for ch in unicodedata.normalize("NFC", text):
        if unicodedata.category(ch)[0] in "LMN" or ch in "\u200c\u200d":
            current.append(ch)
        elif current:
            result.append("".join(current))
            current = []
    if current:
        result.append("".join(current))
    return result


def prepare(args):
    receipt = json.loads((args.source / "acquisition.json").read_text())
    source = args.source / "bengali_nostalgia_labeled.csv"
    if sha256_file(source) != receipt["files"][source.name]["sha256"]:
        raise ValueError("external source changed")
    train = load_partition(args.corpus / "train", allowed=("train",))
    if not train:
        raise ValueError("verified training partition required")
    if args.output.exists():
        raise ValueError("refusing to overwrite external benchmark")
    examples = {}
    normalized_to_ids = {}
    for row in csv.DictReader(source.open()):
        tokens = words(row["raw_text"])
        if len(tokens) < 4:
            continue
        key = digest_json(tokens)
        if key in examples:
            continue
        # Fix a genuinely contextual target: at least two preceding words.
        position = 2 + int(key[:16], 16) % (len(tokens) - 2)
        examples[key] = {
            "id": key,
            "source_id": row["id"],
            "context_tokens": tokens[max(0, position - 16) : position],
            "target": tokens[position],
            "position": position,
            "domain": "bangla-nostalgia-comments",
        }
        bangla = " ".join(iter_bangla_tokens(row["raw_text"]))
        normalized_to_ids.setdefault(unicodedata.normalize("NFC", bangla), set()).add(
            key
        )
    matched = set()
    scanned = 0
    for _, tokens in iter_eval_sentence_tokens(
        args.corpus / "train",
        sources=None,
        skip_sentences_per_source=0,
        max_sentences_per_source=None,
    ):
        scanned += 1
        text = unicodedata.normalize("NFC", " ".join(tokens))
        matched.update(normalized_to_ids.get(text, ()))
    selected = [examples[k] for k in sorted(examples) if k not in matched][: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "version": 1,
        "kind": "obadh-external-typing-benchmark",
        "role": "test-only",
        "source": receipt,
        "sha256": sha256_file(args.output),
        "rows": len(selected),
        "unique_eligible_comments": len(examples),
        "exact_training_matches_excluded": len(matched),
        "training_rows_scanned": scanned,
        "training_dataset_id": train["dataset_id"],
        "selection": "lowest normalized-token SHA256; one fixed target/comment; min four words; min two context words",
        "normalization": "NFC; Unicode letters/marks/numbers/joiners; punctuation omitted; Latin retained",
        "overlap_scope": "whole-comment Bangla token sequence against training sentences; not substring or near-duplicate detection",
        "limitations": "no correction labels; nostalgia topic skew; raw references contain dialect and typing errors",
    }
    write_json(Path(str(args.output) + ".manifest.json"), manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "source"}, indent=2))


def evaluate(args):
    import torch
    from collections import Counter
    from tools.autosuggest.subword_lm import make_model, ModelConfig
    from tools.autosuggest.eval_subword_candidates import BPETokenizer
    from tools.autosuggest.teacher_candidates import candidate_encoding
    from tools.autosuggest.train_next_word_lm import NextWordLm, score_candidate_pool
    from tools.autosuggest.decode_subword_words import WordDecoder
    from tools.corpus.provenance import verify_artifact_training

    torch.set_num_threads(1)
    if args.output.exists():
        raise ValueError("refusing to overwrite frozen evaluation")
    manifest = json.loads(Path(str(args.bank) + ".manifest.json").read_text())
    selection = json.loads(args.selection.read_text())
    if manifest["role"] != "test-only" or manifest["sha256"] != sha256_file(args.bank):
        raise ValueError("immutable external test bank required")
    bound_paths = {
        "word_checkpoint": args.word_checkpoint,
        "subword_checkpoint": args.subword_checkpoint,
        "research_model": args.model,
        "shipped_model": args.shipped_model,
        "bank": args.bank,
        "tokenizer": args.subword_data / "tokenizer.json",
    }
    verify_selection(selection, bound_paths)
    lm = NgramLm(args.model)
    shipped = NgramLm(args.shipped_model)
    word_state = torch.load(
        args.word_checkpoint, map_location="cpu", weights_only=False
    )
    c = word_state["config"]
    training = verify_artifact_training(args.model)
    provenance = word_state.get("report", {}).get("data_provenance", {})
    if (
        not training
        or training["dataset_id"] != manifest["training_dataset_id"]
        or provenance.get("status") != "partition_verified"
        or provenance.get("training_corpus") != training
    ):
        raise ValueError(
            "word checkpoint and external overlap audit must use the same verified training data"
        )
    if (
        word_state["report"].get("artifact", {}).get("sha256")
        != sha256_file(args.model)
        or c["vocab_size"] != lm.vocab_size
    ):
        raise ValueError("word checkpoint retrieval vocabulary mismatch")
    word = NextWordLm(
        vocab_size=c["vocab_size"],
        context_len=c["context_window"],
        embedding_dim=c["embedding_dim"],
        hidden_dim=c["hidden_dim"],
        architecture=c["architecture"],
        dropout=0,
        transformer_layers=c["transformer_layers"],
        transformer_heads=c["transformer_heads"],
        transformer_padding_mode=c.get("transformer_padding_mode", "unmasked-v0"),
        transformer_initialization=c.get("transformer_initialization", "legacy-v0"),
    )
    word.load_state_dict(word_state["state_dict"])
    word.to(args.device).eval()
    state = torch.load(args.subword_checkpoint, map_location="cpu", weights_only=False)
    subword = make_model(ModelConfig(**state["config"]))
    subword.load_state_dict(state["state_dict"])
    subword.to(args.device).eval()
    data = json.loads((args.subword_data / "manifest.json").read_text())
    if data["tokenizer_sha256"] != sha256_file(
        args.subword_data / "tokenizer.json"
    ) or state["contract"]["data_id"] != digest_json(data):
        raise ValueError("subword data contract mismatch")
    if data["dataset_id"] != training["dataset_id"]:
        raise ValueError("subword training differs from external overlap audit")
    tokenizer = BPETokenizer(args.subword_data / "tokenizer.json")
    decoder_policy = selection.get("decoder", {})
    if decoder_policy != {"beam": 16, "max_pieces": 12}:
        raise ValueError("unrecognized frozen decoder policy")
    decoder = WordDecoder(tokenizer.tokenizer, subword, args.device, **decoder_policy)
    boundary = [
        i
        for i in range(len(tokenizer))
        if (s := tokenizer.decode([i])) and (s[0].isspace() or s[0] in "।,.;:!?…")
    ] + [2]
    counts = Counter()
    pairs = []
    with torch.inference_mode():
        for line in args.bank.open():
            row = json.loads(line)
            target = row["target"]
            context = row["context_tokens"]
            counts["total"] += 1
            recent = model_recent_context(
                [1] + [lm.token_id(t) for t in context], max_context=c["context_window"]
            )
            candidates = lm.suggest_ids(recent, 64)
            candidate_text = [lm.token_text(i) for i in candidates]
            counts["candidate_pool_hit"] += target in candidate_text
            counts["word_vocab_oov"] += lm.token_id(target) <= 2
            tensor = torch.tensor(
                [([0] * c["context_window"] + recent)[-c["context_window"] :]],
                device=args.device,
            )
            word_scores = score_candidate_pool(
                word, tensor, torch.tensor([candidates], device=args.device)
            )[0].tolist()
            if not torch.isfinite(torch.tensor(word_scores)).all():
                raise ValueError("nonfinite word score")
            all_word_scores = word(tensor)[0]
            all_word_scores[:3] = -torch.inf
            word_generated = [
                lm.token_text(i) for i in all_word_scores.topk(5).indices.tolist()
            ]
            generated, _ = decoder.predict(" ".join(context))
            subword_generated = [text for text, _ in generated]
            sub_scores = []
            for start in range(0, len(candidates), 16):
                part = candidate_text[start : start + 16]
                encoded = [
                    candidate_encoding(
                        tokenizer, " ".join(context), t, subword.config.sequence_length
                    )
                    for t in part
                ]
                width = max(len(x[0]) for x in encoded)
                ids = torch.zeros(
                    (len(encoded), width), dtype=torch.long, device=args.device
                )
                for i, (tokens, _) in enumerate(encoded):
                    ids[i, : len(tokens)] = torch.tensor(tokens, device=args.device)
                logits = subword(ids).float()
                for i, (tokens, begin) in enumerate(encoded):
                    positions = logits[i, begin - 1 : len(tokens) - 1]
                    labels = torch.tensor(tokens[begin:], device=args.device)
                    score = (
                        positions.gather(1, labels[:, None]).squeeze(1)
                        - positions.logsumexp(-1)
                    ).sum()
                    tail = logits[i, len(tokens) - 1]
                    score += tail[boundary].logsumexp(-1) - tail.logsumexp(-1)
                    if not torch.isfinite(score):
                        raise ValueError("nonfinite score")
                    sub_scores.append(float(score))
            legacy = shipped.suggest_ids(
                [1] + [shipped.token_id(t) for t in context], 64
            )
            orders = {
                "shipped_ngram": [shipped.token_text(i) for i in legacy],
                "research_ngram": candidate_text,
                "word_transformer": [
                    candidate_text[i]
                    for i in sorted(
                        range(len(candidates)), key=lambda i: (-word_scores[i], i)
                    )
                ],
                "subword_transformer": [
                    candidate_text[i]
                    for i in sorted(
                        range(len(candidates)), key=lambda i: (-sub_scores[i], i)
                    )
                ],
                "word_transformer_generated": word_generated,
                "subword_transformer_generated": subword_generated,
            }
            outcomes = {}
            for method, rank in orders.items():
                for k in (1, 3, 5):
                    counts[f"{method}_top{k}"] += target in rank[:k]
                outcomes[method] = target in rank[:3]
            pairs.append(outcomes)
    if not pairs or len(pairs) != manifest["rows"]:
        raise ValueError("external benchmark row count mismatch or empty bank")
    verify_selection(selection, bound_paths)
    result = {
        "benchmark": manifest,
        "selection": selection,
        "counts": dict(counts),
        "rates": {k: v / counts["total"] for k, v in counts.items() if k != "total"},
        "research_model_sha256": sha256_file(args.model),
        "shipped_model_sha256": sha256_file(args.shipped_model),
        "scope": "fresh external next-word prediction; ranking uses a fixed research 64-candidate pool; generated variants use direct word logits or bounded subword beam; not corrections or user acceptance",
    }
    # Paired bootstrap avoids treating the two systems as independent samples.
    import random

    result["paired_top3_minus_shipped_bootstrap_95pct"] = {}
    for method in (
        "word_transformer",
        "subword_transformer",
        "word_transformer_generated",
        "subword_transformer_generated",
    ):
        rng = random.Random(17)
        deltas = []
        for _ in range(2000):
            sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            deltas.append(
                sum(int(x[method]) - int(x["shipped_ngram"]) for x in sample)
                / len(sample)
            )
        deltas.sort()
        result["paired_top3_minus_shipped_bootstrap_95pct"][method] = [
            deltas[50],
            deltas[1949],
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    for name in ("source", "corpus", "output"):
        a.add_argument("--" + name, type=Path, required=True)
    a.add_argument("--limit", type=int, default=1000)
    a = sub.add_parser("evaluate")
    for name in (
        "bank",
        "selection",
        "model",
        "shipped-model",
        "word-checkpoint",
        "subword-checkpoint",
        "subword-data",
        "output",
    ):
        a.add_argument("--" + name, type=Path, required=True)
    a.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if hasattr(args, "limit") and args.limit < 1:
        p.error("limit must be positive")
    {"prepare": prepare, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
