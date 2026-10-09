"""Auditable offline LLM candidate scoring; never runs in the keyboard.

Prepare a verified train/validation bank, then score immutable rows into a
resumable SQLite cache. The teacher sees exactly the student's known context.
Gold insertion is permitted only in training. Evaluation retains OOV and
retrieval misses. No test data or generated prose enters student training.
"""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from tools.corpus.provenance import digest_json, load_partition, sha256_file, write_json
from tools.autosuggest.common import BOS_ID, PAD_ID, UNK_ID
from tools.autosuggest.eval_ngram_lm import (
    NgramLm,
    iter_eval_sentence_tokens,
    model_recent_context,
)


def prepare(args):
    partition = load_partition(args.corpus, allowed=("train", "validation"))
    if partition is None:
        raise ValueError("verified corpus partition required")
    # Verifies the retrieval model was built from this dataset, training only.
    from tools.corpus.provenance import neural_training_provenance

    provenance = neural_training_provenance(
        args.model, args.corpus.parent / "train", args.corpus.parent / "validation"
    )
    if args.output.exists() or Path(str(args.output) + ".manifest.json").exists():
        raise ValueError("refusing to replace a candidate bank")
    lm = NgramLm(args.model)
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output) + ".building")
    with temporary.open("x", encoding="utf-8") as out:
        for source, tokens in iter_eval_sentence_tokens(
            args.corpus,
            sources=None,
            skip_sentences_per_source=0,
            max_sentences_per_source=args.sentences,
        ):
            if not tokens:
                continue
            sentence_hash = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
            position = int(sentence_hash[:16], 16) % len(tokens)
            target = lm.token_id(tokens[position])
            context = model_recent_context(
                [BOS_ID] + [lm.token_id(t) for t in tokens[:position]],
                max_context=args.context,
            )
            candidates = lm.suggest_ids(context, args.pool)
            if (
                partition["split"] == "train"
                and target > UNK_ID
                and target not in candidates
            ):
                candidates = candidates[: args.pool - 1] + [target]
            row = {
                "source": source,
                "sentence_sha256": sentence_hash,
                "position": position,
                "context_ids": ([PAD_ID] * args.context + context)[-args.context :],
                "context": " ".join(lm.token_text(t) for t in context if t > UNK_ID),
                "target_id": target,
                "candidate_ids": candidates,
                "candidates": [lm.token_text(t) for t in candidates],
            }
            row["id"] = digest_json(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[source] += 1
    temporary.replace(args.output)
    manifest = {
        "version": 1,
        "kind": "obadh-teacher-candidate-bank",
        "split": partition["split"],
        "provenance": provenance,
        "sha256": sha256_file(args.output),
        "rows_by_source": dict(counts),
        "policy": {
            "sentences_per_source": args.sentences,
            "one_position_per_sentence": "sha256-mod-token-count",
            "context": args.context,
            "pool": args.pool,
            "gold_inserted": partition["split"] == "train",
        },
        "pretrained_contamination": "unknown; partition verification covers our corpus only",
    }
    write_json(Path(str(args.output) + ".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def candidate_encoding(tokenizer, context, candidate, max_length):
    """Use offsets to include leading-space pieces and reject clipped targets."""
    if not candidate or not candidate.strip():
        raise ValueError("candidate must be nonempty")
    prefix = context
    text = prefix + (" " if prefix else "") + candidate
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    start = next(
        (i for i, (_, end) in enumerate(offsets) if end > len(prefix)), len(ids)
    )
    # A BOS is required for scoring a candidate with no preceding context.
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None:
        raise ValueError("teacher must expose BOS or EOS for an empty prefix")
    ids = [bos] + ids
    start += 1
    if len(ids) - start >= max_length:
        raise ValueError("candidate alone exceeds teacher context cap")
    trim = max(0, len(ids) - max_length)
    ids, start = ids[trim:], start - trim
    if start < 1 or start >= len(ids):
        raise ValueError("target has no causal context or no tokens")
    return ids, start


def score(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    manifest = json.loads(Path(str(args.bank) + ".manifest.json").read_text())
    if manifest["sha256"] != sha256_file(args.bank) or manifest["split"] not in (
        "train",
        "validation",
    ):
        raise ValueError("candidate bank digest or split is invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("this offline teacher profile requires CUDA")
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher, local_files_only=True, trust_remote_code=False
    )
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            args.teacher,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    boundary_ids = [
        i
        for i in range(len(tokenizer))
        if (text := tokenizer.decode([i], clean_up_tokenization_spaces=False))
        and (text[0].isspace() or text[0] in "।,.;:!?…")
    ]
    if tokenizer.eos_token_id is not None:
        boundary_ids.append(tokenizer.eos_token_id)
    boundary_ids = sorted(set(boundary_ids))
    if not boundary_ids:
        raise ValueError("teacher has no word boundary tokens")
    teacher_files = {
        p.name: sha256_file(p)
        for p in sorted(args.teacher.iterdir())
        if p.suffix in (".json", ".safetensors")
    }
    contract = {
        "version": 1,
        "bank_sha256": manifest["sha256"],
        "teacher_files": teacher_files,
        "revision": args.revision,
        "max_length": args.max_length,
        "scoring": "candidate-plus-word-boundary-v2",
        "boundary_ids_sha256": digest_json(boundary_ids),
        "script_sha256": sha256_file(Path(__file__)),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.output)
    db.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS scores (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    stored = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if stored and json.loads(stored[0]) != contract:
        raise ValueError("resume contract mismatch")
    db.execute(
        "INSERT OR IGNORE INTO metadata VALUES ('contract', ?)",
        (json.dumps(contract, sort_keys=True),),
    )
    db.commit()
    started = time.monotonic()
    done = 0
    with args.bank.open() as source:
        for line in source:
            row = json.loads(line)
            if db.execute("SELECT 1 FROM scores WHERE id=?", (row["id"],)).fetchone():
                continue
            if (
                args.max_rows and done >= args.max_rows
            ) or time.time() >= args.stop_epoch:
                break
            sums, means, complete = [], [], []
            for offset in range(0, len(row["candidates"]), args.batch):
                encoded = [
                    candidate_encoding(tokenizer, row["context"], c, args.max_length)
                    for c in row["candidates"][offset : offset + args.batch]
                ]
                length = max(len(x[0]) for x in encoded)
                inputs = torch.full(
                    (len(encoded), length),
                    tokenizer.pad_token_id or tokenizer.eos_token_id,
                    dtype=torch.long,
                    device="cuda",
                )
                mask = torch.zeros_like(inputs)
                owners, positions, targets = [], [], []
                for owner, (ids, start) in enumerate(encoded):
                    inputs[owner, : len(ids)] = torch.tensor(ids, device="cuda")
                    mask[owner, : len(ids)] = 1
                    for pos in range(start, len(ids)):
                        owners.append(owner)
                        positions.append(pos - 1)
                        targets.append(ids[pos])
                with torch.inference_mode():
                    hidden = teacher.model(
                        input_ids=inputs, attention_mask=mask, use_cache=False
                    ).last_hidden_state
                    selected = hidden[owners, positions]
                    probabilities = []
                    for i in range(0, len(targets), 64):
                        logits = teacher.lm_head(selected[i : i + 64]).float()
                        chosen = torch.tensor(targets[i : i + 64], device="cuda")
                        probabilities.extend(
                            (
                                logits.gather(1, chosen[:, None]).squeeze(1)
                                - logits.logsumexp(-1)
                            )
                            .cpu()
                            .tolist()
                        )
                    final_hidden = hidden[
                        torch.arange(len(encoded), device="cuda"),
                        [len(ids) - 1 for ids, _ in encoded],
                    ]
                    final_logits = teacher.lm_head(final_hidden).float()
                    boundary = (
                        (
                            final_logits[:, boundary_ids].logsumexp(-1)
                            - final_logits.logsumexp(-1)
                        )
                        .cpu()
                        .tolist()
                    )
                for owner in range(len(encoded)):
                    values = [v for o, v in zip(owners, probabilities) if o == owner]
                    total = sum(values)
                    if not values or not math.isfinite(total):
                        raise ValueError("non-finite or empty teacher score")
                    sums.append(total)
                    means.append(total / len(values))
                    complete.append(total + boundary[owner])
            payload = {
                "sum_logprob": sums,
                "mean_logprob": means,
                "complete_logprob": complete,
            }
            db.execute(
                "INSERT INTO scores VALUES (?,?)", (row["id"], json.dumps(payload))
            )
            db.commit()  # durable after every context; no 15-minute loss window
            done += 1
            if done % 50 == 0:
                print(
                    json.dumps(
                        {
                            "event": "teacher_progress",
                            "rows_this_run": done,
                            "seconds": round(time.monotonic() - started, 2),
                        }
                    ),
                    flush=True,
                )
    count = db.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    db.close()
    print(
        json.dumps(
            {
                "rows_this_run": done,
                "cached_rows": count,
                "seconds": time.monotonic() - started,
                "gpu": torch.cuda.get_device_name(),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
        ),
        flush=True,
    )


def report(args):
    manifest = json.loads(Path(str(args.bank) + ".manifest.json").read_text())
    if manifest["split"] != "validation" or manifest["sha256"] != sha256_file(
        args.bank
    ):
        raise ValueError("only verified, unmodified validation banks can be reported")
    db = sqlite3.connect(f"file:{args.scores}?mode=ro", uri=True)
    contract = json.loads(
        db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0]
    )
    if contract["bank_sha256"] != manifest["sha256"]:
        raise ValueError("teacher scores belong to another bank")
    counts = {}
    missing = 0
    for line in args.bank.open():
        row = json.loads(line)
        record = db.execute(
            "SELECT payload FROM scores WHERE id=?", (row["id"],)
        ).fetchone()
        if record is None:
            missing += 1
            continue
        scores = json.loads(record[0])
        c = counts.setdefault(row["source"], Counter())
        c["total"] += 1
        c["pool_hits"] += (
            row["target_id"] > UNK_ID and row["target_id"] in row["candidate_ids"]
        )
        for method in ("ngram", *scores.keys()):
            indexes = list(range(len(row["candidate_ids"])))
            if method != "ngram":
                indexes.sort(key=lambda i: (-scores[method][i], i))
            for k in (1, 3, 5):
                c[f"{method}_top{k}"] += row["target_id"] > UNK_ID and row[
                    "target_id"
                ] in [row["candidate_ids"][i] for i in indexes[:k]]
    result = {
        "bank": manifest,
        "teacher_contract": contract,
        "missing_rows": missing,
        "counts_by_source": {s: dict(c) for s, c in counts.items()},
        "rates_by_source": {
            s: {k: v / c["total"] for k, v in c.items() if k != "total"}
            for s, c in counts.items()
        },
        "scope": "validation candidate ranking, all targets; not grammar or autocorrect accuracy; pretrained contamination unknown",
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--model", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--sentences", type=int, default=500)
    a.add_argument("--pool", type=int, default=16)
    a.add_argument("--context", type=int, default=16)
    a = sub.add_parser("score")
    a.add_argument("--bank", type=Path, required=True)
    a.add_argument("--teacher", type=Path, required=True)
    a.add_argument("--revision", required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--batch", type=int, default=16)
    a.add_argument("--threads", type=int, default=8)
    a.add_argument("--max-length", type=int, default=96)
    a.add_argument("--max-rows", type=int, default=0)
    a.add_argument("--stop-epoch", type=float, required=True)
    a = sub.add_parser("report")
    a.add_argument("--bank", type=Path, required=True)
    a.add_argument("--scores", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for name in ("sentences", "pool", "context", "batch", "threads", "max_length"):
        if hasattr(args, name) and getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if hasattr(args, "max_rows") and args.max_rows < 0:
        p.error("max-rows must be nonnegative")
    if hasattr(args, "stop_epoch") and not math.isfinite(args.stop_epoch):
        p.error("stop-epoch must be finite")
    {"prepare": prepare, "score": score, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
