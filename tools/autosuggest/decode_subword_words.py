"""Bounded, open-vocabulary next-word decoding for the research subword model.

The beam scores complete words including their boundary probability. It does
not renormalize away disallowed tokens or silently force malformed UTF-8 into
output. This reference implementation is an evaluation tool, not the keyboard
runtime: incremental KV caching and end-to-end cancellation remain separate.
"""

from __future__ import annotations
import math
import unicodedata


def byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = list(values)
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): value for code, value in zip(codes, values)}


def lexical_word(raw):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text or not any(unicodedata.category(c)[0] in "LN" for c in text):
        return None
    if any(
        unicodedata.category(c)[0] not in "LMN" and c not in "\u200c\u200d"
        for c in text
    ):
        return None
    return unicodedata.normalize("NFC", text)


class WordDecoder:
    def __init__(self, tokenizer, model, device, beam=16, max_pieces=12, scorer=None):
        import torch

        if (
            not 1 <= beam <= 64
            or not 1 <= max_pieces <= 32
            or max_pieces >= model.config.sequence_length
        ):
            raise ValueError("decoder budget out of bounds")
        self.tokenizer = tokenizer
        self.tokenizer.encode_special_tokens = True
        self.model = model
        self.device = device
        self.beam = beam
        self.max_pieces = max_pieces
        self.scorer = scorer
        alphabet = byte_alphabet()
        vocab = tokenizer.get_vocab()
        self.pieces = {
            index: bytes(alphabet[c] for c in text)
            for text, index in vocab.items()
            if index > 3
        }
        self.boundary = [2] + [
            i
            for i, b in self.pieces.items()
            if b[:1] in (b" ", b"\n", b"\t", b"\r")
            or b.startswith(("।".encode(), b",", b".", b";", b":", b"!", b"?"))
        ]

        def possible(b):
            return bool(b) and (b[0] >= 128 or chr(b[0]).isalnum())

        self.initial = [i for i, b in self.pieces.items() if possible(b)]
        self.spaced = [
            i
            for i, b in self.pieces.items()
            if b.startswith(b" ") and (len(b) == 1 or possible(b[1:]))
        ]
        self.continuation = self.initial
        self.allowed_ids = {
            "initial": self.initial,
            "spaced": self.spaced,
            "continuation": self.continuation,
        }
        self.allowed = {
            name: torch.tensor(ids, device=device)
            for name, ids in [
                ("initial", self.initial),
                ("spaced", self.spaced),
                ("continuation", self.continuation),
            ]
        }
        self.boundary_tensor = torch.tensor(self.boundary, device=device)

    def predict(self, context, k=5, prefix=""):
        import torch

        if not 1 <= k <= self.beam:
            raise ValueError("output exceeds beam budget")
        prefix = unicodedata.normalize("NFC", prefix)
        prefix_bytes = prefix.encode("utf-8")
        if prefix and (len(prefix_bytes) > 256 or lexical_word(prefix_bytes) != prefix):
            raise ValueError("prefix must be a bounded lexical fragment")
        ids = [1] + self.tokenizer.encode(context, add_special_tokens=False).ids
        # Reserve capacity once; do not change the context during decoding.
        ids = ids[-(self.model.config.sequence_length - self.max_pieces) :]
        active = [(0.0, [], b"")]
        completed = {}
        calls = 0
        with torch.inference_mode():
            if self.scorer is not None:
                self.scorer.begin(ids)
            for step in range(self.max_pieces + 1):
                if self.scorer is not None:
                    logits = torch.as_tensor(
                        self.scorer.score([tokens for _, tokens, _ in active]),
                        device=self.device,
                    )
                else:
                    batch = torch.tensor(
                        [ids + tokens for _, tokens, _ in active], device=self.device
                    )
                    x = self.model.embedding(batch)
                    length = batch.shape[1]
                    for block in self.model.blocks:
                        x = block(
                            x, self.model.cos[:, :, :length], self.model.sin[:, :, :length]
                        )
                    logits = self.model.norm(x[:, -1]) @ self.model.embedding.weight.T
                logp = logits.float().log_softmax(-1)
                calls += 1
                if not torch.isfinite(logp).all():
                    raise ValueError("nonfinite decoder output")
                endings = logp[:, self.boundary_tensor].logsumexp(-1).tolist()
                for index, (score, tokens, raw) in enumerate(active):
                    word = lexical_word(raw)
                    if word and word.startswith(prefix):
                        value = score + endings[index]
                        if word in completed:
                            old = completed[word]
                            high = max(old, value)
                            value = high + math.log(
                                math.exp(old - high) + math.exp(value - high)
                            )
                        completed[word] = value
                if step == self.max_pieces:
                    break
                policy = (
                    "spaced"
                    if step == 0 and context
                    else ("initial" if step == 0 else "continuation")
                )
                expanded = []
                if not prefix:
                    allowed = self.allowed[policy]
                    values, indices = logp[:, allowed].topk(
                        min(self.beam, len(allowed)), dim=-1
                    )
                    values, indices = values.tolist(), allowed[indices].tolist()
                else:
                    values, indices = [], []
                    for row, (_, _, raw) in enumerate(active):
                        # Filter BEFORE top-k so an unlikely typed prefix is not
                        # lost behind unconstrained continuations. Keep original
                        # model probabilities, including the completion boundary.
                        if raw.startswith(prefix_bytes):
                            allowed = self.allowed[policy]
                        else:
                            allowed_ids = []
                            for token in self.allowed_ids[policy]:
                                piece = self.pieces[token]
                                candidate = raw + (piece[1:] if step == 0 and context else piece)
                                if candidate.startswith(prefix_bytes) or prefix_bytes.startswith(candidate):
                                    allowed_ids.append(token)
                            allowed = torch.tensor(allowed_ids, dtype=torch.long, device=self.device)
                        scores, positions = logp[row, allowed].topk(min(self.beam, len(allowed)))
                        values.append(scores.tolist())
                        indices.append(allowed[positions].tolist())
                for row, (score, tokens, raw) in enumerate(active):
                    for value, index in zip(values[row], indices[row]):
                        piece = self.pieces[index]
                        if step == 0 and context:
                            piece = piece[1:]
                        expanded.append((score + value, tokens + [index], raw + piece))
                active = sorted(expanded, key=lambda x: (-x[0], x[1]))[: self.beam]
                if not active:
                    break
                # A path's uncompleted probability is an upper bound on any
                # one completion. Keep all beams for a fixed, auditable budget;
                # aggregation across alternate BPE paths precludes naive stop.
        result = sorted(completed.items(), key=lambda item: (-item[1], item[0]))[:k]
        return result, {
            "model_calls": calls,
            "beam": self.beam,
            "max_pieces": self.max_pieces,
            "prefix_constrained": bool(prefix),
        }


def main():
    import argparse, json, time
    from collections import Counter
    from pathlib import Path
    import torch
    from tokenizers import Tokenizer
    from tools.autosuggest.subword_lm import ModelConfig, make_model
    from tools.autosuggest.eval_ngram_lm import NgramLm
    from tools.corpus.provenance import (
        sha256_file,
        digest_json,
        write_json,
        verify_artifact_training,
    )

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data", "bank", "model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--beam", type=int, default=16)
    p.add_argument("--max-pieces", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise ValueError("refusing to replace evaluation")
    receipt = json.loads((args.data / "manifest.json").read_text())
    bank = json.loads(Path(str(args.bank) + ".manifest.json").read_text())
    if bank["split"] != "validation" or bank["sha256"] != sha256_file(args.bank):
        raise ValueError("verified validation required")
    if receipt["tokenizer_sha256"] != sha256_file(args.data / "tokenizer.json"):
        raise ValueError("tokenizer mismatch")
    checkpoint_hash = sha256_file(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        state["contract"]["data_id"] != digest_json(receipt)
        or bank["provenance"]["training_corpus"]["dataset_id"] != receipt["dataset_id"]
    ):
        raise ValueError("training lineage mismatch")
    if verify_artifact_training(args.model) != bank["provenance"]["training_corpus"]:
        raise ValueError("retrieval training provenance mismatch")
    model = make_model(ModelConfig(**state["config"]))
    model.load_state_dict(state["state_dict"])
    model.to(args.device).eval()
    decoder = WordDecoder(
        Tokenizer.from_file(str(args.data / "tokenizer.json")),
        model,
        args.device,
        args.beam,
        args.max_pieces,
    )
    lm = NgramLm(args.model)
    counts = {}
    started = time.monotonic()
    for line in args.bank.open():
        row = json.loads(line)
        if [lm.token_text(i) for i in row["candidate_ids"]] != row["candidates"]:
            raise ValueError("candidate vocabulary mismatch")
        rank, _ = decoder.predict(row["context"])
        rank = [x[0] for x in rank]
        target = lm.token_text(row["target_id"])
        c = counts.setdefault(row["source"], Counter())
        c["total"] += 1
        for k in (1, 3, 5):
            c[f"generated_top{k}"] += row["target_id"] > 2 and target in rank[:k]
    if checkpoint_hash != sha256_file(args.checkpoint):
        raise ValueError("checkpoint changed")
    report = {
        "checkpoint_sha256": checkpoint_hash,
        "bank_sha256": bank["sha256"],
        "model_sha256": sha256_file(args.model),
        "config": state["config"],
        "decoder": {"beam": args.beam, "max_pieces": args.max_pieces},
        "counts_by_source": {s: dict(c) for s, c in counts.items()},
        "rates_by_source": {
            s: {k: v / c["total"] for k, v in c.items() if k != "total"}
            for s, c in counts.items()
        },
        "seconds": time.monotonic() - started,
        "scope": "open-vocabulary generation; legacy-bank OOV targets still count as misses; no test selection",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
