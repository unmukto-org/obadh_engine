"""Compare a subword checkpoint on an immutable validation candidate bank."""

from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import torch
from tokenizers import Tokenizer
from tools.autosuggest.subword_lm import ModelConfig, make_model
from tools.autosuggest.teacher_candidates import candidate_encoding
from tools.corpus.provenance import sha256_file, digest_json, write_json


class BPETokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, path):
        self.tokenizer = Tokenizer.from_file(str(path))

    def __call__(self, text, **kwargs):
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return {"input_ids": encoded.ids, "offset_mapping": encoded.offsets}

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def __len__(self):
        return self.tokenizer.get_vocab_size()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data", "bank", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    torch.set_num_threads(1)
    receipt = json.loads((args.data / "manifest.json").read_text())
    manifest = json.loads(Path(str(args.bank) + ".manifest.json").read_text())
    if manifest["split"] != "validation" or manifest["sha256"] != sha256_file(
        args.bank
    ):
        raise ValueError("immutable validation bank required")
    if receipt["tokenizer_sha256"] != sha256_file(args.data / "tokenizer.json"):
        raise ValueError("tokenizer mismatch")
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["contract"]["data_id"] != digest_json(receipt):
        raise ValueError("checkpoint data mismatch")
    if manifest["provenance"]["training_corpus"]["dataset_id"] != receipt["dataset_id"]:
        raise ValueError("candidate bank belongs to another dataset")
    model = make_model(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    tokenizer = BPETokenizer(args.data / "tokenizer.json")
    boundary = [
        i
        for i in range(len(tokenizer))
        if (s := tokenizer.decode([i])) and (s[0].isspace() or s[0] in "।,.;:!?…")
    ] + [2]
    counts = {}
    started = time.monotonic()
    with torch.inference_mode():
        for line in args.bank.open():
            row = json.loads(line)
            encoded = [
                candidate_encoding(
                    tokenizer, row["context"], word, model.config.sequence_length
                )
                for word in row["candidates"]
            ]
            length = max(len(ids) for ids, _ in encoded)
            ids = torch.zeros(
                (len(encoded), length), dtype=torch.long, device=args.device
            )
            for i, (tokens, _) in enumerate(encoded):
                ids[i, : len(tokens)] = torch.tensor(tokens, device=args.device)
            logits = model(ids).float()
            scores = []
            for i, (tokens, start) in enumerate(encoded):
                positions = logits[i, start - 1 : len(tokens) - 1]
                targets = torch.tensor(tokens[start:], device=args.device)
                score = (
                    positions.gather(1, targets[:, None]).squeeze(1)
                    - positions.logsumexp(-1)
                ).sum()
                tail = logits[i, len(tokens) - 1]
                score += tail[boundary].logsumexp(-1) - tail.logsumexp(-1)
                scores.append(float(score))
            if not all(torch.isfinite(torch.tensor(scores))):
                raise ValueError("invalid subword scores")
            c = counts.setdefault(row["source"], Counter())
            c["total"] += 1
            for method, rank in [
                ("ngram", list(range(len(encoded)))),
                ("subword", sorted(range(len(encoded)), key=lambda i: (-scores[i], i))),
            ]:
                for k in (1, 3, 5):
                    c[f"{method}_top{k}"] += row["target_id"] > 2 and row[
                        "target_id"
                    ] in [row["candidate_ids"][i] for i in rank[:k]]
    if checkpoint_hash != sha256_file(args.checkpoint):
        raise ValueError("checkpoint changed during evaluation")
    result = {
        "checkpoint_sha256": checkpoint_hash,
        "step": checkpoint["step"],
        "config": checkpoint["config"],
        "bank_sha256": manifest["sha256"],
        "counts_by_source": {s: dict(c) for s, c in counts.items()},
        "rates_by_source": {
            s: {k: v / c["total"] for k, v in c.items() if k != "total"}
            for s, c in counts.items()
        },
        "seconds": time.monotonic() - started,
        "device": args.device,
        "scoring": "complete word probability including boundary event",
        "scope": "validation ranking in fixed ngram pool; does not measure open-vocabulary generation or autocorrect",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
