"""Paired diagnostic word prediction on an already-observed, fixed text bank.

This is a development comparison, never fresh-test or correction certification.
The previously frozen 16-beam/12-piece decoding budget stays unchanged.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from tools.corpus.provenance import sha256_file, write_json


def completion_rows(rows, prefix_graphemes):
    excluded = {"fully_typed_at_requested_prefix": 0, "invalid_lexical_target": 0}
    if not prefix_graphemes:
        return rows, excluded
    if prefix_graphemes not in (1, 2, 3):
        raise ValueError("invalid prefix length")
    import regex
    from tools.autosuggest.decode_subword_words import lexical_word
    eligible = []
    for row in rows:
        target = row["target"]
        if lexical_word(target.encode("utf-8")) != target:
            excluded["invalid_lexical_target"] += 1
            continue
        clusters = regex.findall(r"\X", target)
        if len(clusters) <= prefix_graphemes:
            excluded["fully_typed_at_requested_prefix"] += 1
            continue
        eligible.append(dict(row, typed_prefix="".join(clusters[:prefix_graphemes])))
    return eligible, excluded


def main():
    import torch
    from tokenizers import Tokenizer
    from tools.autosuggest.subword_lm import ModelConfig, make_model
    from tools.autosuggest.decode_subword_words import WordDecoder

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, help="name=path")
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--bank-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prefix-graphemes", type=int, choices=(0, 1, 2, 3), default=0)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite comparison")
    if sha256_file(args.bank) != args.bank_sha256:
        raise ValueError("bank digest mismatch")
    rows = [json.loads(line) for line in args.bank.open(encoding="utf-8")]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate bank")
    original_count = len(rows)
    rows, excluded = completion_rows(rows, args.prefix_graphemes)
    if not rows:
        raise ValueError("no eligible completion cases")
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    torch.set_num_threads(8)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer_hash = sha256_file(args.tokenizer)
    report = {
        "scope": __doc__,
        "bank_sha256": args.bank_sha256,
        "tokenizer_sha256": tokenizer_hash,
        "examples": len(rows),
        "bank_examples": original_count,
        "prefix_graphemes": args.prefix_graphemes,
        "excluded": excluded,
        "decoder_sha256": sha256_file(Path(__file__).with_name("decode_subword_words.py")),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "beam": 16,
        "max_pieces": 12,
        "models": {},
    }
    for specification in args.checkpoint:
        name, path = specification.split("=", 1)
        if (
            not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name)
            or name in report["models"]
        ):
            raise ValueError("invalid model name")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["data_provenance"]["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError("checkpoint tokenizer mismatch")
        model = make_model(ModelConfig(**checkpoint["config"])).to(args.device).eval()
        model.load_state_dict(checkpoint["state_dict"])
        decoder = WordDecoder(tokenizer, model, args.device, beam=16, max_pieces=12)
        hits = {1: 0, 3: 0, 5: 0}
        started = time.monotonic()
        with (args.output / (name + ".jsonl")).open("w", encoding="utf-8") as out:
            for i, row in enumerate(rows):
                candidates, _ = decoder.predict(
                    row.get("context") or " ".join(row["context_tokens"]),
                    prefix=row.get("typed_prefix", ""),
                )
                words = [word for word, score in candidates]
                if not all(isinstance(word, str) for word in words):
                    raise ValueError("decoder output contract mismatch")
                for k in hits:
                    hits[k] += row["target"] in words[:k]
                out.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "target": row["target"],
                            "typed_prefix": row.get("typed_prefix", ""),
                            "predictions": words,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if (i + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "model": name,
                                "examples": i + 1,
                                "seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
        report["models"][name] = {
            "checkpoint_sha256": sha256_file(Path(path)),
            "step": checkpoint["step"],
            "top_k": {str(k): v / len(rows) for k, v in hits.items()},
            "seconds": time.monotonic() - started,
        }
        write_json(args.output / "report.json", report)
        print(json.dumps(report["models"][name]), flush=True)
        del model, decoder, checkpoint
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    (args.output / ".building").unlink()


if __name__ == "__main__":
    main()
