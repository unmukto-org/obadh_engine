"""Teacher-forced validation on preserved book text, including punctuation.

Token metrics diagnose language modeling; they are not word suggestion accuracy,
punctuation restoration accuracy, or a grammar correction release gate.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import unicodedata

from tools.corpus.provenance import sha256_file, write_json


def main():
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tokenizers import Tokenizer
    from tools.autosuggest.literary_data import LiteraryData
    from tools.autosuggest.subword_lm import ModelConfig, make_model

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite validation")
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    tokenizer_hash = sha256_file(args.data / "tokenizer.json")
    receipt = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    data = LiteraryData(
        args.data,
        tokenizer_sha256=tokenizer_hash,
        sequence_length=receipt["sequence_length"],
    )
    metadata = json.loads((args.data / "validation.json").read_text(encoding="utf-8"))
    group_ids = np.asarray(metadata["group_ids"])
    punctuation = []
    for i in range(4, tokenizer.get_vocab_size()):
        piece = tokenizer.decode([i]).strip()
        if piece and all(unicodedata.category(c).startswith("P") for c in piece):
            punctuation.append(i)
    punctuation_ids = torch.tensor(punctuation, device="cuda")
    torch.set_num_threads(8)
    report = {
        "scope": __doc__,
        "data_id": receipt["data_id"],
        "punctuation_policy": "targets whose individual BPE token decodes exclusively to Unicode punctuation after whitespace stripping",
        "models": {},
    }
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        if name in report["models"]:
            raise ValueError("duplicate model name")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["data_provenance"]["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError("tokenizer mismatch")
        model = make_model(ModelConfig(**state["config"])).cuda().eval()
        model.load_state_dict(state["state_dict"])
        totals = np.zeros((len(metadata["groups"]), 3), dtype=np.float64)
        punct = np.zeros(3)
        with torch.inference_mode():
            for start in range(0, len(data.blocks["validation"]), args.batch):
                rows = torch.from_numpy(
                    np.asarray(
                        data.blocks["validation"][start : start + args.batch],
                        dtype=np.int64,
                    )
                ).cuda()
                x, y = rows[:, :-1], rows[:, 1:]
                valid = y != 0
                logits = model(x)
                losses = F.cross_entropy(
                    logits.flatten(0, 1), y.flatten(), ignore_index=0, reduction="none"
                ).view_as(y)
                correct = (logits.argmax(-1) == y) & valid
                stats = (
                    torch.stack((valid.sum(1), losses.sum(1), correct.sum(1)), dim=1)
                    .cpu()
                    .numpy()
                )
                np.add.at(totals, group_ids[start : start + len(rows)], stats)
                mask = torch.isin(y, punctuation_ids) & valid
                punct += np.array(
                    [
                        mask.sum().item(),
                        losses[mask].sum().item(),
                        correct[mask].sum().item(),
                    ]
                )
        total = totals.sum(0)
        result = {
            "checkpoint_sha256": sha256_file(Path(path)),
            "tokens": int(total[0]),
            "token_loss": total[1] / total[0],
            "token_top1": total[2] / total[0],
            "macro_author_token_loss": float(np.mean(totals[:, 1] / totals[:, 0])),
            "punctuation_token_targets": int(punct[0]),
            "punctuation_token_loss": punct[1] / punct[0] if punct[0] else None,
            "punctuation_token_top1": punct[2] / punct[0] if punct[0] else None,
            "authors": [
                {"names": group, "tokens": int(v[0]), "token_loss": v[1] / v[0]}
                for group, v in zip(metadata["groups"], totals)
            ],
        }
        report["models"][name] = result
        print(json.dumps(result), flush=True)
        del state, model
        torch.cuda.empty_cache()
    write_json(args.output, report)


if __name__ == "__main__":
    main()
