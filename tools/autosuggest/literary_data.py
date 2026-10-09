"""Verified punctuation-preserving sequences with bounded author sampling.

Coauthors/translators join one connected component so a contributor cannot gain
extra probability through multiple bylines. Components receive square-root token
weights capped at 5% of expected nonpadding targets in the literary branch.
Block probabilities compensate for each component's total target count.
No sequence crosses a source text window.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

from tools.corpus.provenance import digest_json, sha256_file, write_json


def capped_probabilities(weights, cap):
    if not weights or any(w <= 0 or not math.isfinite(w) for w in weights):
        raise ValueError("positive finite weights required")
    if not 0 < cap <= 1 or len(weights) * cap < 1 - 1e-12:
        raise ValueError("not enough independent author groups for the cap")
    remaining = set(range(len(weights)))
    result = [0.0] * len(weights)
    mass = 1.0
    while remaining:
        total = sum(weights[i] for i in remaining)
        bound = [i for i in remaining if mass * weights[i] / total > cap]
        if not bound:
            for i in remaining:
                result[i] = mass * weights[i] / total
            break
        for i in bound:
            result[i] = cap
            remaining.remove(i)
            mass -= cap
    return result


def components(bylines):
    parent = {a: a for row in bylines for a in row}

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in bylines:
        for a in row[1:]:
            parent[root(a)] = root(row[0])
    groups = defaultdict(list)
    for a in sorted(parent):
        groups[root(a)].append(a)
    return sorted(groups.values())


def prepare(args):
    import numpy as np
    from tokenizers import Tokenizer

    if args.output.exists():
        raise ValueError("refusing to overwrite literary data")
    if (args.corpus / ".building").exists():
        raise ValueError("incomplete corpus")
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] != "obadh-contextual-text" or manifest[
        "dataset_id"
    ] != digest_json({k: v for k, v in manifest.items() if k != "dataset_id"}):
        raise ValueError("invalid literary corpus identity")
    for item in manifest["files"]:
        if item["path"] not in (
            "train.jsonl.gz",
            "validation.jsonl.gz",
            "test.jsonl.gz",
        ):
            raise ValueError("invalid partition filename")
        p = args.corpus / item["path"]
        if p.stat().st_size != item["bytes"] or sha256_file(p) != item["sha256"]:
            raise ValueError("corpus digest mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if [tokenizer.token_to_id(x) for x in ("[PAD]", "[BOS]", "[EOS]", "[UNK]")] != [
        0,
        1,
        2,
        3,
    ]:
        raise ValueError("special token contract mismatch")
    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("vocabulary exceeds uint16")
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    shutil.copyfile(args.tokenizer, args.output / "tokenizer.json")
    files = []
    split_records = {}
    for split in ("train", "validation"):  # Test text is never tokenized for training.
        bylines = []
        block_bylines = []
        counts = []
        works = []
        path = args.output / (split + ".u16")
        with (
            gzip.open(
                args.corpus / (split + ".jsonl.gz"), "rt", encoding="utf-8"
            ) as rows,
            path.open("wb") as out,
        ):
            for line in rows:
                row = json.loads(line)
                if (
                    row["split"] != split
                    or hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]
                ):
                    raise ValueError("invalid text record")
                authors = row["authors"]
                if not authors:
                    raise ValueError("missing attribution")
                bylines.append(authors)
                ids = (
                    [1]
                    + tokenizer.encode(row["text"], add_special_tokens=False).ids
                    + [2]
                )
                if 3 in ids:
                    raise ValueError("unexpected UNK")
                for start in range(0, len(ids) - 1, args.sequence_length):
                    block = ids[start : start + args.sequence_length + 1]
                    counts.append(len(block) - 1)
                    block_bylines.append(authors)
                    works.append(row["work_id"])
                    block += [0] * (args.sequence_length + 1 - len(block))
                    out.write(np.asarray(block, dtype="<u2").tobytes())
        groups = components(bylines)
        lookup = {a: i for i, g in enumerate(groups) for a in g}
        group_ids = [lookup[x[0]] for x in block_bylines]
        totals = [0] * len(groups)
        for group, count in zip(group_ids, counts):
            totals[group] += count
        probabilities = (
            capped_probabilities([math.sqrt(n) for n in totals], args.author_cap)
            if split == "train"
            else [n / sum(totals) for n in totals]
        )
        metadata = {
            "groups": groups,
            "group_ids": group_ids,
            "target_counts": counts,
            "work_ids": works,
            "probabilities": probabilities,
            "target_tokens": sum(counts),
        }
        write_json(args.output / (split + ".json"), metadata)
        split_records[split] = {
            "blocks": len(counts),
            "targets": sum(counts),
            "author_groups": len(groups),
            "max_author_probability": max(probabilities),
        }
        for name in (split + ".u16", split + ".json"):
            p = args.output / name
            files.append(
                {"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            )
    receipt = {
        "kind": "obadh-literary-blocks",
        "version": 1,
        "corpus_id": manifest["dataset_id"],
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "sequence_length": args.sequence_length,
        "author_cap": args.author_cap,
        "splits": split_records,
        "files": files,
        "policy": __doc__,
    }
    receipt["data_id"] = digest_json(receipt)
    write_json(args.output / "manifest.json", receipt)
    (args.output / ".building").unlink()
    print(json.dumps(receipt, indent=2))


class LiteraryData:
    def __init__(self, root, *, tokenizer_sha256, sequence_length):
        import numpy as np

        self.np = np
        self.root = root
        if (root / ".building").exists():
            raise ValueError("incomplete literary data")
        receipt = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if receipt["kind"] != "obadh-literary-blocks" or receipt[
            "data_id"
        ] != digest_json({k: v for k, v in receipt.items() if k != "data_id"}):
            raise ValueError("invalid literary data identity")
        if (
            receipt["tokenizer_sha256"] != tokenizer_sha256
            or receipt["sequence_length"] != sequence_length
        ):
            raise ValueError("incompatible literary data")
        expected = {
            s + ext for s in ("train", "validation") for ext in (".u16", ".json")
        }
        if {r["path"] for r in receipt["files"]} != expected or len(
            receipt["files"]
        ) != 4:
            raise ValueError("invalid literary inventory")
        for item in receipt["files"]:
            p = root / item["path"]
            if p.stat().st_size != item["bytes"] or sha256_file(p) != item["sha256"]:
                raise ValueError("literary data integrity failure")
        self.receipt = receipt
        self.blocks = {}
        self.cdf = {}
        for split in ("train", "validation"):
            meta = json.loads((root / (split + ".json")).read_text(encoding="utf-8"))
            sizes = np.asarray(meta["target_counts"], dtype=np.float64)
            ids = np.asarray(meta["group_ids"])
            total = np.bincount(ids, weights=sizes, minlength=len(meta["groups"]))
            # Uniform blocks within a group, divided by its total targets:
            # sum(block_probability * valid_targets) is proportional to the
            # declared group weight even when books have different tail lengths.
            probabilities = np.asarray(meta["probabilities"])[ids] / total[ids]
            probabilities /= probabilities.sum()
            self.cdf[split] = np.cumsum(probabilities)
            self.cdf[split][-1] = 1.0
            self.blocks[split] = np.memmap(
                root / (split + ".u16"), mode="r", dtype="<u2"
            ).reshape(-1, sequence_length + 1)
            if len(self.blocks[split]) != len(sizes):
                raise ValueError("block count mismatch")

    def sample(self, split, count, generator=None):
        import torch

        values = torch.rand(count, generator=generator).numpy()
        indices = self.np.searchsorted(self.cdf[split], values, side="right")
        return self.np.asarray(self.blocks[split][indices], dtype=self.np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--author-cap", type=float, default=0.05)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
