"""Prepare and sample provenance-checked, document-bounded language-model data.

The tokenizer learns from training text only. News/wiki/book token exposure is
explicitly balanced; connected literary contributors retain a bounded share of
the book domain. Padding is excluded from the loss. Test text is never tokenized.
"""

from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from tools.autosuggest.literary_data import components, capped_probabilities
from tools.autosuggest.subword_tokenizer import make_tokenizer
from tools.corpus.provenance import digest_json, sha256_file, write_json

DOMAINS = ("news", "wiki", "books")


def domain(row):
    return row["source"] if row["source"] in DOMAINS[:2] else "books"


def records(corpus, split):
    with gzip.open(corpus / (split + ".jsonl.gz"), "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row["split"] != split
                or hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                != row["sha256"]
            ):
                raise ValueError("text identity or split mismatch")
            if domain(row) == "books" and not row["authors"]:
                raise ValueError("missing literary authors")
            yield row


def prepare(args):
    import numpy as np
    from tokenizers import pre_tokenizers, trainers

    if args.output.exists() or (args.corpus / ".building").exists():
        raise ValueError("existing output or incomplete input corpus")
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] != "obadh-contextual-text" or manifest[
        "dataset_id"
    ] != digest_json({k: v for k, v in manifest.items() if k != "dataset_id"}):
        raise ValueError("invalid input manifest")
    if {f["path"] for f in manifest["files"]} != {
        s + ".jsonl.gz" for s in ("train", "validation", "test")
    }:
        raise ValueError("invalid partition inventory")
    for item in manifest["files"]:
        path = args.corpus / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError("input file integrity failure")
    weights = dict(zip(DOMAINS, args.domain_weights))
    if any(
        w <= 0 or not math.isfinite(w) for w in weights.values()
    ) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("three positive domain weights must sum to one")
    if (
        not 260 <= args.vocab <= 65536
        or not 1 <= args.sequence_length <= 65535
        or args.tokenizer_characters_per_domain < 1
    ):
        raise ValueError("invalid tokenizer/block dimensions")
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    tokenizer = make_tokenizer("unicode-marks-v1")
    tokenizer_counts = Counter()
    bylines = set()

    def tokenizer_texts():
        for row in records(args.corpus, "train"):
            d = domain(row)
            if d == "books":
                bylines.add(tuple(row["authors"]))
            if tokenizer_counts[d] < args.tokenizer_characters_per_domain:
                tokenizer_counts[d] += len(row["text"])
                yield row["text"]

    tokenizer.train_from_iterator(
        tokenizer_texts(),
        trainer=trainers.BpeTrainer(
            vocab_size=args.vocab,
            min_frequency=2,
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer.save(str(args.output / "tokenizer.json"))
    # Added-token matching otherwise turns literal user text such as "[PAD]"
    # into a training control ID, even with add_special_tokens=False. This flag
    # is runtime state and is not preserved by tokenizer.json serialization.
    tokenizer.encode_special_tokens = True
    literary_groups = components(sorted(bylines))
    literary_lookup = {
        a: i + 2 for i, group in enumerate(literary_groups) for a in group
    }
    print(
        json.dumps(
            {
                "event": "tokenizer_ready",
                "vocab": tokenizer.get_vocab_size(),
                "characters": dict(tokenizer_counts),
            }
        ),
        flush=True,
    )
    partitions = {}
    files = []
    for split in ("train", "validation"):
        groups = [dict(domain="news", authors=[]), dict(domain="wiki", authors=[])]
        groups += (
            [dict(domain="books", authors=g) for g in literary_groups]
            if split == "train"
            else [dict(domain="books", authors=[])]
        )
        totals = np.zeros(len(groups), dtype=np.int64)
        block_count = 0
        window_count = 0
        with (
            (args.output / (split + ".u16")).open("wb") as blocks,
            (args.output / (split + ".groups.u16")).open("wb") as group_file,
            (args.output / (split + ".targets.u16")).open("wb") as count_file,
            (args.output / (split + ".index.jsonl.gz")).open("wb") as index_raw,
            gzip.GzipFile(filename="", fileobj=index_raw, mode="wb", mtime=0) as index,
        ):
            buffer = []

            def flush():
                nonlocal block_count, window_count
                encoded = tokenizer.encode_batch(
                    [r["text"] for r in buffer], add_special_tokens=False
                )
                for row, encoding in zip(buffer, encoded):
                    d = domain(row)
                    group = (
                        DOMAINS.index(d)
                        if d != "books" or split != "train"
                        else literary_lookup[row["authors"][0]]
                    )
                    ids = [1] + encoding.ids + [2]
                    if 3 in ids:
                        raise ValueError("lossless byte tokenizer produced UNK")
                    first = block_count
                    for start in range(0, len(ids) - 1, args.sequence_length):
                        block = ids[start : start + args.sequence_length + 1]
                        targets = len(block) - 1
                        totals[group] += targets
                        blocks.write(
                            np.asarray(
                                block + [0] * (args.sequence_length + 1 - len(block)),
                                dtype="<u2",
                            ).tobytes()
                        )
                        group_file.write(np.asarray([group], dtype="<u2").tobytes())
                        count_file.write(np.asarray([targets], dtype="<u2").tobytes())
                        block_count += 1
                    item = {
                        k: row[k] for k in ("source", "work_id", "window", "sha256")
                    }
                    item.update(
                        first_block=first, block_count=block_count - first, group=group
                    )
                    index.write(
                        (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
                    )
                    window_count += 1
                buffer.clear()

            for row in records(args.corpus, split):
                buffer.append(row)
                if len(buffer) >= 512:
                    flush()
                    if window_count % 5120 == 0:
                        print(
                            json.dumps(
                                dict(
                                    event="encoding",
                                    split=split,
                                    windows=window_count,
                                    blocks=block_count,
                                )
                            ),
                            flush=True,
                        )
            if buffer:
                flush()
        if any(n <= 0 for n in totals):
            raise ValueError("missing domain or literary group")
        probabilities = [weights["news"], weights["wiki"]]
        probabilities += (
            [
                weights["books"] * p
                for p in capped_probabilities(
                    [math.sqrt(n) for n in totals[2:]], args.author_cap
                )
            ]
            if split == "train"
            else [weights["books"]]
        )
        metadata = dict(
            groups=groups,
            target_tokens=totals.tolist(),
            group_target_probabilities=probabilities,
            blocks=block_count,
            windows=window_count,
        )
        write_json(args.output / (split + ".json"), metadata)
        for suffix in (
            ".u16",
            ".groups.u16",
            ".targets.u16",
            ".index.jsonl.gz",
            ".json",
        ):
            path = args.output / (split + suffix)
            files.append(
                dict(
                    path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path)
                )
            )
        partitions[split] = dict(
            tokens=int(totals.sum()),
            blocks=block_count,
            windows=window_count,
            sha256=sha256_file(args.output / (split + ".u16")),
        )
    receipt = dict(
        version=1,
        kind="obadh-document-training-data",
        dataset_id=manifest["dataset_id"],
        vocab_size=tokenizer.get_vocab_size(),
        tokenizer_sha256=sha256_file(args.output / "tokenizer.json"),
        sequence_length=args.sequence_length,
        domain_weights=weights,
        literary_author_cap=args.author_cap,
        tokenizer_train_characters=dict(tokenizer_counts),
        literal_control_strings=True,
        tokenizer_character_budget_per_domain=args.tokenizer_characters_per_domain,
        tokenizer_sampling="first whole training windows in canonical hashed-work order per domain up to character budget; final window may exceed budget",
        partitions=partitions,
        files=files,
        policy=__doc__,
        builder_sha256=sha256_file(Path(__file__)),
        helper_sha256={
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("literary_data.py", "subword_tokenizer.py")
        },
    )
    receipt["data_id"] = digest_json(receipt)
    write_json(args.output / "manifest.json", receipt)
    (args.output / ".building").unlink()
    print(json.dumps(receipt, indent=2), flush=True)


class DocumentData:
    def __init__(self, root, *, tokenizer_sha256, sequence_length):
        import numpy as np

        self.np = np
        if (root / ".building").exists():
            raise ValueError("incomplete document data")
        receipt = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if receipt["kind"] != "obadh-document-training-data" or receipt[
            "data_id"
        ] != digest_json({k: v for k, v in receipt.items() if k != "data_id"}):
            raise ValueError("invalid document data identity")
        if (
            receipt["tokenizer_sha256"] != tokenizer_sha256
            or receipt["sequence_length"] != sequence_length
            or sha256_file(root / "tokenizer.json") != tokenizer_sha256
        ):
            raise ValueError("incompatible document data")
        expected = {
            s + ext
            for s in ("train", "validation")
            for ext in (
                ".u16",
                ".groups.u16",
                ".targets.u16",
                ".index.jsonl.gz",
                ".json",
            )
        }
        if (
            len(receipt["files"]) != len(expected)
            or {f["path"] for f in receipt["files"]} != expected
        ):
            raise ValueError("invalid block inventory")
        for item in receipt["files"]:
            path = root / item["path"]
            if (
                path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                raise ValueError("block integrity failure")
        self.receipt = receipt
        self.blocks, self.cdf = {}, {}
        for split in ("train", "validation"):
            metadata = json.loads(
                (root / (split + ".json")).read_text(encoding="utf-8")
            )
            groups = np.memmap(root / (split + ".groups.u16"), dtype="<u2", mode="r")
            counts = np.memmap(root / (split + ".targets.u16"), dtype="<u2", mode="r")
            if (
                len(groups) != metadata["blocks"]
                or len(counts) != len(groups)
                or np.any(counts == 0)
                or np.any(counts > sequence_length)
                or np.any(groups >= len(metadata["groups"]))
            ):
                raise ValueError("invalid block metadata")
            totals = np.bincount(
                groups, weights=counts, minlength=len(metadata["groups"])
            )
            probabilities = np.asarray(
                metadata["group_target_probabilities"], dtype=np.float64
            )
            if (
                not np.array_equal(totals, metadata["target_tokens"])
                or np.any(totals <= 0)
                or not np.all(np.isfinite(probabilities))
                or np.any(probabilities <= 0)
                or not np.isclose(probabilities.sum(), 1)
            ):
                raise ValueError("invalid group exposure")
            block_probabilities = probabilities[groups] / totals[groups]
            block_probabilities /= block_probabilities.sum()
            self.cdf[split] = np.cumsum(block_probabilities)
            self.cdf[split][-1] = 1.0
            self.blocks[split] = np.memmap(
                root / (split + ".u16"), dtype="<u2", mode="r"
            ).reshape(-1, sequence_length + 1)
            if len(self.blocks[split]) != len(groups):
                raise ValueError("block dimensions disagree")

    def sample(self, split, count, generator=None):
        import torch

        indices = self.np.searchsorted(
            self.cdf[split],
            torch.rand(count, generator=generator).numpy(),
            side="right",
        )
        return self.np.asarray(self.blocks[split][indices], dtype=self.np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--tokenizer-characters-per-domain", type=int, default=32000000)
    p.add_argument("--domain-weights", type=float, nargs=3, default=[0.5, 0.25, 0.25])
    p.add_argument("--author-cap", type=float, default=0.05)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
