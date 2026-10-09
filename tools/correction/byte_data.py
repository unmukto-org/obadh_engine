"""Versioned UTF-8 correction data for an offline ByT5 accuracy reference.

The external spelling corpus is synthetic, not human gold. Its row split is
replaced by a target-word split; benchmark vocabulary is excluded from external
training. Existing sentence partitions retain their original work split.
"""

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import unicodedata

from tools.corpus.provenance import digest_json, sha256_file, write_json

PREFIX = {0: "sentence: ", 1: "prefix: ", 2: "word: "}


def encode(text):
    return [b + 3 for b in text.encode("utf-8")] + [1]


def target_weights(source, target, edit_weight):
    """Weight complete edited Unicode characters, then expand to UTF-8 bytes.

    A deletion weights the following retained character (or EOS). Identity
    examples retain unit weights throughout, including their terminal mark.
    """
    if not 1 <= edit_weight <= 32:
        raise ValueError("edit weight outside supported range")
    weights = [1.] * (len(target) + 1)
    for kind, _, _, start, end in SequenceMatcher(a=source, b=target, autojunk=False).get_opcodes():
        if kind != "equal":
            for i in range(start, min(len(weights), max(end, start + 1))):
                weights[i] = float(edit_weight)
    return [w for c, w in zip(target, weights) for _ in c.encode("utf-8")] + [weights[-1]]


def decode(tokens):
    if tokens and tokens[0] == 0:
        tokens = tokens[1:]
    if 1 not in tokens:
        return None, "missing_eos"
    tokens = tokens[:tokens.index(1)]
    if not tokens or any(t < 3 or t > 258 for t in tokens):
        return None, "empty_or_control_token"
    try:
        text = bytes(t - 3 for t in tokens).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, "invalid_utf8"
    if not text.strip() or "\ufffd" in text or any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in text):
        return None, "invalid_text"
    return unicodedata.normalize("NFC", text), None


def fits(row, limit):
    return max(len(encode(PREFIX[row["mode"]] + row["source"])), len(encode(row["target"]))) <= limit


def word_partition(word):
    bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 100
    return "reserved" if bucket < 2 else "validation" if bucket < 4 else "train"


def lexical_reason(source, target, kind, valid_words):
    if kind in ("Run-on Error", "Homonym Error") or kind.startswith("Split-word Error"):
        return "unsafe_error_type"
    if not source or not target or any(not ("\u0980" <= c <= "\u09ff" and unicodedata.category(c)[0] in "LM") for c in source + target):
        return "non_word_or_boundary_edit"
    if source != target and source in valid_words:
        return "valid_word_collision"
    if source == target:
        return "unchanged_error"
    return None


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def parent_rows(path, partition, family):
    for row in read_rows(path / (partition + ".jsonl.gz")):
        if row["split"] != partition:
            raise ValueError("parent partition mismatch")
        yield dict(id=family + ":" + row["id"], source=row["source_text"], target=row["target"],
                   mode=row["mode"], kind=family + "/" + row["kind"], family=family)


def verify_parent(path):
    receipt = json.loads((path / "manifest.json").read_text())
    if (path / ".building").exists() or receipt["data_id"] != digest_json({k: v for k, v in receipt.items() if k != "data_id"}):
        raise ValueError("invalid parent manifest")
    for partition in ("train", "validation"):
        name = partition + ".jsonl.gz"
        entry = next(f for f in receipt["files"] if f["path"] == name)
        if sha256_file(path / name) != entry["sha256"]:
            raise ValueError("parent row digest mismatch")
    return receipt


def prepare(args):
    import regex
    base, teacher = verify_parent(args.base), verify_parent(args.teacher)
    if base["corpus_id"] != teacher["corpus_id"] or teacher["parent_data_id"] != base["data_id"]:
        raise ValueError("teacher lineage mismatch")
    upstream = json.loads((args.upstream / "upstream-receipt.json").read_text())
    csv_path = args.upstream / "upstream-data/train.csv"
    expected = next(f for f in upstream["files"] if f["path"] == "upstream-data/train.csv")
    if sha256_file(csv_path) != expected["sha256"]:
        raise ValueError("external corpus digest mismatch")
    bank = [json.loads(line) for line in args.bank.read_text().splitlines()]
    if any(set(r) != {"id", "kind", "source", "target", "mode"} for r in bank):
        raise ValueError("evaluation bank schema mismatch")
    validation = list(parent_rows(args.base, "validation", "base"))
    held_text = {r[k] for r in validation + bank for k in ("source", "target")}
    held_words = {w for text in held_text for w in regex.findall(r"[\p{L}\p{M}]+", text)}
    stats = Counter()
    candidates = {}
    valid_words = set()
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            valid_words.add(unicodedata.normalize("NFC", r["Word"]).strip())
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            s, t = (unicodedata.normalize("NFC", r[k]).strip() for k in ("Error", "Word"))
            reason = lexical_reason(s, t, r["ErrorType"], valid_words)
            if reason:
                stats[reason] += 1
                continue
            if t in held_words or s in held_words:
                stats["benchmark_vocabulary_excluded"] += 1
                continue
            partition = word_partition(t)
            if partition == "reserved":
                stats["reserved_unread_for_training"] += 1
                continue
            previous = candidates.get(s)
            row = dict(id=f"sec:{i}", source=s, target=t, mode=2, kind="lexical/" + r["ErrorType"], family="lexical", partition=partition)
            if previous is None and s not in candidates:
                candidates[s] = row
            elif previous and previous["target"] == t:
                stats["duplicate_pair"] += 1
            else:
                candidates[s] = None
                stats["ambiguous_source"] += 1
    lexical = [r for r in candidates.values() if r]
    del candidates, valid_words
    lexical_validation = [r for r in lexical if r["partition"] == "validation"]
    lexical_train = [r for r in lexical if r["partition"] == "train"]
    for partition, rows in (("train", lexical_train), ("validation", lexical_validation)):
        for word in sorted({r["target"] for r in rows}):
            rows.append(dict(id="sec:identity:" + hashlib.sha256(word.encode()).hexdigest(), source=word,
                             target=word, mode=2, kind="lexical/identity", family="lexical", partition=partition))
    held_text |= {r[k] for r in lexical_validation for k in ("source", "target")}
    train = list(parent_rows(args.base, "train", "base")) + list(parent_rows(args.teacher, "train", "teacher")) + lexical_train
    labels = {}
    for r in train:
        key = (r["mode"], r["source"])
        if key not in labels:
            labels[key] = r["target"]
        elif labels[key] != r["target"]:
            labels[key] = None
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / ".building").touch()
    counts = {}
    for partition, rows in (("train", train), ("validation", validation + lexical_validation)):
        count, seen = Counter(), set()
        with gzip.open(args.output / (partition + ".jsonl.gz"), "wt", encoding="utf-8") as f:
            for r in rows:
                if not fits(r, args.limit):
                    stats[partition + "/capacity_excluded"] += 1
                    continue
                if partition == "train" and (r["source"] in held_text or r["target"] in held_text or labels[(r["mode"], r["source"])] is None):
                    stats["train/heldout_or_conflicting"] += 1
                    continue
                key = (r["source"], r["target"], r["mode"], r["family"])
                if key in seen:
                    stats[partition + "/duplicate"] += 1
                    continue
                seen.add(key)
                f.write(json.dumps({k: v for k, v in r.items() if k != "partition"}, ensure_ascii=False) + "\n")
                count[r["kind"]] += 1
        counts[partition] = dict(count)
    manifest = dict(version=1, kind="obadh-byte-correction-data", base_data_id=base["data_id"], teacher_data_id=teacher["data_id"],
                    upstream=upstream, bank_sha256=digest_json(bank), byte_limit=args.limit, prefixes=PREFIX,
                    counts=counts, exclusions=dict(stats), builder_sha256=sha256_file(Path(__file__)),
                    scope=__doc__, files=[dict(path=p.name, sha256=sha256_file(p)) for p in sorted(args.output.glob("*.gz"))])
    manifest["data_id"] = digest_json(manifest)
    write_json(args.output / "manifest.json", manifest)
    (args.output / ".building").unlink()
    print(json.dumps(dict(counts=counts, exclusions=dict(stats), data_id=manifest["data_id"]), indent=2), flush=True)


class ByteData:
    def __init__(self, root):
        self.receipt = json.loads((root / "manifest.json").read_text())
        if (root / ".building").exists() or self.receipt["data_id"] != digest_json({k: v for k, v in self.receipt.items() if k != "data_id"}):
            raise ValueError("byte data identity mismatch")
        self.rows, self.pools = {}, {}
        for file in self.receipt["files"]:
            path = root / file["path"]
            if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != file["sha256"]:
                raise ValueError("byte data file mismatch")
            split = path.name.split(".")[0]
            self.rows[split] = rows = list(read_rows(path))
            self.pools[split] = pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for i, row in enumerate(rows):
                pools[row["family"]][row["source"] == row["target"]][row["kind"]].append(i)
            if any(not pools[f][identity] for f in ("base", "lexical") for identity in (False, True)):
                raise ValueError("missing sampling pool")

    def sample(self, rng, batch, identity_fraction, split="train", family_weights=None):
        result = []
        pools = self.pools[split]
        families, weights = (("base", "lexical", "teacher"), (.7, .2, .1)) if split == "train" else (("base", "lexical"), (.8, .2))
        if split == "train" and family_weights is not None:
            families, weights = tuple(family_weights), tuple(family_weights.values())
            if not families or any(w <= 0 for w in weights) or any(f not in pools or not all(pools[f].get(i) for i in (False, True)) for f in families):
                raise ValueError("invalid family weights or missing correction/identity pool")
        for _ in range(batch):
            family = rng.choices(families, weights)[0]
            clean = rng.random() < identity_fraction
            kinds = pools[family][clean]
            kind = rng.choice(sorted(kinds))
            result.append(self.rows[split][rng.choice(kinds[kind])])
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "teacher", "upstream", "bank", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--limit", type=int, default=768)
    args = p.parse_args()
    if not 32 <= args.limit <= 2048:
        p.error("byte limit outside supported range")
    prepare(args)


if __name__ == "__main__":
    main()
