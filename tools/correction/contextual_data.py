"""Put spelling edits inside existing, partition-preserving sentence contexts.

Synthetic restoration and preservation controls remain development supervision,
not human-adjudicated grammar gold. Known benchmark word-error pairs are excluded;
ordinary vocabulary may occur in different training contexts. Lexical validation
words retain their reserved partition.
"""

import argparse
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import gzip
import json
from pathlib import Path
import random
import shutil
import unicodedata

import regex

from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.byte_data import fits, lexical_reason, read_rows, word_partition
from tools.correction.data import corruptions


def words(text):
    return list(regex.finditer(r"[\p{L}\p{M}\p{N}\u200c\u200d]+", text))


def changed_word_pairs(source, target):
    a, b = [w.group() for w in words(source)], [w.group() for w in words(target)]
    return {(a[i], b[k]) for tag, i, j, k, l in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
            if tag == "replace" and j - i == l - k == 1}


def replace_word(text, span, wrong):
    return text[:span.start()] + wrong + text[span.end():]


def verify(root):
    m = json.loads((root / "manifest.json").read_text())
    if (root / ".building").exists() or m["data_id"] != digest_json({k: v for k, v in m.items() if k != "data_id"}):
        raise ValueError("invalid parent data")
    for entry in m["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != entry["sha256"]:
            raise ValueError("parent file mismatch")
    return m


def prepare(args):
    parent = verify(args.parent)
    bank = [json.loads(line) for line in args.bank.read_text().splitlines()]
    validation = list(read_rows(args.parent / "validation.jsonl.gz"))
    excluded_texts = {r[k] for r in bank + validation for k in ("source", "target")}
    excluded_pairs = set().union(*(changed_word_pairs(r["source"], r["target"]) for r in bank + validation))
    held_words = {r["target"] for r in validation if r["family"] == "lexical"}
    upstream = next(f for f in parent["upstream"]["files"] if f["path"] == "upstream-data/train.csv")
    if sha256_file(args.lexical_csv) != upstream["sha256"]:
        raise ValueError("external lexical data mismatch")
    clean = [r for r in read_rows(args.parent / "train.jsonl.gz") if r["family"] == "base" and r["source"] == r["target"]]
    valid_words = {w.group() for r in clean for w in words(r["target"])}
    with args.lexical_csv.open(encoding="utf-8-sig", newline="") as f:
        valid_words |= {unicodedata.normalize("NFC", r["Word"]).strip() for r in csv.DictReader(f)}
    by_word = defaultdict(list)
    counts = Counter()
    with args.lexical_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            wrong, correct = (unicodedata.normalize("NFC", r[k]).strip() for k in ("Error", "Word"))
            reason = lexical_reason(wrong, correct, r["ErrorType"], valid_words)
            if reason or correct in held_words or word_partition(correct) != "train" or (wrong, correct) in excluded_pairs:
                counts[reason or "held_word_or_error_pair"] += 1
                continue
            item = ("external/" + r["ErrorType"], wrong)
            if item not in by_word[correct] and len(by_word[correct]) < 16:
                by_word[correct].append(item)
    # Generate additional error forms from training vocabulary, without using
    # benchmark references as examples. This also covers common modern words
    # absent from the external isolated-word inventory.
    eligible_words = sorted({w.group() for r in clean for w in words(r["target"])
                             if w.group() not in held_words and word_partition(w.group()) == "train"})
    for index, word in enumerate(eligible_words):
        for kind, wrong in corruptions(word, index + args.seed):
            if kind in ("punctuation", "agreement_candidate", "missing_space"):
                continue
            if lexical_reason(wrong, word, kind, valid_words) or (wrong, word) in excluded_pairs:
                continue
            item = ("generated/" + kind, wrong)
            if item not in by_word[word]:
                by_word[word].append(item)
    rng = random.Random(args.seed)
    rng.shuffle(clean)
    augmented, word_use, seen = [], Counter(), set()
    labels = {}

    def admit(row):
        if row["source"] in excluded_texts or row["target"] in excluded_texts or not fits(row, parent["byte_limit"]):
            counts["augmented_heldout_or_capacity"] += 1
            return
        key = (row["source"], row["target"], row["mode"], row["kind"])
        if key in seen:
            return
        seen.add(key)
        row["id"] = "context:" + digest_json(key)
        label_key = (row["source"], row["mode"])
        if label_key in labels and labels[label_key] != row["target"]:
            labels[label_key] = None
        else:
            labels[label_key] = row["target"]
        augmented.append(row)

    for original in clean:
        text = original["target"]
        spans = [w for w in words(text) if w.group() in by_word and word_use[w.group()] < args.per_word]
        rng.shuffle(spans)
        made = 0
        for span in spans:
            if made >= args.per_context:
                break
            options = by_word[span.group()]
            kind, wrong = rng.choice(options)
            source = replace_word(text, span, wrong)
            admit(dict(source=source, target=text, mode=original["mode"], kind="context/" + kind,
                       family="context", parent_id=original["id"], lexical_pair=[wrong, span.group()]))
            word_use[span.group()] += 1
            made += 1
        if made:
            admit(dict(source=text, target=text, mode=original["mode"], kind="context/identity", family="context", parent_id=original["id"]))
        if original["mode"] == 0 and text.endswith("।"):
            if any(regex.fullmatch(r"[\p{L}\p{M}]{2,}(?:ব|বো|বে|বেন)", w.group()) for w in words(text)):
                admit(dict(source=text, target=text, mode=0, kind="base/preserve_future", family="base", parent_id=original["id"]))
            if int(digest_json(text)[:8], 16) % 20 == 0:
                alternate = text[:-1] + "…"
                admit(dict(source=alternate, target=alternate, mode=0, kind="base/preserve_ellipsis", family="base", parent_id=original["id"]))
    # Authored invariance controls: text is intentionally preserved verbatim.
    # These are not examples of model-approved grammar or real user messages.
    templates = ["আজ {item} খুলছে না।", "আমি {item} পাঠিয়েছি।", "ওই {item} আবার দেখব।", "এখানে {item} লেখা আছে।"]
    for i in range(2048):
        item = rng.choice(["GitHub", "Chrome", "Excel", "Telegram", "Slack", "API", "URL", "DNS", "Signal"]) + "-" + str(rng.randrange(1000, 99999))
        text = rng.choice(templates).format(item=item)
        admit(dict(source=text, target=text, mode=i % 2, kind="base/preserve_mixed", family="base", supervision="authored exact-copy control"))
    for verb in ("যামু", "আসমু", "করমু", "দেখমু", "খামু", "থাকমু"):
        for time in ("কাল", "পরশু", "পরে", "আগামীকাল", "একটু পরে"):
            for prefix in ("আমি", "আমরা", "আমি কিন্তু", "তাহলে আমি"):
                for ending in ("।", " না।", "…"):
                    text = f"{prefix} {time} {verb}{ending}"
                    admit(dict(source=text, target=text, mode=0, kind="base/preserve_dialect", family="base", supervision="authored dialect-preservation control"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / ".building").touch()
    train_counts = Counter()
    with gzip.open(args.output / "train.jsonl.gz", "wt", encoding="utf-8") as out:
        for row in read_rows(args.parent / "train.jsonl.gz"):
            key = (row["source"], row["mode"])
            if key in labels and labels[key] != row["target"]:
                labels[key] = None
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            train_counts[row["kind"]] += 1
        for row in augmented:
            if labels[(row["source"], row["mode"])] is None:
                counts["augmented_ambiguous_input"] += 1
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            train_counts[row["kind"]] += 1
    shutil.copyfile(args.parent / "validation.jsonl.gz", args.output / "validation.jsonl.gz")
    manifest = dict(version=1, kind="obadh-byte-correction-data", parent_data_id=parent["data_id"], byte_limit=parent["byte_limit"],
                    prefixes=parent["prefixes"], counts=dict(train=dict(train_counts), validation=parent["counts"]["validation"]),
                    bank_sha256=digest_json(bank), lexical_csv_sha256=upstream["sha256"],
                    held_error_pairs=len(excluded_pairs), held_lexical_words=len(held_words), exclusions=dict(counts),
                    per_word=args.per_word, per_context=args.per_context, seed=args.seed, builder_sha256=sha256_file(Path(__file__)),
                    scope=__doc__, files=[dict(path=p.name, sha256=sha256_file(p)) for p in sorted(args.output.glob("*.gz"))])
    manifest["data_id"] = digest_json(manifest)
    write_json(args.output / "manifest.json", manifest)
    (args.output / ".building").unlink()
    print(json.dumps(dict(data_id=manifest["data_id"], train_rows=sum(train_counts.values()), added=sum(train_counts.values()) - sum(parent["counts"]["train"].values()), kinds=dict(train_counts), exclusions=dict(counts)), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "lexical-csv", "bank", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--per-word", type=int, default=64)
    p.add_argument("--per-context", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261006)
    args = p.parse_args()
    if min(args.per_word, args.per_context) < 1:
        p.error("positive augmentation budgets required")
    prepare(args)


if __name__ == "__main__":
    main()
