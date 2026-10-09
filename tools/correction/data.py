"""Construct split-preserving correction pairs with auditable error operators.

Original corpus text is a weak reference, not human-certified grammar gold.
Synthetic errors define a reversal task. Gemma validation is a separate stage.
No test partition or correction benchmark is admitted for training.
"""

from __future__ import annotations
import argparse
from collections import Counter
from difflib import SequenceMatcher
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
import numpy as np

from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.autosuggest.document_data import records, domain

CONFUSIONS = {"ি": "ী", "ী": "ি", "ু": "ূ", "ূ": "ু", "শ": "স", "ষ": "স", "স": "শ", "ণ": "ন", "ন": "ণ"}
VERBS = [
    ("যাই", "যাও", "যায়", "যান"), ("করি", "করো", "করে", "করেন"),
    ("বলি", "বলো", "বলে", "বলেন"), ("দেখি", "দেখো", "দেখে", "দেখেন"),
    ("খাই", "খাও", "খায়", "খান"), ("চাই", "চাও", "চায়", "চান"),
    ("পারি", "পারো", "পারে", "পারেন"), ("আছি", "আছ", "আছে", "আছেন"),
]
SUBJECTS = {"আমি": 0, "আমরা": 0, "তুমি": 1, "তোমরা": 1, "সে": 2, "তারা": 2, "আপনি": 3, "তিনি": 3, "আপনারা": 3, "তাঁরা": 3}


def spans(text):
    import regex
    return list(regex.finditer(r"[\p{L}\p{M}\p{N}\u200c\u200d]+", text))


def sentences(text):
    # Retain terminal marks and closing quotes. Periods are kept inside the
    # span to avoid splitting abbreviations, decimals, domains and initials.
    for line in text.splitlines():
        for match in re.finditer(r"[^।!?]+[।!?]+[\"'”’»)]*", line):
            value = match.group().strip()
            if value:
                yield value


def corruptions(text, seed):
    rng = random.Random(seed)
    words = spans(text)
    result = []
    replaceable = [i for i, c in enumerate(text) if c in CONFUSIONS]
    if replaceable:
        i = rng.choice(replaceable)
        result.append(("phonetic_spelling", text[:i] + CONFUSIONS[text[i]] + text[i + 1:]))
    # Deleting/repeating a complete Bangla grapheme preserves UTF-8 and avoids
    # training accidental Latin/name/number transformations as spelling repair.
    import regex
    eligible = [w for w in words if all("\u0980" <= c <= "\u09ff" or c in "\u200c\u200d" for c in w.group()) and len(regex.findall(r"\X", w.group())) >= 3 and not any(c.isdigit() for c in w.group())]
    if eligible:
        word = rng.choice(eligible)
        clusters = list(regex.finditer(r"\X", word.group()))
        cluster = rng.choice(clusters[1:-1])
        start, end = word.start() + cluster.start(), word.start() + cluster.end()
        result += [("grapheme_deletion", text[:start] + text[end:]), ("grapheme_repeat", text[:end] + text[start:end] + text[end:])]
    gaps = [(a.end(), b.start()) for a, b in zip(words, words[1:]) if text[a.end():b.start()] == " " and all(any("\u0980" <= c <= "\u09ff" for c in w.group()) for w in (a, b))]
    if gaps:
        start, end = rng.choice(gaps)
        result.append(("missing_space", text[:start] + text[end:]))
    question = any(w.group() in {"কোথায়", "কেন", "কখন", "কী", "কেমন", "কত", "কাকে"} for w in words)
    if "?" in text and question:
        i = text.rfind("?")
        result.append(("punctuation", text[:i] + "।" + text[i + 1:]))
    elif text.endswith("।"):
        # This is explicitly a completed-sentence restoration task. Valid
        # unfinished typing prefixes are separately labeled as unchanged.
        result.append(("punctuation", text[:-1]))
    if words and words[0].group() in SUBJECTS:
        person = SUBJECTS[words[0].group()]
        for word in words[1:9]:
            if word.group() in SUBJECTS:
                break
            alternatives = next((v for v in VERBS if word.group() == v[person]), None)
            if alternatives:
                wrong = rng.choice([v for j, v in enumerate(alternatives) if j != person])
                result.append(("agreement_candidate", text[:word.start()] + wrong + text[word.end():]))
                break
    # A whole-grapheme deletion does not simulate omitted vowel signs or
    # hasant: ভালোবাসি -> ভালোবসি and নিশ্চিন্ত -> নিশচিন্ত need these
    # distinct operators. Use a separate RNG to preserve older error draws.
    extra = random.Random(seed ^ 0xC0FFEE)
    marks = [i for i, c in enumerate(text) if c in "ািীুূৃেৈোৌ্ঁ"]
    if marks:
        i = extra.choice(marks)
        result.append(("mark_deletion", text[:i] + text[i + 1:]))
    conjuncts = list(regex.finditer(r"্[ক-হড়ঢ়য়]়?", text))
    if conjuncts:
        match = extra.choice(conjuncts)
        result.append(("conjunct_deletion", text[:match.start()] + text[match.end():]))
    sound_changes = [(a, b, m.start()) for a, b in (("ন্য", "ন্ন"), ("জ্ঞ", "গ্গ"), ("ক্ষ", "খ"), ("দ্ধ", "দ্দ")) for m in regex.finditer(regex.escape(a), text)]
    if sound_changes:
        a, b, i = extra.choice(sound_changes)
        result.append(("conjunct_spelling", text[:i] + b + text[i + len(a):]))
    return [(kind, unicodedata.normalize("NFC", source)) for kind, source in result if source != text]


def edited_target_weights(source, target):
    weights = [1.0] * len(target)
    for tag, _, _, begin, end in SequenceMatcher(a=source, b=target, autojunk=False).get_opcodes():
        if tag != "equal":
            # A deletion supervises the next retained token, including EOS.
            for index in range(begin, min(len(target), max(end, begin + 1))):
                weights[index] = 4.0
    return weights


def prepare(args):
    from tokenizers import Tokenizer
    if args.output.exists() or (args.corpus / ".building").exists():
        raise ValueError("existing output or incomplete corpus")
    corpus = json.loads((args.corpus / "manifest.json").read_text())
    language = json.loads((args.language_data / "manifest.json").read_text())
    if corpus["dataset_id"] != digest_json({k: v for k, v in corpus.items() if k != "dataset_id"}) or language["dataset_id"] != corpus["dataset_id"]:
        raise ValueError("corpus identity mismatch")
    if language["tokenizer_sha256"] != sha256_file(args.language_data / "tokenizer.json"):
        raise ValueError("tokenizer mismatch")
    group_file = next(f for f in language["files"] if f["path"] == "train.json")
    if sha256_file(args.language_data / "train.json") != group_file["sha256"]:
        raise ValueError("contributor inventory mismatch")
    groups = json.loads((args.language_data / "train.json").read_text())["groups"]
    author_group = {author: str(i) for i, group in enumerate(groups) for author in group["authors"]}
    if not 8 <= args.length <= 512 or min(args.train_sentences, args.validation_sentences) < 3:
        raise ValueError("invalid data limits")
    tokenizer = Tokenizer.from_file(str(args.language_data / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    import shutil
    shutil.copy2(args.language_data / "tokenizer.json", args.output / "tokenizer.json")
    held_texts = set()
    summary = {}
    files = []
    for split, limit in (("validation", args.validation_sentences), ("train", args.train_sentences)):
        entry = next(f for f in corpus["files"] if f["path"] == split + ".jsonl.gz")
        if sha256_file(args.corpus / entry["path"]) != entry["sha256"]:
            raise ValueError("partition hash mismatch")
        budgets = {"books": limit // 2, "news": limit // 3, "wiki": limit - limit // 2 - limit // 3}
        admitted = Counter()
        per_work, per_author = Counter(), Counter()
        seen = set()
        selected = []
        excluded = Counter()
        author_cap = max(1, min(int(budgets["books"] * .05), (budgets["books"] + len(set(author_group.values())) - 1) // len(set(author_group.values())))) if split == "train" else limit
        for row in records(args.corpus, split):
            d = domain(row)
            if admitted[d] >= budgets[d]:
                continue
            group = author_group[row["authors"][0]] if d == "books" and split == "train" else d
            if d == "books" and per_author[group] >= author_cap:
                continue
            for text in sentences(row["text"]):
                if admitted[d] >= budgets[d] or per_work[row["work_id"]] >= (max(600, limit // 50) if d == "books" else 4) or (d == "books" and per_author[group] >= author_cap):
                    break
                key = hashlib.sha256(text.encode()).hexdigest()
                if key in seen or (split == "train" and key in held_texts):
                    excluded["duplicate_or_validation_text"] += 1
                    continue
                words = spans(text)
                if not 2 <= len(words) <= 40 or "\ufffd" in text or not any("\u0980" <= c <= "\u09ff" and c.isalpha() for c in text):
                    excluded["text_quality"] += 1
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not 2 <= len(ids) <= args.length - 10 or any(i < 4 for i in ids):
                    excluded["length_or_control"] += 1
                    continue
                seen.add(key)
                admitted[d] += 1
                per_work[row["work_id"]] += 1
                per_author[group] += 1
                selected.append(dict(text=text, clean_sha256=key, domain=d, group=group, source=row["source"], work_id=row["work_id"], window=row["window"], authors=row["authors"], reference=row.get("reference")))
            if all(admitted[d] >= n for d, n in budgets.items()):
                break
        if split == "validation":
            held_texts.update(seen)
        if any(admitted[d] == 0 for d in budgets):
            raise ValueError("missing source domain")
        rows = []
        counts = Counter()
        for clean in selected:
            target = clean["text"]
            seed = int(clean["clean_sha256"][:16], 16)
            variants = corruptions(target, seed)
            # Preserve all available error categories; identities comprise an
            # explicitly sampled group in the trainer rather than dominating
            # or disappearing according to how many operators match a sentence.
            candidates = [("identity", target, target, 0)] + [(k, s, target, 0) for k, s in variants]
            words = spans(target)
            if len(words) >= 4:
                cut = random.Random(seed).randint(2, len(words) - 1)
                prefix = target[:words[cut - 1].end()]
                candidates.append(("prefix_identity", prefix, prefix, 1))
                # Spellcheck must also work before the sentence is finished.
                # Never teach a prefix to grow words or terminal punctuation.
                candidates.extend(("prefix/" + k, s, prefix, 1) for k, s in corruptions(prefix, seed)
                                  if k not in ("punctuation", "agreement_candidate"))
            for kind, source, gold, mode in candidates:
                source_ids = [1] + tokenizer.encode(source, add_special_tokens=False).ids + [2]
                target_ids = tokenizer.encode(gold, add_special_tokens=False).ids + [2]
                if max(len(source_ids), len(target_ids)) > args.length:
                    excluded["corrupted_length"] += 1
                    continue
                record = {k: v for k, v in clean.items() if k != "text"}
                record.update(source_text=source, target=gold, kind=kind, mode=mode, split=split, supervision="source-derived synthetic; not teacher-validated or human-certified")
                record["id"] = digest_json([clean["clean_sha256"], kind, source, gold, mode])
                rows.append((record, source_ids, target_ids, edited_target_weights(source_ids[1:], target_ids)))
                counts[kind] += 1
        # Mix stable record order to avoid a single author/domain at an epoch's start.
        rows.sort(key=lambda row: row[0]["id"])
        n = len(rows)
        source = np.zeros((n, args.length), dtype="<u2")
        target = np.zeros_like(source)
        weight = np.zeros((n, args.length), dtype="<f4")
        mode = np.zeros(n, dtype="u1")
        identity = np.zeros(n, dtype="u1")
        with gzip.open(args.output / (split + ".jsonl.gz"), "wt", encoding="utf-8") as handle:
            for i, (record, s, t, w) in enumerate(rows):
                source[i, :len(s)], target[i, :len(t)], weight[i, :len(w)] = s, t, w
                mode[i], identity[i] = record["mode"], int(record["source_text"] == record["target"])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        for name, values in (("source", source), ("target", target), ("weight", weight), ("mode", mode), ("identity", identity)):
            path = args.output / f"{split}.{name}.npy"
            np.save(path, values, allow_pickle=False)
        for path in sorted(args.output.glob(split + ".*")):
            files.append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path)))
        summary[split] = dict(rows=n, clean_sentences=len(selected), source_sentences=dict(admitted), kinds=dict(counts), exclusions=dict(excluded), contributor_clean_sentence_counts=dict(per_author), contributor_cap=author_cap)
        print(json.dumps(dict(event="correction_pairs", split=split, **summary[split])), flush=True)
    manifest = dict(kind="obadh-correction-pairs", version=1, corpus_id=corpus["dataset_id"], tokenizer_sha256=language["tokenizer_sha256"], vocab_size=language["vocab_size"], sequence_length=args.length, partitions=summary, files=files, builder_sha256=sha256_file(Path(__file__)), supervision=__doc__, modes={"sentence": 0, "valid_prefix": 1})
    manifest["data_id"] = digest_json(manifest)
    write_json(args.output / "manifest.json", manifest)
    (args.output / ".building").unlink()


class PairData:
    def __init__(self, root):
        self.root = Path(root)
        if (self.root / ".building").exists():
            raise ValueError("incomplete correction data")
        self.receipt = json.loads((self.root / "manifest.json").read_text())
        r = self.receipt
        if r["kind"] != "obadh-correction-pairs" or r["data_id"] != digest_json({k: v for k, v in r.items() if k != "data_id"}) or sha256_file(self.root / "tokenizer.json") != r["tokenizer_sha256"]:
            raise ValueError("correction data identity mismatch")
        for item in r["files"]:
            p = self.root / item["path"]
            if not p.resolve().is_relative_to(self.root.resolve()) or p.stat().st_size != item["bytes"] or sha256_file(p) != item["sha256"]:
                raise ValueError("correction file mismatch")
        self.arrays = {split: {name: np.load(self.root / f"{split}.{name}.npy", mmap_mode="r", allow_pickle=False) for name in ("source", "target", "weight", "mode", "identity")} for split in ("train", "validation")}
        self.pools = {split: {i: np.flatnonzero(a["identity"] == i) for i in (0, 1)} for split, a in self.arrays.items()}
        for split, a in self.arrays.items():
            shape = (r["partitions"][split]["rows"], r["sequence_length"])
            if any(a[k].shape != shape for k in ("source", "target", "weight")) or any(a[k].shape != shape[:1] for k in ("mode", "identity")) or any(not len(p) for p in self.pools[split].values()):
                raise ValueError("invalid correction array shape or pools")

    def sample(self, split, batch, identity_fraction, generator):
        import torch
        clean = round(batch * identity_fraction)
        indices = np.concatenate([pool[torch.randint(len(pool), (count,), generator=generator).numpy()] for pool, count in ((self.pools[split][1], clean), (self.pools[split][0], batch - clean))])
        return {name: torch.from_numpy(np.array(values[indices], dtype=np.float32 if name == "weight" else np.int64)) for name, values in self.arrays[split].items() if name != "identity"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("corpus", "language-data", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--train-sentences", type=int, default=50000)
    p.add_argument("--validation-sentences", type=int, default=3000)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
