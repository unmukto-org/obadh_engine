"""Admit only completed, source-blind Gemma agreement into training.

Every accepted row must match its original training record, recover the known
reference exactly and retain protected spans. The parent's validation partition
is copied unchanged. These remain machine-validated labels, not human gold.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from tools.autosuggest.compare_proofreaders import protected
from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.data import PairData, edited_target_weights
from tools.correction.teacher import select_bank


def admitted_rows(bank, rows):
    if [r["id"] for r in rows] != [r["id"] for r in bank]:
        raise ValueError("teacher bank incomplete or reordered")
    accepted = []
    for original, row in zip(bank, rows):
        if any(row.get(k) != v for k, v in original.items()) or row["split"] != "train":
            raise ValueError("teacher row changed original training evidence")
        expected = row["ended"] and row["teacher_answer"] == original["target"] and protected(original["source_text"]) == protected(original["target"])
        if row["accepted"] != expected:
            raise ValueError("teacher acceptance mismatch")
        if expected:
            accepted.append(dict(original, supervision="Gemma source-blind exact reference agreement; not human gold"))
    if not accepted or not any(r["source_text"] == r["target"] for r in accepted) or not any(r["source_text"] != r["target"] for r in accepted):
        raise ValueError("teacher data requires corrections and identity controls")
    return accepted


def publish_pairs(parent, accepted, output, contract, rows_sha256, builder, scope):
    if not accepted or len({r["id"] for r in accepted}) != len(accepted) or any(r["split"] != "train" or r["mode"] not in (0, 1) for r in accepted):
        raise ValueError("invalid admitted training records")
    accepted = sorted(accepted, key=lambda r: r["id"])
    output.mkdir(parents=True, exist_ok=False)
    (output / ".building").touch()
    shutil.copyfile(parent.root / "tokenizer.json", output / "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    length = parent.receipt["sequence_length"]
    source = np.zeros((len(accepted), length), dtype="<u2")
    target = np.zeros_like(source)
    weight = np.zeros_like(source, dtype="<f4")
    mode = np.zeros(len(accepted), dtype="u1")
    identity = np.zeros_like(mode)
    with gzip.open(output / "train.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i, row in enumerate(accepted):
            s = [1] + tokenizer.encode(row["source_text"], add_special_tokens=False).ids + [2]
            t = tokenizer.encode(row["target"], add_special_tokens=False).ids + [2]
            if max(len(s), len(t)) > length:
                raise ValueError("admitted row exceeds capacity")
            source[i, :len(s)], target[i, :len(t)] = s, t
            weight[i, :len(t)] = edited_target_weights(s[1:], t)
            mode[i], identity[i] = row["mode"], row["source_text"] == row["target"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name, array in (("source", source), ("target", target), ("weight", weight), ("mode", mode), ("identity", identity)):
        np.save(output / f"train.{name}.npy", array, allow_pickle=False)
    for item in parent.receipt["files"]:
        if item["path"].startswith("validation."):
            shutil.copyfile(parent.root / item["path"], output / item["path"])
    files = [dict(path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path)) for path in sorted(output.iterdir()) if path.name != ".building"]
    receipt = dict(kind="obadh-correction-pairs", version=1, teacher_validated=True,
                   parent_data_id=parent.receipt["data_id"], corpus_id=parent.receipt["corpus_id"],
                   tokenizer_sha256=parent.receipt["tokenizer_sha256"], vocab_size=parent.receipt["vocab_size"], sequence_length=length,
                   partitions=dict(train=dict(rows=len(accepted), kinds=dict(Counter(r["kind"] for r in accepted))), validation=parent.receipt["partitions"]["validation"]),
                   files=files, modes=parent.receipt["modes"], teacher_contract=contract,
                   teacher_rows_sha256=rows_sha256, builder_sha256=sha256_file(builder), publisher_sha256=sha256_file(Path(__file__)), supervision=scope)
    receipt["data_id"] = digest_json(receipt)
    write_json(output / "manifest.json", receipt)
    (output / ".building").unlink()
    PairData(output)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "teacher", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    parent = PairData(args.parent)
    report = json.loads((args.teacher / "report.json").read_text())
    contract = json.loads((args.teacher / "contract.json").read_text())
    if not report["completed"] or report["contract"] != contract or contract["data_id"] != parent.receipt["data_id"]:
        raise ValueError("complete matched teacher run required")
    _, bank = select_bank(args.parent, contract["per_kind"])
    if digest_json(bank) != contract["bank_sha256"]:
        raise ValueError("teacher bank identity mismatch")
    rows = [json.loads(line) for line in (args.teacher / "rows.jsonl").read_text().splitlines()]
    accepted = admitted_rows(bank, rows)
    receipt = publish_pairs(parent, accepted, args.output, contract, sha256_file(args.teacher / "rows.jsonl"), Path(__file__), __doc__)
    print(json.dumps(dict(data_id=receipt["data_id"], accepted=len(accepted), kinds=receipt["partitions"]["train"]["kinds"]), indent=2))


if __name__ == "__main__":
    main()
