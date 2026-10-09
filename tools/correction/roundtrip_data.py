"""Admit grammar generation only after blind recovery and explicit critique.

Requested error categories are generation directives, not human-verified error
annotations. Validated references also produce identity/prefix controls. No
development or test example is used for training.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import regex

from tools.corpus.provenance import digest_json, sha256_file
from tools.correction.data import PairData, spans
from tools.correction.distill_data import publish_pairs
from tools.correction.teacher_roundtrip import select_bank, candidate, critic_accepts


def conservative_rejection(source, target):
    # A declarative Bangla word sequence can also be a legitimate question.
    # Blind recovery and the same teacher's critique do not establish intent.
    if any(source.count(mark) != target.count(mark) for mark in ("?", "!", "…")):
        return "ambiguous_punctuation_intent"
    source_words = regex.findall(r"[\p{L}\p{M}\p{N}]+", source)
    target_words = regex.findall(r"[\p{L}\p{M}\p{N}]+", target)
    if len(source_words) == len(target_words):
        for first, second in zip(source_words, target_words):
            if first == second:
                continue
            # Third-person honorific choice is often stylistic, particularly
            # with named subjects. Conservatively omit these teacher labels;
            # this filter does not claim to parse Bangla subject agreement.
            for formal, plain in (("লেন", "ল"), ("তেন", "ত"), ("বেন", "বে"),
                                  ("ছেন", "ছে"), ("েন", "ে"), ("ান", "ায়")):
                if any(a.endswith(formal) and a[:-len(formal)] + plain == b
                       for a, b in ((first, second), (second, first))):
                    return "ambiguous_honorific_choice"
    return None


def admitted(bank, rows, exclusions=None):
    if [r["id"] for r in bank] != [r["id"] for r in rows]:
        raise ValueError("incomplete or mismatched grammar bank")
    result = []
    for original, row in zip(bank, rows):
        if any(row.get(k) != v for k, v in original.items()) or row["split"] != "train":
            raise ValueError("grammar row changed original evidence")
        source, reason = candidate(row["generation"]["answer"], row["target"], row["requested_error"])
        expected = bool(source is not None and row["generation"]["ended"] and row.get("verification", {}).get("ended") and row["verification"]["answer"] == row["target"] and row.get("critique", {}).get("ended") and critic_accepts(row["critique"]["answer"]))
        # A capacity rejection can be stricter than the text-level filter.
        if row["rejection"] == "student_capacity":
            expected = False
        if row["accepted"] != expected or (expected and source != row["proposed_source"]):
            raise ValueError("grammar admission does not match teacher evidence")
        if not expected:
            continue
        reason = conservative_rejection(source, row["target"])
        if reason:
            if exclusions is not None:
                exclusions[reason] += 1
            continue
        common = {k: v for k, v in original.items() if k not in ("id", "kind", "source_text", "target", "mode", "supervision")}
        candidates = [("teacher_grammar/" + row["requested_error"], source, row["target"], 0), ("identity", row["target"], row["target"], 0)]
        words = spans(row["target"])
        if len(words) >= 4:
            prefix = row["target"][:words[len(words) // 2 - 1].end()]
            candidates.append(("prefix_identity", prefix, prefix, 1))
        for kind, source, target, mode in candidates:
            result.append(dict(common, kind=kind, source_text=source, target=target, mode=mode,
                               id=digest_json(["roundtrip-admitted-v1", original["id"], kind, source, target, mode]),
                               supervision="Gemma generation, blind recovery and separate critique; not human gold"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "teacher", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    parent = PairData(args.parent)
    report = json.loads((args.teacher / "report.json").read_text())
    contract = json.loads((args.teacher / "contract.json").read_text())
    if not report["completed"] or report["contract"] != contract or contract["data_id"] != parent.receipt["data_id"] or report["rows_sha256"] != sha256_file(args.teacher / "rows.jsonl"):
        raise ValueError("complete verified grammar generation required")
    _, bank = select_bank(args.parent, contract["count"], contract.get("source_offset", 0))
    if digest_json(bank) != contract["bank_sha256"]:
        raise ValueError("grammar bank identity mismatch")
    rows = [json.loads(line) for line in (args.teacher / "rows.jsonl").read_text().splitlines()]
    exclusions = Counter()
    accepted = admitted(bank, rows, exclusions)
    if sum(r["kind"].startswith("teacher_grammar/") for r in accepted) < 64:
        raise ValueError("insufficient accepted grammar supervision")
    admission_contract = dict(generation=contract, exclusions=dict(exclusions),
                              policy="Omit ambiguous question/emphasis punctuation and third-person honorific substitutions even when the teacher agrees")
    receipt = publish_pairs(parent, accepted, args.output, admission_contract, report["rows_sha256"], Path(__file__), __doc__)
    print(json.dumps(dict(data_id=receipt["data_id"], rows=len(accepted), kinds=receipt["partitions"]["train"]["kinds"], exclusions=dict(exclusions)), indent=2))


if __name__ == "__main__":
    main()
