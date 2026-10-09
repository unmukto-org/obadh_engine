"""Bounded correction evaluation; synthetic restoration is not human accuracy.

Raw model outputs and guarded fallbacks are reported separately. IndiGEC is a
development diagnostic with imperfect references, never training supervision.
Grapheme edit overlap below is an explicit local metric, not M2/ERRANT F0.5.
"""

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import gzip
import json
from pathlib import Path
import time
import unicodedata

import regex
import torch
from tokenizers import Tokenizer

from tools.autosuggest.compare_proofreaders import make_bank, normalize, protected
from tools.autosuggest.decode_subword_words import byte_alphabet
from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.data import PairData
from tools.correction.model import CorrectionConfig, Corrector
from tools.correction.edit_policy import token_groups


def edits(source, target):
    a, b = regex.findall(r"\X", source), regex.findall(r"\X", target)
    return {(i, j, "".join(b[k:l])) for tag, i, j, k, l in
            SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if tag != "equal"}


def decode_strict(tokens, pieces):
    if 2 not in tokens:
        return None, "missing_eos"
    tokens = tokens[:tokens.index(2)]
    if not tokens or any(t not in pieces for t in tokens):
        return None, "empty_or_control_token"
    try:
        text = b"".join(pieces[t] for t in tokens).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, "invalid_utf8"
    if not text.strip() or "\ufffd" in text or any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in text):
        return None, "invalid_text"
    return unicodedata.normalize("NFC", text), None


def measure(rows):
    groups = defaultdict(Counter)
    for row in rows:
        source, target, raw, answer = (row[k] for k in ("source", "target", "raw_answer", "answer"))
        for name in (row["kind"], "all"):
            c = groups[name]
            c["count"] += 1
            c["in_capacity"] += not row["capacity_exceeded"]
            c["raw_invalid"] += raw is None
            c["raw_exact_reference"] += raw == target
            c["raw_changed"] += raw is not None and raw != source
            c["guarded_exact_reference"] += answer == target
            c["guarded_changed"] += answer != source
            c["fallbacks"] += row["fallback_reason"] is not None
            if source == target:
                c["identity_count"] += 1
                c["raw_identity_preserved"] += raw == source
                c["guarded_identity_preserved"] += answer == source
            if protected(source):
                c["protected_count"] += 1
                c["raw_protected_changed_or_invalid"] += raw is None or protected(raw) != protected(source)
            expected, proposed = edits(source, target), edits(source, answer)
            c["reference_grapheme_edits"] += len(expected)
            c["proposed_grapheme_edits"] += len(proposed)
            c["matched_grapheme_edits"] += len(expected & proposed)
    result = {}
    for name, c in groups.items():
        tp, proposed, expected = c["matched_grapheme_edits"], c["proposed_grapheme_edits"], c["reference_grapheme_edits"]
        result[name] = dict(c, grapheme_edit_precision=tp / proposed if proposed else None,
                            grapheme_edit_recall=tp / expected if expected else None,
                            grapheme_edit_f05=1.25 * tp / (.25 * expected + proposed) if expected + proposed else None)
    return result


def extra_cases(path):
    """Load explicitly identified development cases, never training records."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(set(r) != {"id", "kind", "source", "target", "mode"} for r in rows):
        raise ValueError("invalid diagnostic bank schema")
    if any(not all(isinstance(r[k], str) and r[k].strip() for k in ("id", "kind", "source", "target"))
           or not r["id"].startswith("diagnostic:") or not r["kind"].startswith("diagnostic/")
           or type(r["mode"]) is not int or r["mode"] not in (0, 1) for r in rows):
        raise ValueError("invalid diagnostic bank record")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate diagnostic case identity")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "checkpoint", "dev", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--per-kind", type=int, default=64)
    p.add_argument("--dev-count", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--edit-threshold", type=float, default=.5)
    p.add_argument("--edit-confidence", choices=("action", "joint"), default="joint")
    p.add_argument("--extra-bank", type=Path, help="Explicit development-only JSONL cases")
    args = p.parse_args()
    if min(args.per_kind, args.batch) < 1:
        p.error("positive limits required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / ".building").touch()
    torch.set_num_threads(8)
    data = PairData(args.data)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["kind"] not in ("obadh-contextual-corrector", "obadh-contextual-edit-corrector") or any(state["data_provenance"][key] != data.receipt[key] for key in ("corpus_id", "tokenizer_sha256", "vocab_size", "sequence_length")):
        raise ValueError("checkpoint/data mismatch")
    if state["kind"] == "obadh-contextual-edit-corrector":
        from tools.correction.edit_model import EditConfig, EditCorrector
        model = EditCorrector(EditConfig(**state["config"]))
        model_file = "edit_model.py"
    else:
        model = Corrector(CorrectionConfig(**state["config"]))
        model_file = "model.py"
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(args.device).eval()
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    alphabet = byte_alphabet()
    pieces = {index: bytes(alphabet[c] for c in token) for token, index in tokenizer.get_vocab().items() if index > 3}
    groups = defaultdict(list)
    with gzip.open(args.data / "validation.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "validation":
                raise ValueError("unexpected evaluation split")
            if len(groups[row["kind"]]) < args.per_kind:
                groups[row["kind"]].append(dict(id=row["id"], kind="synthetic/" + row["kind"], source=row["source_text"], target=row["target"], mode=row["mode"]))
    bank = [r for rows in groups.values() for r in rows]
    bank += [dict(r, mode=0) for r in make_bank(args.dev, args.dev_count)]
    bank += [dict(id=f"prefix:{i}", kind="probe/prefix_identity", source=s, target=s, mode=1)
             for i, s in enumerate(("আমি আজ", "তুমি কি", "কাল meeting", "রিমা বলল, “আমি", "এই বইটা খুব", "আমার ID AB123"))]
    if args.extra_bank:
        bank += extra_cases(args.extra_bank)
    contract = dict(checkpoint_sha256=sha256_file(args.checkpoint), step=state["step"], data_id=data.receipt["data_id"], training_data_id=state["contract"]["data_id"],
                    bank_sha256=digest_json(bank), script_sha256=sha256_file(Path(__file__)),
                    model_sha256=sha256_file(Path(__file__).with_name(model_file)),
                    dev_files={n: sha256_file(args.dev / n) for n in ("dev.src", "dev.tgt")},
                    batch=args.batch, device=args.device, torch=torch.__version__, precision="float32",
                    edit_policy_sha256=sha256_file(Path(__file__).with_name("edit_policy.py")),
                    extra_bank_sha256=sha256_file(args.extra_bank) if args.extra_bank else None,
                    decoding=dict(algorithm="atomic token edits" if model_file == "edit_model.py" else "bounded greedy", edit_threshold=args.edit_threshold if model_file == "edit_model.py" else None, edit_confidence=args.edit_confidence if model_file == "edit_model.py" else None, calibrated=False))
    write_json(args.output / "contract.json", contract)
    del state
    started, rows = time.monotonic(), []
    with torch.inference_mode(), (args.output / "rows.jsonl").open("w", encoding="utf-8") as output:
        for start in range(0, len(bank), args.batch):
            batch = [dict(r, source=normalize(r["source"]), target=normalize(r["target"])) for r in bank[start:start + args.batch]]
            encodings = [tokenizer.encode(r["source"], add_special_tokens=False) for r in batch]
            encoded = [[1] + e.ids + [2] for e in encodings]
            active = [i for i, ids in enumerate(encoded) if len(ids) <= model.config.sequence_length and not any(t in (0, 1, 3) for t in ids[1:])]
            decoded = {}
            if active:
                length = max(len(encoded[i]) for i in active)
                source = torch.tensor([encoded[i] + [0] * (length - len(encoded[i])) for i in active], device=args.device)
                mode = torch.tensor([batch[i]["mode"] for i in active], device=args.device)
                options = dict(threshold=args.edit_threshold, confidence=args.edit_confidence) if model_file == "edit_model.py" else {}
                if model_file == "edit_model.py":
                    groups = [[-1] + token_groups(batch[i]["source"], encodings[i].offsets) + [-2] + [-3] * (length - len(encoded[i])) for i in active]
                    options["edit_groups"] = torch.tensor(groups, device=args.device)
                tokens, probabilities, _ = model.greedy(source, mode, **options)
                score_rows = probabilities.tolist() if probabilities is not None else [None] * len(active)
                for i, ids, scores in zip(active, tokens.tolist(), score_rows):
                    raw, reason = decode_strict(ids, pieces)
                    end = ids.index(2) + 1 if 2 in ids else len(ids)
                    decoded[i] = (raw, reason, min(scores[:end]) if scores is not None else None)
            for i, row in enumerate(batch):
                raw, reason, minimum = decoded.get(i, (None, "capacity_exceeded", None))
                if raw is not None and protected(raw) != protected(row["source"]):
                    reason = "protected_span_changed"
                answer = row["source"] if reason else raw
                result = dict(row, raw_answer=raw, answer=answer, fallback_reason=reason,
                              capacity_exceeded=i not in active, minimum_token_probability=minimum)
                rows.append(result)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps(dict(event="correction_evaluate", rows=len(rows), total=len(bank))), flush=True)
    report = dict(completed=True, contract=contract, scope=__doc__, groups=measure(rows),
                  seconds=time.monotonic() - started, rows_sha256=sha256_file(args.output / "rows.jsonl"))
    write_json(args.output / "report.json", report)
    (args.output / ".building").unlink()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
