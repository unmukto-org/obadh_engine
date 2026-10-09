"""Hard-distillation label validation on public training pairs only.

The teacher never sees the reference target. A label is admitted only when its
untruncated answer exactly recovers that source-derived reference and preserves
protected Latin/number spans. Agreement is a precision filter, not human review.
"""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time

from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.autosuggest.compare_proofreaders import SYSTEM, parse_answer, protected

PREFIX_SYSTEM = SYSTEM + " The input is an unfinished typing prefix. Never complete it or add terminal punctuation. If it can be continued into an acceptable sentence, preserve it exactly."


def select_bank(root, per_kind):
    manifest = json.loads((root / "manifest.json").read_text())
    if (root / ".building").exists() or manifest["kind"] != "obadh-correction-pairs":
        raise ValueError("complete correction training data required")
    item = next(i for i in manifest["files"] if i["path"] == "train.jsonl.gz")
    if sha256_file(root / item["path"]) != item["sha256"]:
        raise ValueError("training pairs hash mismatch")
    groups = {}
    with gzip.open(root / item["path"], "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "train":
                raise ValueError("non-training example in teacher data")
            bucket = groups.setdefault(row["kind"], [])
            if len(bucket) < per_kind:
                bucket.append(row)
    bank = sorted((r for bucket in groups.values() for r in bucket), key=lambda r: r["id"])
    if not bank or len({r["id"] for r in bank}) != len(bank):
        raise ValueError("empty or duplicate teacher bank")
    return manifest, bank


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--per-kind", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int)
    p.add_argument("--stop-epoch", type=float, required=True)
    args = p.parse_args()
    if min(args.per_kind, args.batch, args.max_new_tokens) < 1:
        p.error("positive limits required")
    receipt, bank = select_bank(args.data, args.per_kind)
    selection = json.loads((args.model_root / "model-selection.json").read_text())
    contract = dict(
        model=selection["model"], revision=selection["revision"], data_id=receipt["data_id"], bank_sha256=digest_json(bank),
        per_kind=args.per_kind, batch=args.batch, max_new_tokens=args.max_new_tokens,
        thinking=False, script_sha256=sha256_file(Path(__file__)), helper_sha256=sha256_file(Path(__file__).parents[1] / "autosuggest/compare_proofreaders.py"),
        torch=torch.__version__, transformers=transformers.__version__, scope=__doc__,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("teacher resume contract mismatch")
    write_json(path, contract)
    results = args.output / "rows.jsonl"
    rows = []
    if results.exists():
        with results.open("r+b") as f:
            position = 0
            while line := f.readline():
                if not line.endswith(b"\n"):
                    f.truncate(position)
                    break
                rows.append(json.loads(line))
                position += len(line)
    if [r["id"] for r in rows] != [r["id"] for r in bank[:len(rows)]]:
        raise ValueError("teacher results differ from fixed bank")
    counts = Counter((r["kind"], r["accepted"]) for r in rows)
    started = time.monotonic()
    torch.set_num_threads(8)
    invocation_end = min(len(bank), len(rows) + args.limit) if args.limit else len(bank)
    if len(rows) < invocation_end:
        tokenizer = AutoTokenizer.from_pretrained(args.model_root / "model", trust_remote_code=False, padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(args.model_root / "model", dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", trust_remote_code=False).eval()
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos or []
        with results.open("a", encoding="utf-8") as handle:
            for start in range(len(rows), invocation_end, args.batch):
                if time.time() >= args.stop_epoch:
                    break
                batch = bank[start:min(invocation_end, start + args.batch)]
                prompts = [tokenizer.apply_chat_template([
                    dict(role="system", content=PREFIX_SYSTEM if r["mode"] else SYSTEM),
                    dict(role="user", content=r["source_text"]),
                ], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
                for row, tokens in zip(batch, outputs[:, inputs.input_ids.shape[1]:]):
                    raw = tokenizer.decode(tokens, skip_special_tokens=False)
                    ended = any(t in eos for t in tokens.tolist())
                    answer = parse_answer(raw, "gemma", False) if ended else None
                    accepted = answer == row["target"] and protected(row["source_text"]) == protected(row["target"])
                    result = dict(row, teacher_answer=answer, ended=ended, accepted=accepted, raw=raw)
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    rows.append(result)
                    counts[(row["kind"], accepted)] += 1
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps(dict(event="teacher_validation", rows=len(rows), total=len(bank), accepted=sum(v for (_, yes), v in counts.items() if yes), seconds=time.monotonic() - started)), flush=True)
    report = dict(contract=contract, completed=len(rows) == len(bank), rows=len(rows), bank_rows=len(bank), acceptance={kind: {str(accepted): counts[(kind, accepted)] for accepted in (False, True)} for kind in sorted({k for k, _ in counts})}, peak_cuda_bytes=torch.cuda.max_memory_allocated(), seconds_this_run=time.monotonic() - started)
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
