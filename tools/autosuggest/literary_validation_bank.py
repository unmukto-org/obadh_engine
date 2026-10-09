"""Freeze a punctuation-preserving book validation bank before model comparison.

Uses validation works only; at most four examples per window separated by 80
characters. This is book continuation, not native chat or grammatical gold data.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from tools.corpus.provenance import digest_json, sha256_file, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite validation bank")
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    if (args.corpus / ".building").exists() or manifest["dataset_id"] != digest_json(
        {k: v for k, v in manifest.items() if k != "dataset_id"}
    ):
        raise ValueError("invalid corpus")
    receipt = next(r for r in manifest["files"] if r["path"] == "validation.jsonl.gz")
    path = args.corpus / receipt["path"]
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("validation digest mismatch")
    examples = []
    with gzip.open(path, "rt", encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            text = row["text"]
            words = list(re.finditer(r"[\u0980-\u09ff\u200c\u200d]+", text))
            candidates = []
            for match in words[3:]:
                position = match.start()
                if not text[position - 1].isspace() or text[position - 1] == "\n":
                    continue
                if (
                    match.end() < len(text)
                    and unicodedata.category(text[match.end()])[0] in "LMN"
                ):
                    continue
                key = digest_json([row["sha256"], position])
                candidates.append((key, position, match.group()))
            selected = []
            for key, position, target in sorted(candidates):
                if any(abs(position - other) < 80 for other in selected):
                    continue
                start = max(0, position - 500)
                if start:
                    while start < position and not text[start].isspace():
                        start += 1
                context = text[start:position].strip()
                if not context:
                    continue
                examples.append(
                    {
                        "id": key,
                        "work_id": row["work_id"],
                        "authors": row["authors"],
                        "context": context,
                        "target": target,
                        "position": position,
                        "domain": "literary-validation",
                    }
                )
                selected.append(position)
                if len(selected) == 4:
                    break
    if not examples:
        raise ValueError("empty validation bank")
    with args.output.open("w", encoding="utf-8") as out:
        for row in sorted(examples, key=lambda r: r["id"]):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        Path(str(args.output) + ".manifest.json"),
        {
            "scope": __doc__,
            "role": "validation",
            "corpus_id": manifest["dataset_id"],
            "rows": len(examples),
            "sha256": sha256_file(args.output),
            "builder_sha256": sha256_file(Path(__file__)),
        },
    )
    print(len(examples), sha256_file(args.output))


if __name__ == "__main__":
    main()
