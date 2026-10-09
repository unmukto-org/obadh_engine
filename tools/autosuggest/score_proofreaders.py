"""Score a paired teacher development run with the upstream IndiGEC GLEU code.

The upstream implementation and input files are hashed in the result. Reference
targets contain residual spelling/formatting issues; changes to them are reported
as reference changes, not established false corrections. Handwritten probes are
small development checks, not a representative keyboard benchmark.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

from tools.autosuggest.compare_proofreaders import normalize, protected
from tools.corpus.provenance import sha256_file, write_json


def main():
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--gemma", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("indigec_gleu", args.upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = {}
    receipts = {}
    for name in ("gemma", "qwen"):
        root = getattr(args, name)
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        if not report["completed"]:
            raise ValueError("cannot score an incomplete run")
        runs[name] = [
            json.loads(line)
            for line in (root / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        receipts[name] = {
            "contract": report["contract"],
            "rows_sha256": sha256_file(root / "rows.jsonl"),
        }
    for name in (
        "bank_sha256",
        "batch",
        "thinking",
        "max_new_tokens",
        "decoding",
        "script_sha256",
    ):
        if receipts["gemma"]["contract"][name] != receipts["qwen"]["contract"][name]:
            raise ValueError(f"unmatched evaluation condition: {name}")
    bank = runs["gemma"]
    if [(r["id"], r["source"], r["target"]) for r in bank] != [
        (r["id"], r["source"], r["target"]) for r in runs["qwen"]
    ]:
        raise ValueError("unpaired input rows")
    error_rows = [r for r in bank if r["kind"] == "indigec_error"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for key in ("source", "target"):
            (root / key).write_text(
                "\n".join(normalize(r[key]) for r in error_rows) + "\n",
                encoding="utf-8",
            )
        gleu = module.GLEU(4)
        gleu.load_sources(root / "source")
        gleu.load_references([root / "target"])
        statistics = {}
        outcomes = {}
        for name in ("unchanged", "gemma", "qwen"):
            rows = (
                error_rows
                if name == "unchanged"
                else [r for r in runs[name] if r["kind"] == "indigec_error"]
            )
            stats = []
            for i, row in enumerate(rows):
                answer = row["source"] if name == "unchanged" else row["answer"] or ""
                gleu.load_hypothesis_sentence(normalize(answer).split())
                stats.append(list(gleu.gleu_stats(i, r_ind=0)))
            statistics[name] = np.asarray(stats, dtype=np.int64)
            outcomes[name] = {
                "corpus_gleu": 100 * gleu.gleu(statistics[name].sum(axis=0).tolist())
            }
            if name != "unchanged":
                probes = [r for r in runs[name] if r["kind"].startswith("probe/")]
                clean = [
                    r
                    for r in probes
                    if normalize(r["source"]) == normalize(r["target"])
                ]
                reference = [r for r in runs[name] if r["kind"] == "indigec_clean"]
                outcomes[name].update(
                    correction_reference_matches=sum(
                        r["answer"] == normalize(r["target"]) for r in rows
                    ),
                    correction_count=len(rows),
                    reference_target_changes=sum(
                        r["answer"] != normalize(r["source"]) for r in reference
                    ),
                    reference_target_count=len(reference),
                    probe_matches=sum(
                        r["answer"] == normalize(r["target"]) for r in probes
                    ),
                    probe_count=len(probes),
                    clean_probe_changes=sum(
                        r["answer"] != normalize(r["source"]) for r in clean
                    ),
                    clean_probe_count=len(clean),
                    protected_probe_changes=sum(
                        protected(r["source"]) != protected(r["answer"] or "")
                        for r in clean
                        if protected(r["source"])
                    ),
                )
        rng = np.random.default_rng(20261005)
        differences = []
        for _ in range(2000):
            sample = rng.integers(0, len(error_rows), len(error_rows))
            scores = {
                name: 100 * gleu.gleu(values[sample].sum(axis=0).tolist())
                for name, values in statistics.items()
            }
            differences.append(scores["gemma"] - scores["qwen"])
    report = dict(
        scope=__doc__,
        outcomes=outcomes,
        paired_bootstrap_gemma_minus_qwen_gleu_95pct=np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
        bootstrap_replicates=2000,
        receipts=receipts,
        gleu_implementation_sha256=sha256_file(args.upstream),
        scorer_sha256=sha256_file(Path(__file__)),
        tokenization="NFC, outer whitespace stripped, whitespace tokenization; punctuation retained",
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "outcomes": outcomes,
                "paired_bootstrap_gleu_difference": report[
                    "paired_bootstrap_gemma_minus_qwen_gleu_95pct"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
