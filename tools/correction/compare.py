"""Compare completed correction runs on identical development cases.

References are imperfect; GLEU and exact matches are diagnostics. This does not
turn source-derived synthetic restoration into a native-user accuracy measure.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
from tools.corpus.provenance import sha256_file, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, help="NAME=completed-evaluation-directory")
    p.add_argument("--upstream-gleu", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runs, receipts, reports = {}, {}, {}
    for item in args.run:
        name, value = item.split("=", 1)
        root = Path(value)
        if name in runs or (root / ".building").exists():
            raise ValueError("duplicate run or incomplete evaluation")
        report = json.loads((root / "report.json").read_text())
        if not report["completed"] or sha256_file(root / "rows.jsonl") != report["rows_sha256"]:
            raise ValueError("invalid evaluation receipt")
        runs[name] = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
        reports[name] = report
        receipts[name] = dict(report_sha256=sha256_file(root / "report.json"), **report["contract"])
    if len({r["bank_sha256"] for r in receipts.values()}) != 1:
        raise ValueError("unpaired evaluation banks")
    first = next(iter(runs.values()))
    bank = [(r["id"], r["source"], r["target"], r["mode"]) for r in first]
    if any([(r["id"], r["source"], r["target"], r["mode"]) for r in rows] != bank for rows in runs.values()):
        raise ValueError("unpaired evaluation rows")
    spec = importlib.util.spec_from_file_location("indigec_gleu", args.upstream_gleu)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    error = [r for r in first if r["kind"] == "indigec_error"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for key in ("source", "target"):
            (root / key).write_text("\n".join(r[key] for r in error) + "\n")
        gleu = module.GLEU(4)
        gleu.load_sources(root / "source")
        gleu.load_references([root / "target"])
        stats = {}
        conditions = {"unchanged": [r["source"] for r in error]}
        for name, rows in runs.items():
            selected = [r for r in rows if r["kind"] == "indigec_error"]
            for field in ("raw_answer", "answer"):
                conditions[name + "/" + field] = [r[field] or "" for r in selected]
        for name, answers in conditions.items():
            values = []
            for i, answer in enumerate(answers):
                gleu.load_hypothesis_sentence(answer.split())
                values.append(list(gleu.gleu_stats(i, r_ind=0)))
            stats[name] = np.asarray(values, dtype=np.int64)
        scores = {name: 100 * gleu.gleu(values.sum(0).tolist()) for name, values in stats.items()}
        rng = np.random.default_rng(20261005)
        differences = {name: [] for name in stats if name != "unchanged"}
        for _ in range(2000):
            sample = rng.integers(0, len(error), len(error))
            baseline = 100 * gleu.gleu(stats["unchanged"][sample].sum(0).tolist())
            for name in differences:
                differences[name].append(100 * gleu.gleu(stats[name][sample].sum(0).tolist()) - baseline)
    summary = {}
    for name, rows in runs.items():
        synthetic_errors = [r for r in rows if r["kind"].startswith("synthetic/") and r["source"] != r["target"]]
        identities = [r for r in rows if r["kind"].startswith("probe/") and r["source"] == r["target"]]
        probes = [r for r in rows if r["kind"].startswith("probe/")]
        summary[name] = dict(synthetic_exact_restoration=sum(r["raw_answer"] == r["target"] for r in synthetic_errors), synthetic_error_count=len(synthetic_errors),
                             authored_identity_preserved=sum(r["raw_answer"] == r["source"] for r in identities), authored_identity_count=len(identities),
                             authored_probe_matches=sum(r["raw_answer"] == r["target"] for r in probes), authored_probe_count=len(probes),
                             groups=reports[name]["groups"])
    result = dict(scope=__doc__, indigec_dev_count=len(error), corpus_gleu=scores,
                  paired_gleu_difference_from_unchanged_95pct={n: np.quantile(v, [.025, .975]).tolist() for n, v in differences.items()},
                  bootstrap_replicates=2000, outcomes=summary, receipts=receipts,
                  upstream_gleu_sha256=sha256_file(args.upstream_gleu), scorer_sha256=sha256_file(Path(__file__)))
    write_json(args.output, result)
    print(json.dumps(dict(corpus_gleu=scores, outcomes={n: {k: v for k, v in r.items() if k != "groups"} for n, r in summary.items()}), indent=2))


if __name__ == "__main__":
    main()
