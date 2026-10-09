"""Learn and calibrate an edit-acceptance model on disjoint development data.

Features use input text, proposed output, generator likelihoods, and optional
training-corpus word frequencies. Reference text supplies training/evaluation
labels, never prediction features. This is a
reference-exactness verifier, not a guarantee of grammatical or semantic truth.
"""

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import random
import re
import unicodedata

import numpy as np

from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.byte_data import encode, read_rows, target_weights

FEATURES = ["mean_logp_gain", "sum_logp_gain", "candidate_mean_logp", "copy_mean_logp",
            "edited_mean_logp", "edited_min_logp", "edit_byte_fraction", "length_ratio",
            "source_word_count", "word_edit_fraction", "punctuation_delta", "prefix_mode", "word_mode"]
LEXICAL_FEATURES = ["edited_source_oov_fraction", "edited_candidate_oov_fraction",
                    "edited_source_mean_log_frequency", "edited_candidate_mean_log_frequency",
                    "edited_source_words", "edited_candidate_words", "edited_word_frequency_gain",
                    "nonword_to_known_word"]


def features(source, candidate, mode, candidate_logp, source_logp):
    if len(candidate_logp) != len(encode(candidate)) or len(source_logp) != len(encode(source)):
        raise ValueError("likelihood alignment mismatch")
    candidate_logp, source_logp = np.asarray(candidate_logp), np.asarray(source_logp)
    if not np.isfinite(candidate_logp).all() or not np.isfinite(source_logp).all():
        raise ValueError("nonfinite likelihood features")
    changed = np.asarray(target_weights(source, candidate, 2)) > 1
    edited = candidate_logp[changed]
    if not len(edited):
        edited = candidate_logp[-1:]
    a, b = source.split(), candidate.split()
    distance = sum(max(j-i, l-k) for tag, i, j, k, l in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if tag != "equal")
    punctuation_delta = sum(abs(source.count(c) - candidate.count(c)) for c in "।?!,…;:\"“”‘’")
    return [float(candidate_logp.mean() - source_logp.mean()), float(candidate_logp.sum() - source_logp.sum()),
            float(candidate_logp.mean()), float(source_logp.mean()), float(edited.mean()), float(edited.min()),
            float(changed.mean()), len(candidate_logp) / len(source_logp), len(a), distance / max(1, len(a)),
            punctuation_delta, int(mode == 1), int(mode == 2)]


def make_banks(args):
    from tools.correction.contextual_data import verify
    receipt = verify(args.data)
    evaluation = [json.loads(line) for line in args.evaluation_bank.read_text().splitlines()]
    forbidden = {unicodedata.normalize("NFC", r[k].strip()) for r in evaluation for k in ("source", "target")}
    rng = random.Random(20261006)
    selected = {}
    for split, limit in (("train", args.train_count), ("validation", args.calibration_count)):
        pools = defaultdict(list)
        # Bounded reservoirs by family/identity avoid materializing the corpus.
        seen_count = Counter()
        for row in read_rows(args.data / (split + ".jsonl.gz")):
            if row["source"] in forbidden or row["target"] in forbidden:
                continue
            if row["mode"] == 2:
                continue
            key = (row["family"], row["source"] == row["target"])
            seen_count[key] += 1
            pool = pools[key]
            if len(pool) < limit:
                pool.append(row)
            else:
                index = rng.randrange(seen_count[key])
                if index < limit:
                    pool[index] = row
        keys = sorted(pools)
        for pool in pools.values():
            rng.shuffle(pool)
        chosen, seen_text = [], set()
        while len(chosen) < limit and any(pools.values()):
            for key in keys:
                if not pools[key] or len(chosen) >= limit:
                    continue
                row = pools[key].pop()
                # Do not put variants of the same reference into both sets.
                if row["source"] in forbidden or row["target"] in forbidden or row["target"] in seen_text:
                    continue
                seen_text.add(row["target"])
                chosen.append(dict(id=row["id"], kind="acceptance/" + row["kind"], source=row["source"], target=row["target"], mode=row["mode"]))
        selected[split] = chosen
        forbidden |= {r[k] for r in chosen for k in ("source", "target")}
    if len(selected["train"]) < 1000 or len(selected["validation"]) < 500:
        raise ValueError("insufficient disjoint acceptance examples")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / ".building").touch()
    for split, bank in selected.items():
        (args.output / (split + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bank))
    write_json(args.output / "manifest.json", dict(data_id=receipt["data_id"], excluded_evaluation_bank_sha256=sha256_file(args.evaluation_bank),
               files={s: dict(rows=len(b), sha256=sha256_file(args.output / (s + ".jsonl"))) for s, b in selected.items()},
               scope="Machine-reference labels. Training, calibration and existing evaluation exclude identical source/reference texts; not native-human release gold."))
    (args.output / ".building").unlink()


def score(args):
    import torch
    from tools.correction.byte_train import load_model, collate
    torch.set_num_threads(8)
    report = json.loads((args.predictions / "report.json").read_text())
    if (args.predictions / ".building").exists() or not report["completed"] or sha256_file(args.predictions / "rows.jsonl") != report["rows_sha256"]:
        raise ValueError("incomplete or mismatched generator predictions")
    checkpoint_hash = sha256_file(args.checkpoint)
    if report["contract"]["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("generator checkpoint mismatch")
    model = load_model(args.pretrained)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    limit = state["contract"]["byte_limit"]
    del state
    model.to("cuda").eval()
    rows = [json.loads(line) for line in (args.predictions / "rows.jsonl").read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / ".building").touch()
    results = []

    @torch.inference_mode()
    def likelihood(batch, targets):
        pairs = [dict(r, target=t) for r, t in zip(batch, targets)]
        tensors = collate(pairs, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**tensors, use_cache=False)
        logits = output.logits.float().log_softmax(-1)
        logp = logits.gather(-1, tensors["labels"].clamp_min(0)[..., None]).squeeze(-1)
        return [r[:len(encode(t))] for r, t in zip(logp.tolist(), targets)]

    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        active = [r for r in batch if r["raw_answer"] is not None and r["raw_answer"] != r["source"]
                  and not r["fallback_reason"] and len(encode(r["raw_answer"])) <= limit]
        scored = {}
        if active:
            candidates = likelihood(active, [r["raw_answer"] for r in active])
            copies = likelihood(active, [r["source"] for r in active])
            for r, c, s in zip(active, candidates, copies):
                scored[r["id"]] = features(r["source"], r["raw_answer"], r["mode"], c, s)
        for r in batch:
            results.append(dict(r, features=scored.get(r["id"]), reference_correct=r["raw_answer"] == r["target"]))
        print(json.dumps(dict(event="acceptance_features", rows=len(results), total=len(rows))), flush=True)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    write_json(args.output / "report.json", dict(completed=True, features=FEATURES, rows=len(results),
               checkpoint_sha256=checkpoint_hash, generator_report_sha256=sha256_file(args.predictions / "report.json"),
               generator_contract=report["contract"],
               script_sha256=sha256_file(Path(__file__)), rows_sha256=sha256_file(args.output / "rows.jsonl")))
    (args.output / ".building").unlink()


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def fit_logistic(x, y, ridge=1.):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    beta = np.zeros(z.shape[1])
    penalty = np.diag([0.] + [ridge] * (len(beta) - 1))
    def loss(b):
        logits = z @ b
        return np.logaddexp(0., logits).sum() - y @ logits + .5 * b @ penalty @ b
    for _ in range(100):
        p = sigmoid(z @ beta)
        gradient = z.T @ (p - y) + penalty @ beta
        hessian = (z.T * (p * (1 - p))) @ z + penalty + np.eye(len(beta)) * 1e-8
        direction = np.linalg.solve(hessian, gradient)
        rate = 1.
        current = loss(beta)
        while rate > 1e-8 and loss(beta - rate * direction) > current:
            rate *= .5
        beta -= rate * direction
        if np.linalg.norm(rate * direction) < 1e-7:
            break
    return dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist(), ridge=ridge)


def probabilities(model, x):
    z = (np.asarray(x) - model["mean"]) / model["scale"]
    return sigmoid(np.column_stack([np.ones(len(z)), z]) @ np.asarray(model["beta"]))


def load_scored(path):
    receipt = json.loads((path / "report.json").read_text())
    if (path / ".building").exists() or not receipt["completed"] or receipt["features"] not in (FEATURES, FEATURES + LEXICAL_FEATURES) or sha256_file(path / "rows.jsonl") != receipt["rows_sha256"]:
        raise ValueError("invalid scored data")
    lexical = receipt.get("lexical_vocabulary_sha256")
    if receipt["features"] == FEATURES + LEXICAL_FEATURES and (not isinstance(lexical, str) or len(lexical) != 64):
        raise ValueError("missing lexical vocabulary identity")
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    if any(r["features"] is not None and (len(r["features"]) != len(receipt["features"])
           or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in r["features"])) for r in rows):
        raise ValueError("invalid acceptance feature vector")
    return receipt, rows


def fit(args):
    receipts, banks = {}, {}
    for name, path in (("train", args.train), ("calibration", args.calibration), ("evaluation", args.evaluation)):
        receipts[name], banks[name] = load_scored(path)
    if len({r["checkpoint_sha256"] for r in receipts.values()}) != 1:
        raise ValueError("acceptance features require a single generator checkpoint")
    if len({tuple(r["features"]) for r in receipts.values()}) != 1 or len({r.get("lexical_vocabulary_sha256") for r in receipts.values()}) != 1:
        raise ValueError("acceptance feature schema or vocabulary mismatch")
    texts = [{r[k] for r in banks[s] for k in ("source", "target")} for s in banks]
    if any(texts[i] & texts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("training/calibration/evaluation text overlap")
    train = [r for r in banks["train"] if r["features"] is not None]
    calibration = [r for r in banks["calibration"] if r["features"] is not None]
    if min(len(train), len(calibration)) < 50 or len({r["reference_correct"] for r in train}) < 2:
        raise ValueError("insufficient positive/negative acceptance examples")
    model = fit_logistic(np.asarray([r["features"] for r in train]), np.asarray([r["reference_correct"] for r in train], dtype=float))
    p = probabilities(model, [r["features"] for r in calibration])
    y = np.asarray([r["reference_correct"] for r in calibration])
    objective = getattr(args, "calibration_objective", "proposal-f05")
    total_errors = sum(r["source"] != r["target"] for r in banks["calibration"])
    options = []
    for threshold in sorted(set([0., 1.] + p.tolist())):
        accepted = p >= threshold
        tp, fp, fn = int((accepted & y).sum()), int((accepted & ~y).sum()), int((~accepted & y).sum())
        if objective == "sentence-f05":
            fn = total_errors - tp
        f05 = 1.25 * tp / (1.25 * tp + .25 * fn + fp) if tp else 0.
        options.append(dict(threshold=threshold, f05=f05, correct=tp, incorrect=fp, missed=fn))
    chosen = max(options, key=lambda o: (o["f05"], -o["incorrect"], o["threshold"]))
    model.update(features=receipts["train"]["features"], threshold=chosen["threshold"], selection="Maximum exact-reference F0.5 on separate calibration: " + objective, calibration=chosen,
                 receipts={k: receipts[k] for k in ("train", "calibration")},
                 fitter_sha256=sha256_file(Path(__file__)), numpy=np.__version__, scope=__doc__)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / ".building").touch()
    write_json(args.output / "model.json", model)
    result = []
    for r in banks["evaluation"]:
        score_value = float(probabilities(model, [r["features"]])[0]) if r["features"] is not None else None
        accepted = score_value is not None and score_value >= model["threshold"]
        result.append(dict(r, acceptance_score=score_value, accepted_edit=accepted,
                           accepted_answer=r["raw_answer"] if accepted else r["source"]))
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result))
    by_kind = defaultdict(Counter)
    for r in result:
        for kind in (r["kind"], "all"):
            c = by_kind[kind]
            c["count"] += 1
            c["raw_exact"] += r["raw_answer"] == r["target"]
            c["accepted_exact"] += r["accepted_answer"] == r["target"]
            c["accepted_changes"] += r["accepted_answer"] != r["source"]
            if r["source"] == r["target"]:
                c["clean"] += 1
                c["raw_clean_preserved"] += r["raw_answer"] == r["source"]
                c["accepted_clean_preserved"] += r["accepted_answer"] == r["source"]
    write_json(args.output / "report.json", dict(completed=True, groups=dict(by_kind), calibration=chosen,
               evaluation_receipt=receipts["evaluation"],
               train_proposals=len(train), calibration_proposals=len(calibration), model_sha256=sha256_file(args.output / "model.json"),
               rows_sha256=sha256_file(args.output / "rows.jsonl")))
    # Publish the composed policy through the same paired-evaluation interface.
    # Keep its generator proposal explicitly; raw_answer now means the policy's
    # own output, so existing scorers measure the actual accepted text.
    from tools.correction.evaluate import measure
    evaluation = args.output / "evaluation"
    evaluation.mkdir()
    policy_rows = [dict(r, proposal_answer=r["raw_answer"], raw_answer=r["accepted_answer"],
                        answer=r["accepted_answer"],
                        fallback_reason=r["fallback_reason"] or (
                            "acceptance_rejected" if r["features"] is not None and not r["accepted_edit"] else None))
                   for r in result]
    (evaluation / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in policy_rows))
    contract = dict(receipts["evaluation"]["generator_contract"], calibrated=True,
                    policy="generator plus separately calibrated edit acceptance",
                    acceptance_model_sha256=sha256_file(args.output / "model.json"))
    write_json(evaluation / "contract.json", contract)
    write_json(evaluation / "report.json", dict(completed=True, contract=contract,
               groups=measure(policy_rows), rows_sha256=sha256_file(evaluation / "rows.jsonl"), scope=__doc__))
    (args.output / ".building").unlink()
    print(json.dumps(dict(calibration=chosen, groups=dict(by_kind)), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("banks")
    for n in ("data", "evaluation-bank", "output"):
        b.add_argument("--" + n, type=Path, required=True)
    b.add_argument("--train-count", type=int, default=5000)
    b.add_argument("--calibration-count", type=int, default=1500)
    s = sub.add_parser("score")
    for n in ("checkpoint", "pretrained", "predictions", "output"):
        s.add_argument("--" + n, type=Path, required=True)
    s.add_argument("--batch", type=int, default=16)
    f = sub.add_parser("fit")
    for n in ("train", "calibration", "evaluation", "output"):
        f.add_argument("--" + n, type=Path, required=True)
    f.add_argument("--calibration-objective", choices=("proposal-f05", "sentence-f05"), default="proposal-f05",
                   help="Sentence F0.5 counts all missed errors, including those without a proposal")
    args = p.parse_args()
    {"banks": make_banks, "score": score, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
