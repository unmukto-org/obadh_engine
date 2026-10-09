"""Compare complete-word decoding before/after export on a fixed development bank.

Timing includes Python tokenization, beam search, cache copies and runtime calls.
It is a desktop reference-host measurement, not native keyboard latency or memory.
The frozen 16-beam/12-piece policy is shared by every backend.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from tools.corpus.provenance import sha256_file, digest_json, write_json


def main():
    import torch
    from tokenizers import Tokenizer
    from tools.autosuggest.subword_lm import ModelConfig, make_model
    from tools.autosuggest.decode_subword_words import WordDecoder
    from tools.autosuggest.cached_word_scorer import CachedWordScorer

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data", "export", "bank", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--bank-sha256", required=True)
    p.add_argument("--backend", choices=("onnx", "coreml-int8"), required=True)
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.limit <= 4096:
        p.error("new output directory and limit 1..4096 required")
    if any((d / ".building").exists() for d in (args.export, args.data)):
        raise ValueError("incomplete input")
    receipt = json.loads((args.data / "manifest.json").read_text())
    export = json.loads((args.export / "report.json").read_text())
    checkpoint_hash = sha256_file(args.checkpoint)
    if (
        export["kind"] != "obadh-subword-explicit-cache-research-export"
        or export["checkpoint_sha256"] != checkpoint_hash
        or export["data_id"] != digest_json(receipt)
        or receipt["tokenizer_sha256"] != sha256_file(args.data / "tokenizer.json")
        or sha256_file(args.bank) != args.bank_sha256
    ):
        raise ValueError("evaluation identity mismatch")
    for relative, expected in export["files"].items():
        path = args.export / relative
        if not path.resolve().is_relative_to(args.export.resolve()):
            raise ValueError("export path escapes package")
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError("export file integrity failure")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["contract"]["data_id"] != digest_json(receipt) or state["config"] != export["config"]:
        raise ValueError("checkpoint data/config mismatch")
    rows = [json.loads(line) for line in args.bank.open(encoding="utf-8")]
    if len({r["id"] for r in rows}) != len(rows) or not rows:
        raise ValueError("empty or duplicate evaluation bank")
    rows = rows[:args.limit]
    torch.set_num_threads(1)
    model = make_model(ModelConfig(**state["config"])).eval()
    model.load_state_dict(state["state_dict"])
    del state
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    reference = WordDecoder(tokenizer, model, "cpu", beam=16, max_pieces=12)
    sessions = {}
    if args.backend == "onnx":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        for stage in ("prefill", "step"):
            sessions[stage] = ort.InferenceSession(
                str(args.export / (stage + ".onnx")), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        def call(stage, features):
            return dict(zip(("logits", "next_keys", "next_values"), sessions[stage].run(None, features)))
        dtype = np.float32
    else:
        import coremltools as ct
        for stage in ("prefill", "step"):
            sessions[stage] = ct.models.MLModel(
                str(args.export / (stage + ".int8.mlpackage")),
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )
        def call(stage, features):
            return sessions[stage].predict(features)
        # Core ML's Python bridge promotes both FP16 cache inputs and outputs
        # to FP32. Keep that bridge representation here; the compiled graph
        # still has the verified FP16 external-cache contract. This is not a
        # measurement of native Swift FP16 host-buffer memory.
        dtype = np.float32
    host = CachedWordScorer(
        model.config, lambda f: call("prefill", f), lambda f: call("step", f), cache_dtype=dtype,
    )
    cached = WordDecoder(tokenizer, model, "cpu", beam=16, max_pieces=12, scorer=host)
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    counts = dict(top1_agreement=0, ordered_top3_agreement=0)
    hits = {name: {k: 0 for k in (1, 3, 5)} for name in ("reference", args.backend)}
    timings = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            context = row.get("context") or " ".join(row["context_tokens"])
            expected, _ = reference.predict(context)
            started = time.perf_counter_ns()
            try:
                actual, _ = cached.predict(context)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                model_calls = host.model_calls
            finally:
                host.close()
            words = {"reference": [w for w, _ in expected], args.backend: [w for w, _ in actual]}
            for name, predictions in words.items():
                for k in hits[name]:
                    hits[name][k] += row["target"] in predictions[:k]
            counts["top1_agreement"] += words["reference"][:1] == words[args.backend][:1]
            counts["ordered_top3_agreement"] += words["reference"][:3] == words[args.backend][:3]
            timings.append(elapsed)
            handle.write(json.dumps(dict(
                id=row["id"], target=row["target"], predictions=words,
                decode_ms=elapsed, model_calls=model_calls,
            ), ensure_ascii=False) + "\n")
            if (index + 1) % 10 == 0:
                handle.flush()
                print(json.dumps(dict(examples=index + 1, backend=args.backend)), flush=True)
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed during evaluation")
    report = dict(
        scope=__doc__, backend=args.backend, examples=len(rows),
        checkpoint_sha256=checkpoint_hash, data_id=digest_json(receipt),
        export_report_sha256=sha256_file(args.export / "report.json"),
        bank_sha256=args.bank_sha256, selection="first rows in fixed bank order",
        selected_ids_sha256=digest_json([r["id"] for r in rows]),
        policy=dict(beam=16, max_pieces=12),
        host_cache_dtype=np.dtype(dtype).name,
        agreement={k: v / len(rows) for k, v in counts.items()},
        top_k={name: {str(k): v / len(rows) for k, v in values.items()} for name, values in hits.items()},
        first_decode_ms=timings[0],
        warm_decode_ms={str(q): float(np.percentile(timings[1:], q)) for q in (50, 95, 99)} if len(timings) > 1 else {},
        maximum_live_logical_cache_bytes=2 * (16 + 1) * export["coreml_cache_bytes_per_sequence"] * (dtype().itemsize // 2),
        memory_scope="logical buffer upper bound only; excludes models, runtimes, allocator retention and Python",
        implementation_sha256={name: sha256_file(Path(__file__).with_name(name)) for name in (
            "evaluate_cached_words.py", "cached_word_scorer.py", "decode_subword_words.py",
        )},
    )
    write_json(args.output / "report.json", report)
    (args.output / ".building").unlink()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
