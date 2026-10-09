"""Compress a candidate scorer and measure ranking changes on export fixtures.

Weight-file reduction is not a claim about resident memory or iPhone latency.
This tool produces research artifacts; promotion requires full quality and device
measurements. Inputs and outputs are immutable and bound by SHA-256 inventories.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.corpus.provenance import digest_json, sha256_file, write_json


def inventory(root):
    return {
        p.relative_to(root).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument(
        "--mode", choices=("palettize4", "palettize8", "linear8"), required=True
    )
    args = p.parse_args()
    if args.output.exists() or args.report.exists():
        p.error("refusing to replace compressed artifacts")
    import numpy as np
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpPalettizerConfig,
        OpLinearQuantizerConfig,
        palettize_weights,
        linear_quantize_weights,
    )

    fixture_hash = sha256_file(args.fixtures)
    fixtures = json.loads(args.fixtures.read_text())
    if (
        fixtures["version"] != 1
        or len(fixtures["contexts"]) != len(fixtures["candidates"])
        or not fixtures["contexts"]
    ):
        raise ValueError("invalid fixture dimensions")
    targets = fixtures.get("targets")
    if targets is not None and len(targets) != len(fixtures["contexts"]):
        raise ValueError("target count mismatch")
    source_inventory = inventory(args.model)
    original = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    if args.mode.startswith("palettize"):
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                mode="kmeans", nbits=int(args.mode[9:]), num_kmeans_workers=1
            )
        )
        compressed = palettize_weights(original, config)
    else:
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype=np.int8
            )
        )
        compressed = linear_quantize_weights(original, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compressed.save(str(args.output))
    # Reload serialized graph; validating an in-memory transformation is insufficient.
    compressed = ct.models.MLModel(
        str(args.output), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    agree = {k: 0 for k in (1, 3, 5)}
    hits = {m: {k: 0 for k in agree} for m in ("original", "compressed")}
    max_error = 0.0
    error_sum = 0.0
    score_count = 0
    for index, (ctx, ids) in enumerate(
        zip(fixtures["contexts"], fixtures["candidates"])
    ):
        if (
            not ctx
            or not ids
            or any(
                type(v) != int or not 0 <= v < fixtures["vocabularySize"]
                for v in ctx + ids
            )
        ):
            raise ValueError("invalid fixture IDs")
        features = {
            "contexts": np.asarray([ctx], dtype=np.int32),
            "candidate_ids": np.asarray([ids], dtype=np.int32),
        }
        outputs = [
            np.asarray(m.predict(features)["scores"]).reshape(-1)
            for m in (original, compressed)
        ]
        if any(len(s) != len(ids) or not np.isfinite(s).all() for s in outputs):
            raise ValueError("invalid model output")
        error = np.abs(outputs[0] - outputs[1])
        max_error = max(max_error, float(error.max()))
        error_sum += float(error.sum())
        score_count += len(ids)
        ranks = [np.argsort(-s, kind="stable").tolist() for s in outputs]
        for k in agree:
            agree[k] += ranks[0][:k] == ranks[1][:k]
            if targets is not None:
                for name, rank in zip(hits, ranks):
                    hits[name][k] += targets[index] in [ids[i] for i in rank[:k]]
    if source_inventory != inventory(args.model):
        raise ValueError("source model changed during compression")
    if fixture_hash != sha256_file(args.fixtures):
        raise ValueError("fixtures changed during compression")
    output_inventory = inventory(args.output)
    n = len(fixtures["contexts"])
    result = {
        "version": 1,
        "coremltools": ct.__version__,
        "mode": args.mode,
        "compute_units": "cpuOnly",
        "source": {
            "inventory": source_inventory,
            "sha256": digest_json(source_inventory),
        },
        "compressed": {
            "inventory": output_inventory,
            "sha256": digest_json(output_inventory),
        },
        "source_bytes": sum(x["bytes"] for x in source_inventory.values()),
        "compressed_bytes": sum(x["bytes"] for x in output_inventory.values()),
        "fixture_sha256": fixture_hash,
        "cases": n,
        "max_abs_logit_error": max_error,
        "mean_abs_logit_error": error_sum / score_count,
        "exact_ordered_topk_agreement": {k: v / n for k, v in agree.items()},
        "topk_accuracy": {m: {k: v / n for k, v in c.items()} for m, c in hits.items()}
        if targets is not None
        else None,
        "scope": fixtures.get("targetScope", "ranking parity only; no labels provided"),
        "device_memory_or_latency_claim": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
