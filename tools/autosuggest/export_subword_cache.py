"""Export and verify explicit-cache prefill/decode graphs for a frozen model.

This measures the cache optimization without retraining or changing decoder
search. Core ML uses FP16 external caches and linear 8-bit weights. Both graphs
are separate research packages; their combined resident cost needs measuring.
"""

from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path
import numpy as np
import torch
from tools.autosuggest.subword_lm import make_model, ModelConfig
from tools.autosuggest.export_fixtures import ValidationSequences
from tools.autosuggest.subword_cache import PrefillScorer, CachedStepScorer, cache_shape
from tools.corpus.provenance import sha256_file, digest_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cases", type=int, default=32)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.cases <= 256:
        p.error("new output and 1..256 cases required")
    torch.set_num_threads(1)
    receipt = json.loads((args.data / "manifest.json").read_text())
    checkpoint_hash = sha256_file(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        state["contract"]["data_id"] != digest_json(receipt)
        or receipt["partitions"]["validation"]["sha256"] != sha256_file(args.data / "validation.u16")
        or receipt["tokenizer_sha256"] != sha256_file(args.data / "tokenizer.json")
    ):
        raise ValueError("data provenance mismatch")
    config = ModelConfig(**state["config"])
    model = make_model(config)
    model.load_state_dict(state["state_dict"])
    model.eval()
    prefill = PrefillScorer(model).eval()
    step = CachedStepScorer(model).eval()
    tokens = torch.zeros((1, config.sequence_length), dtype=torch.int32)
    tokens[0, 0] = 1
    position = torch.tensor([0], dtype=torch.int32)
    empty = torch.zeros(cache_shape(config))
    examples = {
        "prefill": (tokens, position),
        "step": (tokens[:, :1], position, empty, empty),
    }
    wrappers = {"prefill": prefill, "step": step}
    inputs = {
        "prefill": ["tokens", "last_index"],
        "step": ["token", "position", "key_cache", "value_cache"],
    }
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    import coremltools as ct
    import onnx, onnxruntime as ort
    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        linear_quantize_weights,
    )

    sessions = {}
    cores = {}
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    for name, wrapper in wrappers.items():
        path = args.output / f"{name}.onnx"
        torch.onnx.export(
            wrapper,
            examples[name],
            path,
            input_names=inputs[name],
            output_names=["logits", "next_keys", "next_values"],
            opset_version=17,
        )
        onnx.checker.check_model(onnx.load(path))
        sessions[name] = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        traced = torch.jit.trace(wrapper, examples[name])
        core = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            inputs=[
                ct.TensorType(
                    name=key,
                    shape=tuple(value.shape),
                    dtype=np.int32 if value.dtype == torch.int32 else np.float16,
                )
                for key, value in zip(inputs[name], examples[name])
            ],
            outputs=[
                ct.TensorType(name="logits", dtype=np.float32),
                ct.TensorType(name="next_keys", dtype=np.float16),
                ct.TensorType(name="next_values", dtype=np.float16),
            ],
        )
        core.save(str(args.output / f"{name}.mlpackage"))
        core = linear_quantize_weights(
            core,
            OptimizationConfig(
                global_config=OpLinearQuantizerConfig(
                    mode="linear_symmetric", dtype=np.int8
                )
            ),
        )
        core.save(str(args.output / f"{name}.int8.mlpackage"))
        cores[name] = ct.models.MLModel(
            str(args.output / f"{name}.int8.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
    sequences = ValidationSequences(args.data, receipt, config.sequence_length)
    rng = np.random.default_rng(17)
    stats = {
        name: {
            "max_abs_logit_error": 0.0,
            "top1_agreement": 0,
            "cases": 0,
            "prefill_us": [],
            "step_us": [],
        }
        for name in ("onnx", "coreml_int8")
    }

    def compare(name, actual, expected):
        if not np.isfinite(actual).all():
            raise ValueError("nonfinite cached output")
        s = stats[name]
        s["max_abs_logit_error"] = max(
            s["max_abs_logit_error"], float(np.abs(actual - expected).max())
        )
        s["top1_agreement"] += int(actual.argmax() == expected.argmax())
        s["cases"] += 1

    with torch.no_grad():
        for case in range(args.cases):
            sequence, length = sequences.sample(
                rng, maximum_prefix=config.sequence_length - 8, continuation=8
            )
            padded = np.zeros((1, config.sequence_length), dtype=np.int32)
            padded[0, :length] = sequence[:length]
            expected = model(torch.from_numpy(sequence[None]).long()).numpy()
            for name in stats:
                features = {
                    "tokens": padded,
                    "last_index": np.array([length - 1], dtype=np.int32),
                }
                started = time.perf_counter_ns()
                result = (
                    sessions["prefill"].run(None, features)
                    if name == "onnx"
                    else cores["prefill"].predict(features)
                )
                stats[name]["prefill_us"].append(
                    (time.perf_counter_ns() - started) / 1000
                )
                logits, keys, values = (
                    result
                    if name == "onnx"
                    else [result[k] for k in ("logits", "next_keys", "next_values")]
                )
                compare(name, logits, expected[:, length - 1])
                for index in range(length, length + 8):
                    features = {
                        "token": sequence[index : index + 1].reshape(1, 1),
                        "position": np.array([index], dtype=np.int32),
                        "key_cache": keys.astype(
                            np.float32 if name == "onnx" else np.float16
                        ),
                        "value_cache": values.astype(
                            np.float32 if name == "onnx" else np.float16
                        ),
                    }
                    started = time.perf_counter_ns()
                    result = (
                        sessions["step"].run(None, features)
                        if name == "onnx"
                        else cores["step"].predict(features)
                    )
                    stats[name]["step_us"].append(
                        (time.perf_counter_ns() - started) / 1000
                    )
                    logits, keys, values = (
                        result
                        if name == "onnx"
                        else [result[k] for k in ("logits", "next_keys", "next_values")]
                    )
                    compare(name, logits, expected[:, index])
    for s in stats.values():
        s["top1_agreement"] /= s["cases"]
        for stage in ("prefill", "step"):
            values = s.pop(stage + "_us")[1:]
            s[stage + "_warm_p50_us"] = float(np.percentile(values, 50))
            s[stage + "_warm_p95_us"] = float(np.percentile(values, 95))
    if checkpoint_hash != sha256_file(args.checkpoint):
        raise ValueError("checkpoint changed")
    shutil.copy2(args.data / "tokenizer.json", args.output / "tokenizer.json")
    files = {
        str(f.relative_to(args.output)): {
            "bytes": f.stat().st_size,
            "sha256": sha256_file(f),
        }
        for f in sorted(args.output.rglob("*"))
        if f.is_file() and f.name != ".building"
    }
    report = {
        "version": 1,
        "kind": "obadh-subword-explicit-cache-research-export",
        "checkpoint_sha256": checkpoint_hash,
        "config": state["config"],
        "data_id": digest_json(receipt),
        "implementation_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("export_subword_cache.py", "export_fixtures.py", "subword_cache.py")
        },
        "verification": stats,
        "files": files,
        "cache_shape": cache_shape(config),
        "coreml_cache_bytes_per_sequence": int(np.prod(cache_shape(config)) * 2 * 2),
        "contract": {
            "position": "absolute next-token index, 0 <= position < capacity; host must validate; never wrap",
            "ownership": "per exact prefix and model; discard on cursor edits, reset, or model replacement",
            "capacity_policy": "re-prefill a truncated context when full; mask excludes all future slots",
            "beam_policy": "each beam needs independent logical caches; shared-prefix/copy-on-write is a host optimization",
        },
        "scope": "32 validation prefixes by default, eight cached steps each; CPU Mac model-call timings, separate packages; not a complete decoder or iPhone benchmark",
    }
    write_json(args.output / "report.json", report)
    (args.output / ".building").unlink()
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
