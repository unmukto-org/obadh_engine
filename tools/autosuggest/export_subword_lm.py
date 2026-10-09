"""Export a verified subword checkpoint as a bounded prefix-scoring graph.

The graph projects only the final valid hidden state, avoiding a full vocabulary
projection at every context position. This is a portable deployment primitive;
word decoding, prefix constraints and keyboard confidence policy are separate.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from tools.autosuggest.subword_lm import make_model, ModelConfig
from tools.autosuggest.export_fixtures import ValidationSequences
from tools.corpus.provenance import sha256_file, digest_json, write_json


class PrefixScorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, last_index):
        x = self.model.embedding(tokens.long())
        length = tokens.shape[1]
        for block in self.model.blocks:
            x = block(x, self.model.cos[:, :, :length], self.model.sin[:, :, :length])
        hidden = self.model.norm(x)[:, last_index[0].long(), :]
        return hidden @ self.model.embedding.weight.T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cases", type=int, default=256)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite export")
    if not 1 <= args.cases <= 4096:
        p.error("cases must be 1..4096")
    torch.set_num_threads(1)
    receipt = json.loads((args.data / "manifest.json").read_text())
    checkpoint_hash = sha256_file(args.checkpoint)
    if receipt["tokenizer_sha256"] != sha256_file(
        args.data / "tokenizer.json"
    ) or receipt["partitions"]["validation"]["sha256"] != sha256_file(
        args.data / "validation.u16"
    ):
        raise ValueError("data receipt mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["contract"]["data_id"] != digest_json(receipt):
        raise ValueError("checkpoint provenance mismatch")
    config = ModelConfig(**checkpoint["config"])
    model = make_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    wrapper = PrefixScorer(model).eval()
    sequences = ValidationSequences(args.data, receipt, config.sequence_length)
    rng = np.random.default_rng(17)
    cases = []
    for _ in range(args.cases):
        sequence, length = sequences.sample(
            rng, maximum_prefix=config.sequence_length, continuation=1
        )
        tokens = np.zeros((1, config.sequence_length), dtype=np.int32)
        tokens[0, :length] = sequence[:length]
        cases.append(
            (
                tokens,
                np.array([length - 1], dtype=np.int32),
                int(sequence[length]),
            )
        )
    # Prove selective projection preserves the original causal model exactly.
    with torch.no_grad():
        for tokens, last, _ in cases[:8]:
            a = wrapper(torch.from_numpy(tokens), torch.from_numpy(last))
            b = model(torch.from_numpy(tokens).long())[:, int(last[0]), :]
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()
    inputs = (torch.from_numpy(cases[0][0]), torch.from_numpy(cases[0][1]))
    onnx_path = args.output / "prefix.onnx"
    torch.onnx.export(
        wrapper,
        inputs,
        onnx_path,
        input_names=["tokens", "last_index"],
        output_names=["logits"],
        opset_version=17,
    )
    import onnx, onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx.checker.check_model(onnx.load(onnx_path))
    quantize_dynamic(
        str(onnx_path),
        str(args.output / "prefix.int8.onnx"),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        linear_quantize_weights,
    )

    traced = torch.jit.trace(wrapper, inputs)
    coreml = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(
                name="tokens", shape=(1, config.sequence_length), dtype=np.int32
            ),
            ct.TensorType(name="last_index", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
    )
    coreml.save(str(args.output / "prefix.mlpackage"))
    compressed = linear_quantize_weights(
        coreml,
        OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype=np.int8
            )
        ),
    )
    compressed.save(str(args.output / "prefix.int8.mlpackage"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sessions = {
        name: ort.InferenceSession(
            str(args.output / path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        for name, path in [("onnx", "prefix.onnx"), ("onnx_int8", "prefix.int8.onnx")]
    }
    cores = {
        name: ct.models.MLModel(
            str(args.output / path), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        for name, path in [
            ("coreml", "prefix.mlpackage"),
            ("coreml_int8", "prefix.int8.mlpackage"),
        ]
    }
    statistics = {
        name: {
            "max_abs_logit_error": 0.0,
            "top1_agreement": 0,
            "top1_hits": 0,
            "top5_hits": 0,
            "warm_us": [],
        }
        for name in [*sessions, *cores]
    }
    reference_hits = {1: 0, 5: 0}
    with torch.no_grad():
        for index, (tokens, last, target) in enumerate(cases):
            expected = (
                wrapper(torch.from_numpy(tokens), torch.from_numpy(last))
                .numpy()
                .reshape(-1)
            )
            for k in reference_hits:
                reference_hits[k] += target in np.argsort(-expected, kind="stable")[:k]
            for name in statistics:
                features = {"tokens": tokens, "last_index": last}
                start = time.perf_counter_ns()
                actual = (
                    sessions[name].run(["logits"], features)[0]
                    if name in sessions
                    else cores[name].predict(features)["logits"]
                ).reshape(-1)
                elapsed = (time.perf_counter_ns() - start) / 1000
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError("invalid exported output")
                stat = statistics[name]
                stat["max_abs_logit_error"] = max(
                    stat["max_abs_logit_error"],
                    float(np.max(np.abs(actual - expected))),
                )
                stat["top1_agreement"] += int(np.argmax(actual) == np.argmax(expected))
                for k in (1, 5):
                    stat[f"top{k}_hits"] += int(
                        target in np.argsort(-actual, kind="stable")[:k]
                    )
                if index > 0:
                    stat["warm_us"].append(elapsed)
    for stat in statistics.values():
        timings = stat.pop("warm_us")
        stat["warm_model_call_mean_us"] = float(np.mean(timings)) if timings else None
        stat["warm_model_call_p95_us"] = (
            float(np.percentile(timings, 95)) if timings else None
        )
        for key in ("top1_agreement", "top1_hits", "top5_hits"):
            stat[key] /= len(cases)
    shutil.copy2(args.data / "tokenizer.json", args.output / "tokenizer.json")
    write_json(
        args.output / "fixtures.json",
        {
            "version": 1,
            "kind": "subword-prefix",
            "sequence_length": config.sequence_length,
            "vocab_size": config.vocab_size,
            "tokens": [x.tolist() for x, _, _ in cases],
            "last_indices": [x.tolist() for _, x, _ in cases],
            "targets": [y for _, _, y in cases],
            "checkpoint_sha256": checkpoint_hash,
            "export_fixture_helper_sha256": sha256_file(Path(__file__).with_name("export_fixtures.py")),
        },
    )
    if checkpoint_hash != sha256_file(args.checkpoint):
        raise ValueError("checkpoint changed during export")
    files = {
        str(p.relative_to(args.output)): {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(args.output.rglob("*"))
        if p.is_file() and p.name != ".building"
    }
    report = {
        "kind": "obadh-subword-prefix-research-export",
        "version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "step": checkpoint["step"],
        "config": checkpoint["config"],
        "data_id": digest_json(receipt),
        "files": files,
        "verification": statistics,
        "pytorch_next_token_accuracy": {
            k: v / len(cases) for k, v in reference_hits.items()
        },
        "cases": len(cases),
        "runtime_contract": {
            "tokens": [1, config.sequence_length],
            "last_index": [1],
            "dtype": "int32",
            "right_padding_id": 0,
            "empty_context": [1],
            "outputs": {"logits": [1, config.vocab_size]},
        },
        "scope": "validation next-subword parity/accuracy and Mac CPU model calls; not word accuracy, full keyboard timing or iPhone memory",
    }
    write_json(args.output / "report.json", report)
    (args.output / ".building").unlink()
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
