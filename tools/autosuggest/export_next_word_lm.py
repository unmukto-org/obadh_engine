#!/usr/bin/env python3
"""Export and benchmark Obadh's neural autosuggest models.

The shipped autosuggest runtime is still the static n-gram artifact. This tool
is a production gate for the optional neural layer: it exports either a bounded
candidate-only scorer or a full-vocabulary top-k generator, verifies the graph,
optionally applies ONNX Runtime dynamic quantization, and measures fixed-batch
keyboard-time latency on real corpus contexts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from functools import wraps
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from collections import Counter
from dataclasses import dataclass

from tools.autosuggest.common import PAD_ID, UNK_ID
from tools.corpus.provenance import (
    evaluation_provenance, load_partition, neural_training_provenance,
    sha256_file, verify_checkpoint_provenance,
)
from tools.autosuggest.eval_ngram_lm import NgramLm, model_recent_context
from tools.autosuggest.train_next_word_lm import (
    NextWordLm,
    collect_candidate_pool,
    collect_examples,
    parameter_count,
    score_candidate_pool,
)


REPORT_CUTOFFS = (1, 3, 5, 10)
DEFAULT_RANK_PENALTIES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
LEARNED_UNION_FEATURE_NAMES = (
    "model_score",
    "is_static",
    "negative_static_rank",
    "is_generated",
    "negative_generated_rank",
    "is_static_and_generated",
)
SCORED_UNION_POLICY_FIELDS = (
    "locked_static_prefix",
    "static_bonus",
    "static_rank_penalty",
    "generated_penalty",
    "overlap_bonus",
    "generated_rank_penalty",
    "static_log_count_scale",
    "static_source_bonus",
)
STATIC_SOURCE_ORDER = {
    "unigram": 0,
    "bigram": 1,
    "trigram": 2,
    "fourgram": 3,
}


@dataclass
class ExportInputSet:
    contexts: np.ndarray
    candidate_ids: np.ndarray
    candidate_rows: list[list[int]]
    candidate_counts: np.ndarray
    candidate_source_order: np.ndarray
    labels: np.ndarray
    source_ids: np.ndarray
    source_names: tuple[str, ...]
    total_targets: int
    eligible_targets: int

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


class CandidateOnlyScorer(nn.Module):
    def __init__(self, model: NextWordLm) -> None:
        super().__init__()
        self.model = model

    def forward(self, contexts: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        return score_candidate_pool(self.model, contexts, candidate_ids)


class FullVocabTopKGenerator(nn.Module):
    def __init__(self, model: NextWordLm, top_k: int) -> None:
        super().__init__()
        self.model = model
        self.top_k = top_k

    def forward(self, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(contexts)
        return torch.topk(logits, k=self.top_k, dim=1)


class FullVocabTopKAndCandidateScorer(nn.Module):
    def __init__(self, model: NextWordLm, top_k: int) -> None:
        super().__init__()
        self.model = model
        self.top_k = top_k

    def forward(
        self,
        contexts: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.model.encode_context(contexts)
        logits = hidden @ self.model.token_embedding.weight.T + self.model.output_bias
        topk_scores, token_ids = torch.topk(logits, k=self.top_k, dim=1)
        candidate_scores = torch.gather(logits, dim=1, index=candidate_ids.long())
        return topk_scores, token_ids, candidate_scores


def load_model(checkpoint_path: Path, *, provenance: dict | None = None,
               retrieval_model: Path | None = None) -> tuple[NextWordLm, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if provenance is not None:
        if retrieval_model is None:
            raise ValueError("checkpoint verification requires a retrieval artifact")
        verify_checkpoint_provenance(checkpoint, provenance, retrieval_model)
    config = checkpoint["config"]
    model = NextWordLm(
        vocab_size=int(config["vocab_size"]),
        context_len=int(config["context_window"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        architecture=str(config["architecture"]),
        dropout=0.0,
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        transformer_padding_mode=config.get("transformer_padding_mode", "unmasked-v0"),
        transformer_initialization=config.get("transformer_initialization", "legacy-v0"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def portable_attention_export(function):
    """Scope tracing to portable primitives and restore the caller's backend."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        enabled = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            return function(*args, **kwargs)
        finally:
            torch.backends.mha.set_fastpath_enabled(enabled)
    return wrapped


@portable_attention_export
def export_onnx(
    model: NextWordLm,
    output_path: Path,
    context_window: int,
    pool_k: int,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scorer = CandidateOnlyScorer(model)
    contexts = torch.ones((1, context_window), dtype=torch.long)
    candidate_ids = torch.full((1, pool_k), 3, dtype=torch.long)
    torch.onnx.export(
        scorer,
        (contexts, candidate_ids),
        output_path,
        input_names=["contexts", "candidate_ids"],
        output_names=["scores"],
        opset_version=opset,
        dynamic_axes=None,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


@portable_attention_export
def export_topk_onnx(
    model: NextWordLm,
    output_path: Path,
    context_window: int,
    top_k: int,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = FullVocabTopKGenerator(model, top_k)
    contexts = torch.ones((1, context_window), dtype=torch.long)
    torch.onnx.export(
        generator,
        (contexts,),
        output_path,
        input_names=["contexts"],
        output_names=["scores", "token_ids"],
        opset_version=opset,
        dynamic_axes=None,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


@portable_attention_export
def export_combined_onnx(
    model: NextWordLm,
    output_path: Path,
    context_window: int,
    pool_k: int,
    top_k: int,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = FullVocabTopKAndCandidateScorer(model, top_k)
    contexts = torch.ones((1, context_window), dtype=torch.long)
    candidate_ids = torch.full((1, pool_k), 3, dtype=torch.long)
    torch.onnx.export(
        generator,
        (contexts, candidate_ids),
        output_path,
        input_names=["contexts", "candidate_ids"],
        output_names=["topk_scores", "token_ids", "candidate_scores"],
        opset_version=opset,
        dynamic_axes=None,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


@portable_attention_export
def export_coreml(
    model: NextWordLm,
    output_path: Path,
    context_window: int,
    pool_k: int,
    minimum_deployment_target: str,
    precision: str,
) -> None:
    import coremltools as ct

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    scorer = CandidateOnlyScorer(model).eval()
    contexts = torch.ones((1, context_window), dtype=torch.int32)
    candidate_ids = torch.full((1, pool_k), 3, dtype=torch.int32)
    traced = torch.jit.trace(scorer, (contexts, candidate_ids))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="contexts", shape=contexts.shape, dtype=np.int32),
            ct.TensorType(name="candidate_ids", shape=candidate_ids.shape, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="scores")],
        convert_to="mlprogram",
        minimum_deployment_target=coreml_target(ct, minimum_deployment_target),
        compute_precision=coreml_precision(ct, precision),
    )
    mlmodel.save(str(output_path))


@portable_attention_export
def export_topk_coreml(
    model: NextWordLm,
    output_path: Path,
    context_window: int,
    top_k: int,
    minimum_deployment_target: str,
    precision: str,
) -> None:
    import coremltools as ct

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    generator = FullVocabTopKGenerator(model, top_k).eval()
    contexts = torch.ones((1, context_window), dtype=torch.int32)
    traced = torch.jit.trace(generator, (contexts,))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="contexts", shape=contexts.shape, dtype=np.int32)],
        outputs=[
            ct.TensorType(name="scores"),
            ct.TensorType(name="token_ids"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=coreml_target(ct, minimum_deployment_target),
        compute_precision=coreml_precision(ct, precision),
    )
    mlmodel.save(str(output_path))


@portable_attention_export
def export_combined_coreml(
    model: NextWordLm,
    output_path: Path,
    context_window: int,
    pool_k: int,
    top_k: int,
    minimum_deployment_target: str,
    precision: str,
) -> None:
    import coremltools as ct

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    generator = FullVocabTopKAndCandidateScorer(model, top_k).eval()
    contexts = torch.ones((1, context_window), dtype=torch.int32)
    candidate_ids = torch.full((1, pool_k), 3, dtype=torch.int32)
    traced = torch.jit.trace(generator, (contexts, candidate_ids))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="contexts", shape=contexts.shape, dtype=np.int32),
            ct.TensorType(name="candidate_ids", shape=candidate_ids.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="topk_scores"),
            ct.TensorType(name="token_ids"),
            ct.TensorType(name="candidate_scores"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=coreml_target(ct, minimum_deployment_target),
        compute_precision=coreml_precision(ct, precision),
    )
    mlmodel.save(str(output_path))


def coreml_target(ct, target: str):
    targets = {
        "ios16": ct.target.iOS16,
        "ios17": ct.target.iOS17,
        "ios18": ct.target.iOS18,
    }
    try:
        return targets[target.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported Core ML deployment target: {target}") from error


def coreml_precision(ct, precision: str):
    if precision == "float16":
        return ct.precision.FLOAT16
    if precision == "float32":
        return ct.precision.FLOAT32
    raise ValueError(f"unsupported Core ML precision: {precision}")


def quantize_onnx(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


def collect_real_inputs(
    lm: NgramLm,
    corpus_dir: Path,
    context_window: int,
    pool_k: int,
    skip_sentences_per_source: int,
    max_examples_per_source: int,
) -> ExportInputSet:
    examples = collect_examples(
        lm,
        corpus_dir,
        sources=None,
        skip_sentences_per_source=skip_sentences_per_source,
        max_sentences_per_source=None,
        max_examples_per_source=max_examples_per_source,
        context_window=context_window,
        log_every_targets=0,
    )
    candidate_pool = collect_candidate_pool(lm, examples, pool_k)
    candidate_counts, candidate_source_order = collect_static_candidate_features(
        lm,
        examples,
        candidate_pool.rows,
        pool_k,
    )
    return ExportInputSet(
        contexts=examples.contexts.numpy().astype(np.int64),
        candidate_ids=candidate_pool.ids.numpy().astype(np.int64),
        candidate_rows=candidate_pool.rows,
        candidate_counts=candidate_counts,
        candidate_source_order=candidate_source_order,
        labels=examples.labels.numpy().astype(np.int64),
        source_ids=examples.source_ids.numpy().astype(np.int16),
        source_names=examples.source_names,
        total_targets=examples.total_targets,
        eligible_targets=examples.eligible_targets,
    )


def collect_static_candidate_features(
    lm: NgramLm,
    examples,
    candidate_rows: list[list[int]],
    pool_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((len(candidate_rows), pool_k), dtype=np.float32)
    source_order = np.zeros((len(candidate_rows), pool_k), dtype=np.float32)
    max_context = lm.max_context_order
    for row_index, (context_row, expected_row) in enumerate(
        zip(examples.contexts.tolist(), candidate_rows)
    ):
        context_ids = [token_id for token_id in context_row if token_id != PAD_ID]
        features = suggest_static_candidate_features(
            lm,
            context_ids[-max_context:],
            pool_k,
        )
        feature_ids = [token_id for token_id, _, _ in features]
        if feature_ids != expected_row[:pool_k]:
            raise RuntimeError("static candidate feature extraction diverged from candidate pool")
        for column_index, (_, count, source) in enumerate(features[:pool_k]):
            counts[row_index, column_index] = float(count)
            source_order[row_index, column_index] = float(source)
    return counts, source_order


def suggest_static_candidate_features(
    lm: NgramLm,
    context_ids: list[int],
    limit: int,
) -> list[tuple[int, int, int]]:
    recent = model_recent_context(context_ids, max_context=lm.max_context_order)
    if lm.score_mode != "backoff":
        return [
            (token_id, 0, 0)
            for token_id in lm.suggest_ids(context_ids, limit)
        ]

    output: list[tuple[int, int, int]] = []
    seen: set[int] = set()

    def append_row(row: tuple[int, int] | None, source_name: str) -> None:
        if row is None or len(output) >= limit:
            return
        start, length = row
        source = STATIC_SOURCE_ORDER[source_name]
        for index in range(start, start + length):
            if len(output) >= limit:
                break
            offset = lm.candidates_offset + index * lm.candidate_record_len
            candidate = lm._candidate_at(offset)
            if candidate.token_id > UNK_ID and candidate.token_id not in seen:
                seen.add(candidate.token_id)
                output.append((candidate.token_id, candidate.count, source))

    if len(recent) == 3:
        append_row(lm._find_fourgram_row(recent[0], recent[1], recent[2]), "fourgram")
    if len(recent) >= 2:
        append_row(lm._find_trigram_row(recent[-2], recent[-1]), "trigram")
    if recent:
        append_row(lm._find_bigram_row(recent[-1]), "bigram")
    if len(output) < limit:
        source = STATIC_SOURCE_ORDER["unigram"]
        for index in range(lm.unigram_count):
            if len(output) >= limit:
                break
            offset = lm.unigrams_offset + index * lm.candidate_record_len
            candidate = lm._candidate_at(offset)
            if candidate.token_id > UNK_ID and candidate.token_id not in seen:
                seen.add(candidate.token_id)
                output.append((candidate.token_id, candidate.count, source))
    return output


def compare_with_pytorch(
    model: NextWordLm,
    session: ort.InferenceSession,
    inputs: ExportInputSet,
    sample_count: int,
) -> dict:
    sample_count = min(sample_count, inputs.size)
    if sample_count <= 0:
        return {}
    with torch.no_grad():
        torch_scores = score_candidate_pool(
            model,
            torch.from_numpy(inputs.contexts[:sample_count]),
            torch.from_numpy(inputs.candidate_ids[:sample_count]),
        ).numpy()
    onnx_scores = np.concatenate(
        [
            session.run(
                ["scores"],
                {
                    "contexts": inputs.contexts[index : index + 1],
                    "candidate_ids": inputs.candidate_ids[index : index + 1],
                },
            )[0]
            for index in range(sample_count)
        ],
        axis=0,
    )
    absolute = np.abs(torch_scores - onnx_scores)
    return {
        "sample_count": sample_count,
        "max_abs_diff": float(absolute.max(initial=0.0)),
        "mean_abs_diff": float(absolute.mean() if absolute.size else 0.0),
    }


def compare_topk_onnx_with_pytorch(
    model: NextWordLm,
    session: ort.InferenceSession,
    inputs: ExportInputSet,
    sample_count: int,
    top_k: int,
) -> dict:
    sample_count = min(sample_count, inputs.size)
    if sample_count <= 0:
        return {}
    with torch.no_grad():
        torch_scores, torch_ids = torch.topk(
            model(torch.from_numpy(inputs.contexts[:sample_count])),
            k=top_k,
            dim=1,
        )
        torch_scores_np = torch_scores.numpy()
        torch_ids_np = torch_ids.numpy()
    onnx_scores = []
    onnx_ids = []
    for index in range(sample_count):
        scores, token_ids = session.run(
            ["scores", "token_ids"],
            {"contexts": inputs.contexts[index : index + 1]},
        )
        onnx_scores.append(scores)
        onnx_ids.append(token_ids)
    onnx_scores_np = np.concatenate(onnx_scores, axis=0)
    onnx_ids_np = np.concatenate(onnx_ids, axis=0)
    absolute = np.abs(torch_scores_np - onnx_scores_np)
    return {
        "sample_count": sample_count,
        "top_k": top_k,
        "max_abs_diff": float(absolute.max(initial=0.0)),
        "mean_abs_diff": float(absolute.mean() if absolute.size else 0.0),
        "id_mismatch_count": int(np.count_nonzero(torch_ids_np != onnx_ids_np)),
    }


def compare_combined_onnx_with_pytorch(
    model: NextWordLm,
    session: ort.InferenceSession,
    inputs: ExportInputSet,
    sample_count: int,
    top_k: int,
) -> dict:
    sample_count = min(sample_count, inputs.size)
    if sample_count <= 0:
        return {}
    torch_topk_scores, torch_ids, torch_candidate_scores = pytorch_combined_outputs(
        model,
        inputs.contexts[:sample_count],
        inputs.candidate_ids[:sample_count],
        top_k,
    )
    onnx_topk_scores = []
    onnx_ids = []
    onnx_candidate_scores = []
    for index in range(sample_count):
        topk_scores, token_ids, candidate_scores = session.run(
            ["topk_scores", "token_ids", "candidate_scores"],
            {
                "contexts": inputs.contexts[index : index + 1],
                "candidate_ids": inputs.candidate_ids[index : index + 1],
            },
        )
        onnx_topk_scores.append(topk_scores)
        onnx_ids.append(token_ids)
        onnx_candidate_scores.append(candidate_scores)
    topk_scores_np = np.concatenate(onnx_topk_scores, axis=0)
    ids_np = np.concatenate(onnx_ids, axis=0)
    candidate_scores_np = np.concatenate(onnx_candidate_scores, axis=0)
    topk_absolute = np.abs(torch_topk_scores - topk_scores_np)
    candidate_absolute = np.abs(torch_candidate_scores - candidate_scores_np)
    return {
        "sample_count": sample_count,
        "top_k": top_k,
        "topk_max_abs_diff": float(topk_absolute.max(initial=0.0)),
        "topk_mean_abs_diff": float(topk_absolute.mean() if topk_absolute.size else 0.0),
        "candidate_max_abs_diff": float(candidate_absolute.max(initial=0.0)),
        "candidate_mean_abs_diff": float(
            candidate_absolute.mean() if candidate_absolute.size else 0.0
        ),
        "id_mismatch_count": int(np.count_nonzero(torch_ids != ids_np)),
    }


def compare_coreml_with_pytorch(
    model: NextWordLm,
    mlmodel,
    inputs: ExportInputSet,
    sample_count: int,
) -> dict:
    sample_count = min(sample_count, inputs.size)
    if sample_count <= 0:
        return {}
    with torch.no_grad():
        torch_scores = score_candidate_pool(
            model,
            torch.from_numpy(inputs.contexts[:sample_count].astype(np.int32)),
            torch.from_numpy(inputs.candidate_ids[:sample_count].astype(np.int32)),
        ).numpy()
    coreml_scores = np.concatenate(
        [
            mlmodel.predict(
                {
                    "contexts": inputs.contexts[index : index + 1].astype(np.int32),
                    "candidate_ids": inputs.candidate_ids[index : index + 1].astype(np.int32),
                }
            )["scores"]
            for index in range(sample_count)
        ],
        axis=0,
    )
    absolute = np.abs(torch_scores - coreml_scores)
    return {
        "sample_count": sample_count,
        "max_abs_diff": float(absolute.max(initial=0.0)),
        "mean_abs_diff": float(absolute.mean() if absolute.size else 0.0),
    }


def compare_topk_coreml_with_pytorch(
    model: NextWordLm,
    mlmodel,
    inputs: ExportInputSet,
    sample_count: int,
    top_k: int,
) -> dict:
    sample_count = min(sample_count, inputs.size)
    if sample_count <= 0:
        return {}
    with torch.no_grad():
        torch_scores, torch_ids = torch.topk(
            model(torch.from_numpy(inputs.contexts[:sample_count].astype(np.int32))),
            k=top_k,
            dim=1,
        )
        torch_scores_np = torch_scores.numpy()
        torch_ids_np = torch_ids.numpy()
    coreml_scores = []
    coreml_ids = []
    for index in range(sample_count):
        prediction = mlmodel.predict(
            {"contexts": inputs.contexts[index : index + 1].astype(np.int32)}
        )
        coreml_scores.append(prediction["scores"])
        coreml_ids.append(prediction["token_ids"])
    coreml_scores_np = np.concatenate(coreml_scores, axis=0)
    coreml_ids_np = np.concatenate(coreml_ids, axis=0)
    absolute = np.abs(torch_scores_np - coreml_scores_np)
    return {
        "sample_count": sample_count,
        "top_k": top_k,
        "max_abs_diff": float(absolute.max(initial=0.0)),
        "mean_abs_diff": float(absolute.mean() if absolute.size else 0.0),
        "id_mismatch_count": int(np.count_nonzero(torch_ids_np != coreml_ids_np)),
    }


def compare_combined_coreml_with_pytorch(
    model: NextWordLm,
    mlmodel,
    inputs: ExportInputSet,
    sample_count: int,
    top_k: int,
) -> dict:
    sample_count = min(sample_count, inputs.size)
    if sample_count <= 0:
        return {}
    torch_topk_scores, torch_ids, torch_candidate_scores = pytorch_combined_outputs(
        model,
        inputs.contexts[:sample_count].astype(np.int32),
        inputs.candidate_ids[:sample_count].astype(np.int32),
        top_k,
    )
    coreml_topk_scores = []
    coreml_ids = []
    coreml_candidate_scores = []
    for index in range(sample_count):
        prediction = mlmodel.predict(
            {
                "contexts": inputs.contexts[index : index + 1].astype(np.int32),
                "candidate_ids": inputs.candidate_ids[index : index + 1].astype(np.int32),
            }
        )
        coreml_topk_scores.append(prediction["topk_scores"])
        coreml_ids.append(prediction["token_ids"])
        coreml_candidate_scores.append(prediction["candidate_scores"])
    topk_scores_np = np.concatenate(coreml_topk_scores, axis=0)
    ids_np = np.concatenate(coreml_ids, axis=0)
    candidate_scores_np = np.concatenate(coreml_candidate_scores, axis=0)
    topk_absolute = np.abs(torch_topk_scores - topk_scores_np)
    candidate_absolute = np.abs(torch_candidate_scores - candidate_scores_np)
    return {
        "sample_count": sample_count,
        "top_k": top_k,
        "topk_max_abs_diff": float(topk_absolute.max(initial=0.0)),
        "topk_mean_abs_diff": float(topk_absolute.mean() if topk_absolute.size else 0.0),
        "candidate_max_abs_diff": float(candidate_absolute.max(initial=0.0)),
        "candidate_mean_abs_diff": float(
            candidate_absolute.mean() if candidate_absolute.size else 0.0
        ),
        "id_mismatch_count": int(np.count_nonzero(torch_ids != ids_np)),
    }


@torch.no_grad()
def pytorch_combined_outputs(
    model: NextWordLm,
    contexts: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts_tensor = torch.from_numpy(contexts)
    candidate_ids_tensor = torch.from_numpy(candidate_ids).long()
    hidden = model.encode_context(contexts_tensor)
    logits = hidden @ model.token_embedding.weight.T + model.output_bias
    topk_scores, token_ids = torch.topk(logits, k=top_k, dim=1)
    candidate_scores = torch.gather(logits, dim=1, index=candidate_ids_tensor)
    return (
        topk_scores.numpy(),
        token_ids.numpy(),
        candidate_scores.numpy(),
    )


def benchmark_onnx(
    session: ort.InferenceSession,
    inputs: ExportInputSet,
    iterations: int,
    batch_size: int,
) -> dict | None:
    if iterations <= 0:
        return None
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if batch_size != 1:
        raise ValueError("fixed keyboard-time ONNX export currently supports batch size 1")
    sample_size = min(inputs.size, max(batch_size, min(iterations * batch_size, 4096)))
    if sample_size == 0:
        return None
    contexts = inputs.contexts[:sample_size]
    candidate_ids = inputs.candidate_ids[:sample_size]
    warmup = min(100, iterations)
    for index in range(warmup):
        start, end = batch_window(index, batch_size, sample_size)
        session.run(
            ["scores"],
            {"contexts": contexts[start:end], "candidate_ids": candidate_ids[start:end]},
        )
    started_at = time.perf_counter()
    for index in range(iterations):
        start, end = batch_window(index, batch_size, sample_size)
        session.run(
            ["scores"],
            {"contexts": contexts[start:end], "candidate_ids": candidate_ids[start:end]},
        )
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "iterations": iterations,
        "batch_size": batch_size,
        "sample_size": sample_size,
        "elapsed_seconds": elapsed_seconds,
        "mean_us_per_batch": elapsed_seconds * 1_000_000.0 / iterations,
        "mean_us_per_item": elapsed_seconds * 1_000_000.0 / (iterations * batch_size),
    }


def benchmark_topk_onnx(
    session: ort.InferenceSession,
    inputs: ExportInputSet,
    iterations: int,
    batch_size: int,
) -> dict | None:
    if iterations <= 0:
        return None
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if batch_size != 1:
        raise ValueError("fixed keyboard-time ONNX export currently supports batch size 1")
    sample_size = min(inputs.size, max(batch_size, min(iterations * batch_size, 4096)))
    if sample_size == 0:
        return None
    contexts = inputs.contexts[:sample_size]
    warmup = min(100, iterations)
    for index in range(warmup):
        start, end = batch_window(index, batch_size, sample_size)
        session.run(["scores", "token_ids"], {"contexts": contexts[start:end]})
    started_at = time.perf_counter()
    for index in range(iterations):
        start, end = batch_window(index, batch_size, sample_size)
        session.run(["scores", "token_ids"], {"contexts": contexts[start:end]})
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "iterations": iterations,
        "batch_size": batch_size,
        "sample_size": sample_size,
        "elapsed_seconds": elapsed_seconds,
        "mean_us_per_batch": elapsed_seconds * 1_000_000.0 / iterations,
        "mean_us_per_item": elapsed_seconds * 1_000_000.0 / (iterations * batch_size),
    }


def benchmark_combined_onnx(
    session: ort.InferenceSession,
    inputs: ExportInputSet,
    iterations: int,
    batch_size: int,
) -> dict | None:
    if iterations <= 0:
        return None
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if batch_size != 1:
        raise ValueError("fixed keyboard-time ONNX export currently supports batch size 1")
    sample_size = min(inputs.size, max(batch_size, min(iterations * batch_size, 4096)))
    if sample_size == 0:
        return None
    contexts = inputs.contexts[:sample_size]
    candidate_ids = inputs.candidate_ids[:sample_size]
    warmup = min(100, iterations)
    for index in range(warmup):
        start, end = batch_window(index, batch_size, sample_size)
        session.run(
            ["topk_scores", "token_ids", "candidate_scores"],
            {"contexts": contexts[start:end], "candidate_ids": candidate_ids[start:end]},
        )
    started_at = time.perf_counter()
    for index in range(iterations):
        start, end = batch_window(index, batch_size, sample_size)
        session.run(
            ["topk_scores", "token_ids", "candidate_scores"],
            {"contexts": contexts[start:end], "candidate_ids": candidate_ids[start:end]},
        )
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "iterations": iterations,
        "batch_size": batch_size,
        "sample_size": sample_size,
        "elapsed_seconds": elapsed_seconds,
        "mean_us_per_batch": elapsed_seconds * 1_000_000.0 / iterations,
        "mean_us_per_item": elapsed_seconds * 1_000_000.0 / (iterations * batch_size),
    }


def benchmark_coreml(
    mlmodel,
    inputs: ExportInputSet,
    iterations: int,
    batch_size: int,
) -> dict | None:
    if iterations <= 0:
        return None
    if batch_size != 1:
        raise ValueError("fixed keyboard-time Core ML export currently supports batch size 1")
    sample_size = min(inputs.size, max(batch_size, min(iterations * batch_size, 4096)))
    if sample_size == 0:
        return None
    contexts = inputs.contexts[:sample_size].astype(np.int32)
    candidate_ids = inputs.candidate_ids[:sample_size].astype(np.int32)
    warmup = min(100, iterations)
    for index in range(warmup):
        start, end = batch_window(index, batch_size, sample_size)
        mlmodel.predict(
            {"contexts": contexts[start:end], "candidate_ids": candidate_ids[start:end]}
        )
    started_at = time.perf_counter()
    for index in range(iterations):
        start, end = batch_window(index, batch_size, sample_size)
        mlmodel.predict(
            {"contexts": contexts[start:end], "candidate_ids": candidate_ids[start:end]}
        )
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "iterations": iterations,
        "batch_size": batch_size,
        "sample_size": sample_size,
        "elapsed_seconds": elapsed_seconds,
        "mean_us_per_batch": elapsed_seconds * 1_000_000.0 / iterations,
        "mean_us_per_item": elapsed_seconds * 1_000_000.0 / (iterations * batch_size),
    }


def benchmark_topk_coreml(
    mlmodel,
    inputs: ExportInputSet,
    iterations: int,
    batch_size: int,
) -> dict | None:
    if iterations <= 0:
        return None
    if batch_size != 1:
        raise ValueError("fixed keyboard-time Core ML export currently supports batch size 1")
    sample_size = min(inputs.size, max(batch_size, min(iterations * batch_size, 4096)))
    if sample_size == 0:
        return None
    contexts = inputs.contexts[:sample_size].astype(np.int32)
    warmup = min(100, iterations)
    for index in range(warmup):
        start, end = batch_window(index, batch_size, sample_size)
        mlmodel.predict({"contexts": contexts[start:end]})
    started_at = time.perf_counter()
    for index in range(iterations):
        start, end = batch_window(index, batch_size, sample_size)
        mlmodel.predict({"contexts": contexts[start:end]})
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "iterations": iterations,
        "batch_size": batch_size,
        "sample_size": sample_size,
        "elapsed_seconds": elapsed_seconds,
        "mean_us_per_batch": elapsed_seconds * 1_000_000.0 / iterations,
        "mean_us_per_item": elapsed_seconds * 1_000_000.0 / (iterations * batch_size),
    }


def benchmark_combined_coreml(
    mlmodel,
    inputs: ExportInputSet,
    iterations: int,
    batch_size: int,
) -> dict | None:
    if iterations <= 0:
        return None
    if batch_size != 1:
        raise ValueError("fixed keyboard-time Core ML export currently supports batch size 1")
    sample_size = min(inputs.size, max(batch_size, min(iterations * batch_size, 4096)))
    if sample_size == 0:
        return None
    contexts = inputs.contexts[:sample_size].astype(np.int32)
    candidate_ids = inputs.candidate_ids[:sample_size].astype(np.int32)
    warmup = min(100, iterations)
    for index in range(warmup):
        start, end = batch_window(index, batch_size, sample_size)
        mlmodel.predict({"contexts": contexts[start:end], "candidate_ids": candidate_ids[start:end]})
    started_at = time.perf_counter()
    for index in range(iterations):
        start, end = batch_window(index, batch_size, sample_size)
        mlmodel.predict({"contexts": contexts[start:end], "candidate_ids": candidate_ids[start:end]})
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "iterations": iterations,
        "batch_size": batch_size,
        "sample_size": sample_size,
        "elapsed_seconds": elapsed_seconds,
        "mean_us_per_batch": elapsed_seconds * 1_000_000.0 / iterations,
        "mean_us_per_item": elapsed_seconds * 1_000_000.0 / (iterations * batch_size),
    }


def evaluate_static_pool(inputs: ExportInputSet) -> dict:
    hits = Counter()
    reciprocal_rank_sum = 0.0
    pool_hits = 0
    for label, candidates in zip(inputs.labels.tolist(), inputs.candidate_rows):
        try:
            rank = candidates.index(label) + 1
        except ValueError:
            rank = 0
        if rank:
            pool_hits += 1
            reciprocal_rank_sum += 1.0 / rank
        for cutoff in REPORT_CUTOFFS:
            if rank and rank <= cutoff:
                hits[cutoff] += 1
    report = rank_report(hits, reciprocal_rank_sum, inputs)
    report["pool_recall"] = pool_hits / max(1, inputs.size)
    report["pool_recall_all_targets"] = pool_hits / max(1, inputs.total_targets)
    return report


def evaluate_onnx_rerank(
    session: ort.InferenceSession,
    inputs: ExportInputSet,
    rank_penalties: tuple[float, ...],
    lock_first: bool,
) -> list[dict]:
    counters = {
        penalty: {
            "hits": Counter(),
            "reciprocal_rank_sum": 0.0,
        }
        for penalty in rank_penalties
    }
    for index, (label, candidates) in enumerate(zip(inputs.labels.tolist(), inputs.candidate_rows)):
        if not candidates:
            continue
        scores = session.run(
            ["scores"],
            {
                "contexts": inputs.contexts[index : index + 1],
                "candidate_ids": inputs.candidate_ids[index : index + 1],
            },
        )[0][0]
        indexed_candidates = list(enumerate(candidates))
        for penalty, counter in counters.items():
            scored_tail = sorted(
                indexed_candidates[1:] if lock_first else indexed_candidates,
                key=lambda item: (-float(scores[item[0]]) + penalty * item[0], item[0]),
            )
            ranked = [indexed_candidates[0], *scored_tail] if lock_first else scored_tail
            rank = 0
            for sorted_index, (_, token_id) in enumerate(ranked, start=1):
                if token_id == label:
                    rank = sorted_index
                    break
            if rank:
                counter["reciprocal_rank_sum"] += 1.0 / rank
            for cutoff in REPORT_CUTOFFS:
                if rank and rank <= cutoff:
                    counter["hits"][cutoff] += 1
    reports = []
    for penalty in rank_penalties:
        report = rank_report(
            counters[penalty]["hits"],
            counters[penalty]["reciprocal_rank_sum"],
            inputs,
        )
        report["rank_penalty"] = penalty
        report["lock_first"] = lock_first
        reports.append(report)
    return reports


def evaluate_topk_candidates(
    token_id_rows: np.ndarray,
    inputs: ExportInputSet,
) -> dict:
    hits = Counter()
    reciprocal_rank_sum = 0.0
    neural_hit_count = 0
    union_hit_count = 0
    top_k = int(token_id_rows.shape[1]) if token_id_rows.ndim == 2 else 0

    for label, neural_row, ngram_row in zip(
        inputs.labels.tolist(),
        token_id_rows.tolist(),
        inputs.candidate_rows,
    ):
        try:
            rank = neural_row.index(label) + 1
        except ValueError:
            rank = 0
        ngram_hit = label in ngram_row
        union_hit = ngram_hit or rank > 0

        if rank:
            neural_hit_count += 1
            reciprocal_rank_sum += 1.0 / rank
        if union_hit:
            union_hit_count += 1
        for cutoff in REPORT_CUTOFFS:
            if rank and rank <= min(cutoff, top_k):
                hits[cutoff] += 1

    report = rank_report(hits, reciprocal_rank_sum, inputs)
    ngram_pool = evaluate_static_pool(inputs)
    report["top_k"] = top_k
    report["ngram_pool_recall"] = ngram_pool["pool_recall"]
    report["ngram_pool_recall_all_targets"] = ngram_pool["pool_recall_all_targets"]
    report["neural_recall"] = neural_hit_count / max(1, inputs.size)
    report["neural_recall_all_targets"] = neural_hit_count / max(1, inputs.total_targets)
    report["union_recall"] = union_hit_count / max(1, inputs.size)
    report["union_recall_all_targets"] = union_hit_count / max(1, inputs.total_targets)
    report["absolute_union_gain"] = report["union_recall"] - report["ngram_pool_recall"]
    report["absolute_union_gain_all_targets"] = (
        report["union_recall_all_targets"] - report["ngram_pool_recall_all_targets"]
    )
    return report


def evaluate_topk_merge_policies(
    token_id_rows: np.ndarray,
    inputs: ExportInputSet,
    visible_candidates: tuple[int, ...],
    locked_static_prefixes: tuple[int, ...],
) -> list[dict]:
    reports = []
    static_pool = evaluate_static_pool(inputs)
    for visible in visible_candidates:
        for locked_prefix in locked_static_prefixes:
            reports.append(
                evaluate_topk_merge_policy(
                    token_id_rows,
                    inputs,
                    visible,
                    locked_prefix,
                    static_pool,
                )
            )
    return reports


def evaluate_topk_merge_policy(
    token_id_rows: np.ndarray,
    inputs: ExportInputSet,
    visible_candidates: int,
    locked_static_prefix: int,
    static_pool: dict,
) -> dict:
    hits = Counter()
    reciprocal_rank_sum = 0.0
    merged_hit_count = 0
    visible_candidates = max(1, int(visible_candidates))
    locked_static_prefix = max(0, int(locked_static_prefix))

    for label, neural_row, ngram_row in zip(
        inputs.labels.tolist(),
        token_id_rows.tolist(),
        inputs.candidate_rows,
    ):
        merged = merged_candidate_row(
            ngram_row,
            neural_row,
            visible_candidates,
            locked_static_prefix,
        )
        try:
            rank = merged.index(label) + 1
        except ValueError:
            rank = 0
        if rank:
            merged_hit_count += 1
            reciprocal_rank_sum += 1.0 / rank
        for cutoff in REPORT_CUTOFFS:
            if rank and rank <= cutoff:
                hits[cutoff] += 1

    report = rank_report(hits, reciprocal_rank_sum, inputs)
    report["visible_candidates"] = visible_candidates
    report["locked_static_prefix"] = locked_static_prefix
    report["merged_recall"] = merged_hit_count / max(1, inputs.size)
    report["merged_recall_all_targets"] = merged_hit_count / max(1, inputs.total_targets)
    for cutoff in REPORT_CUTOFFS:
        report[f"top{cutoff}_all_target_gain_vs_static"] = (
            report[f"top{cutoff}_all_targets"] - static_pool[f"top{cutoff}_all_targets"]
        )
    report["mrr_all_target_gain_vs_static"] = (
        report["mrr_all_targets"] - static_pool["mrr_all_targets"]
    )
    return report


def evaluate_scored_union_policies(
    model: NextWordLm,
    token_id_rows: np.ndarray,
    inputs: ExportInputSet,
    static_bonuses: tuple[float, ...],
    static_rank_penalties: tuple[float, ...],
    generated_penalties: tuple[float, ...],
    locked_static_prefixes: tuple[int, ...],
    overlap_bonuses: tuple[float, ...] = (0.0,),
    generated_rank_penalties: tuple[float, ...] = (0.0,),
    static_log_count_scales: tuple[float, ...] = (0.0,),
    static_source_bonuses: tuple[float, ...] = (0.0,),
    batch_size: int = 256,
    top_profiles: int = 20,
) -> dict:
    union = build_union_candidate_matrix(token_id_rows, inputs)
    if union["ids"].shape[0] == 0:
        return {"profiles": [], "best_by_top5": {}, "best_by_mrr": {}}

    model_scores = score_union_candidates(model, inputs.contexts, union["ids"], batch_size)
    generated_rank = generated_rank_matrix(union, token_id_rows)
    return evaluate_scored_union_score_grid(
        model_scores,
        union,
        generated_rank,
        inputs,
        static_bonuses,
        static_rank_penalties,
        generated_penalties,
        locked_static_prefixes,
        overlap_bonuses,
        generated_rank_penalties,
        static_log_count_scales,
        static_source_bonuses,
        top_profiles,
        candidate_scorer="pytorch_reference",
    )


def evaluate_exported_scored_union_policies(
    token_id_rows: np.ndarray,
    topk_score_rows: np.ndarray,
    candidate_score_rows: np.ndarray,
    inputs: ExportInputSet,
    static_bonuses: tuple[float, ...],
    static_rank_penalties: tuple[float, ...],
    generated_penalties: tuple[float, ...],
    locked_static_prefixes: tuple[int, ...],
    overlap_bonuses: tuple[float, ...] = (0.0,),
    generated_rank_penalties: tuple[float, ...] = (0.0,),
    static_log_count_scales: tuple[float, ...] = (0.0,),
    static_source_bonuses: tuple[float, ...] = (0.0,),
    top_profiles: int = 20,
) -> dict:
    union = build_union_candidate_matrix(token_id_rows, inputs)
    if union["ids"].shape[0] == 0:
        return {"profiles": [], "best_by_top5": {}, "best_by_mrr": {}}

    model_scores = exported_union_model_scores(
        union,
        token_id_rows,
        topk_score_rows,
        candidate_score_rows,
        inputs,
    )
    generated_rank = generated_rank_matrix(union, token_id_rows)
    return evaluate_scored_union_score_grid(
        model_scores,
        union,
        generated_rank,
        inputs,
        static_bonuses,
        static_rank_penalties,
        generated_penalties,
        locked_static_prefixes,
        overlap_bonuses,
        generated_rank_penalties,
        static_log_count_scales,
        static_source_bonuses,
        top_profiles,
        candidate_scorer="exported_graph",
    )


def evaluate_exported_scored_union_split_selection(
    token_id_rows: np.ndarray,
    topk_score_rows: np.ndarray,
    candidate_score_rows: np.ndarray,
    inputs: ExportInputSet,
    static_bonuses: tuple[float, ...],
    static_rank_penalties: tuple[float, ...],
    generated_penalties: tuple[float, ...],
    locked_static_prefixes: tuple[int, ...],
    overlap_bonuses: tuple[float, ...] = (0.0,),
    generated_rank_penalties: tuple[float, ...] = (0.0,),
    static_log_count_scales: tuple[float, ...] = (0.0,),
    static_source_bonuses: tuple[float, ...] = (0.0,),
    top_profiles: int = 20,
    eval_mod: int = 5,
    eval_remainder: int = 0,
) -> dict:
    union = build_union_candidate_matrix(token_id_rows, inputs)
    if union["ids"].shape[0] == 0:
        return {"enabled": False, "reason": "empty union"}

    model_scores = exported_union_model_scores(
        union,
        token_id_rows,
        topk_score_rows,
        candidate_score_rows,
        inputs,
    )
    generated_rank = generated_rank_matrix(union, token_id_rows)
    return evaluate_scored_union_split_selection_grid(
        model_scores,
        union,
        generated_rank,
        inputs,
        static_bonuses,
        static_rank_penalties,
        generated_penalties,
        locked_static_prefixes,
        overlap_bonuses,
        generated_rank_penalties,
        static_log_count_scales,
        static_source_bonuses,
        top_profiles,
        candidate_scorer="exported_graph",
        eval_mod=eval_mod,
        eval_remainder=eval_remainder,
    )


def evaluate_scored_union_score_grid(
    model_scores: np.ndarray,
    union: dict,
    generated_rank: np.ndarray,
    inputs: ExportInputSet,
    static_bonuses: tuple[float, ...],
    static_rank_penalties: tuple[float, ...],
    generated_penalties: tuple[float, ...],
    locked_static_prefixes: tuple[int, ...],
    overlap_bonuses: tuple[float, ...],
    generated_rank_penalties: tuple[float, ...],
    static_log_count_scales: tuple[float, ...],
    static_source_bonuses: tuple[float, ...],
    top_profiles: int,
    candidate_scorer: str,
) -> dict:
    static_pool = evaluate_static_pool(inputs)

    profiles = []
    for locked_prefix in locked_static_prefixes:
        for static_bonus in static_bonuses:
            for static_rank_penalty in static_rank_penalties:
                for generated_penalty in generated_penalties:
                    for overlap_bonus in overlap_bonuses:
                        for generated_rank_penalty in generated_rank_penalties:
                            for static_log_count_scale in static_log_count_scales:
                                for static_source_bonus in static_source_bonuses:
                                    policy = scored_union_policy(
                                        locked_prefix,
                                        static_bonus,
                                        static_rank_penalty,
                                        generated_penalty,
                                        overlap_bonus,
                                        generated_rank_penalty,
                                        static_log_count_scale,
                                        static_source_bonus,
                                    )
                                    profiles.append(
                                        evaluate_scored_union_policy_profile(
                                            model_scores,
                                            union,
                                            generated_rank,
                                            inputs,
                                            static_pool,
                                            policy,
                                        )
                                    )

    top_profiles = max(1, top_profiles)
    profiles_by_top5 = sorted(
        profiles,
        key=lambda profile: (
            profile["top5_all_targets"],
            profile["mrr_all_targets"],
            profile["top10_all_targets"],
        ),
        reverse=True,
    )[:top_profiles]
    profiles_by_mrr = sorted(
        profiles,
        key=lambda profile: (
            profile["mrr_all_targets"],
            profile["top5_all_targets"],
            profile["top10_all_targets"],
        ),
        reverse=True,
    )[:top_profiles]
    return {
        "union_candidate_count_max": int(union["ids"].shape[1]),
        "union_candidate_count_mean": float(union["candidate_counts"].mean()),
        "candidate_scorer": candidate_scorer,
        "best_by_top5": profiles_by_top5[0],
        "best_by_mrr": profiles_by_mrr[0],
        "top_profiles_by_top5": profiles_by_top5,
        "top_profiles_by_mrr": profiles_by_mrr,
    }


def evaluate_scored_union_split_selection_grid(
    model_scores: np.ndarray,
    union: dict,
    generated_rank: np.ndarray,
    inputs: ExportInputSet,
    static_bonuses: tuple[float, ...],
    static_rank_penalties: tuple[float, ...],
    generated_penalties: tuple[float, ...],
    locked_static_prefixes: tuple[int, ...],
    overlap_bonuses: tuple[float, ...],
    generated_rank_penalties: tuple[float, ...],
    static_log_count_scales: tuple[float, ...],
    static_source_bonuses: tuple[float, ...],
    top_profiles: int,
    candidate_scorer: str,
    eval_mod: int,
    eval_remainder: int,
) -> dict:
    eval_mod = int(eval_mod)
    if eval_mod < 2:
        return {
            "enabled": False,
            "reason": "selection eval split disabled",
            "candidate_scorer": candidate_scorer,
        }
    eval_remainder = int(eval_remainder) % eval_mod
    indexes = np.arange(inputs.size)
    eval_indexes = indexes[indexes % eval_mod == eval_remainder]
    selection_indexes = indexes[indexes % eval_mod != eval_remainder]
    if selection_indexes.size == 0 or eval_indexes.size == 0:
        return {
            "enabled": False,
            "reason": "empty selection/eval split",
            "candidate_scorer": candidate_scorer,
            "selection_eval_mod": eval_mod,
            "selection_eval_remainder": eval_remainder,
        }

    selection_inputs = subset_inputs(inputs, selection_indexes)
    eval_inputs = subset_inputs(inputs, eval_indexes)
    selection_static = evaluate_static_pool(selection_inputs)
    eval_static = evaluate_static_pool(eval_inputs)
    selection_union = sliced_union(union, selection_indexes)
    eval_union = sliced_union(union, eval_indexes)
    selection_source_views = build_source_eval_views(selection_inputs, selection_union)
    eval_source_views = build_source_eval_views(eval_inputs, eval_union)

    selection_profiles = []
    for locked_prefix in locked_static_prefixes:
        for static_bonus in static_bonuses:
            for static_rank_penalty in static_rank_penalties:
                for generated_penalty in generated_penalties:
                    for overlap_bonus in overlap_bonuses:
                        for generated_rank_penalty in generated_rank_penalties:
                            for static_log_count_scale in static_log_count_scales:
                                for static_source_bonus in static_source_bonuses:
                                    policy = scored_union_policy(
                                        locked_prefix,
                                        static_bonus,
                                        static_rank_penalty,
                                        generated_penalty,
                                        overlap_bonus,
                                        generated_rank_penalty,
                                        static_log_count_scale,
                                        static_source_bonus,
                                    )
                                    selection_profiles.append(
                                        evaluate_scored_union_policy_profile(
                                            model_scores[selection_indexes],
                                            selection_union,
                                            generated_rank[selection_indexes],
                                            selection_inputs,
                                            selection_static,
                                            policy,
                                        )
                                    )

    top_profiles = max(1, top_profiles)
    profiles_by_top5 = sorted(
        selection_profiles,
        key=lambda profile: (
            profile["top5_all_targets"],
            profile["mrr_all_targets"],
            profile["top10_all_targets"],
        ),
        reverse=True,
    )[:top_profiles]
    profiles_by_mrr = sorted(
        selection_profiles,
        key=lambda profile: (
            profile["mrr_all_targets"],
            profile["top5_all_targets"],
            profile["top10_all_targets"],
        ),
        reverse=True,
    )[:top_profiles]

    source_balanced_profiles = [
        annotate_profile_source_balance(
            profile,
            model_scores[selection_indexes],
            selection_union,
            generated_rank[selection_indexes],
            selection_inputs,
            selection_source_views,
        )
        for profile in selection_profiles
    ]
    profiles_balanced_by_top5 = sorted(
        source_balanced_profiles,
        key=lambda profile: source_balanced_selection_key(profile, "top5"),
        reverse=True,
    )[:top_profiles]
    profiles_balanced_by_mrr = sorted(
        source_balanced_profiles,
        key=lambda profile: source_balanced_selection_key(profile, "mrr"),
        reverse=True,
    )[:top_profiles]

    selected_by_top5 = split_selected_policy_report(
        profiles_by_top5[0],
        model_scores,
        generated_rank,
        eval_indexes,
        eval_union,
        eval_inputs,
        eval_static,
        "best_by_top5",
        eval_source_views,
    )
    selected_by_mrr = split_selected_policy_report(
        profiles_by_mrr[0],
        model_scores,
        generated_rank,
        eval_indexes,
        eval_union,
        eval_inputs,
        eval_static,
        "best_by_mrr",
        eval_source_views,
    )
    selected_by_balanced_top5 = split_selected_policy_report(
        profiles_balanced_by_top5[0],
        model_scores,
        generated_rank,
        eval_indexes,
        eval_union,
        eval_inputs,
        eval_static,
        "balanced_by_top5",
        eval_source_views,
    )
    selected_by_balanced_mrr = split_selected_policy_report(
        profiles_balanced_by_mrr[0],
        model_scores,
        generated_rank,
        eval_indexes,
        eval_union,
        eval_inputs,
        eval_static,
        "balanced_by_mrr",
        eval_source_views,
    )

    return {
        "enabled": True,
        "union_candidate_count_max": int(union["ids"].shape[1]),
        "union_candidate_count_mean": float(union["candidate_counts"].mean()),
        "candidate_scorer": candidate_scorer,
        "selection_eval_mod": eval_mod,
        "selection_eval_remainder": eval_remainder,
        "selection_size": int(selection_indexes.size),
        "eval_size": int(eval_indexes.size),
        "selection_static_pool": pick_rank_metrics(selection_static),
        "eval_static_pool": pick_rank_metrics(eval_static),
        "selected_by_top5": selected_by_top5,
        "selected_by_mrr": selected_by_mrr,
        "selected_by_balanced_top5": selected_by_balanced_top5,
        "selected_by_balanced_mrr": selected_by_balanced_mrr,
        "selection_top_profiles_by_top5": profiles_by_top5,
        "selection_top_profiles_by_mrr": profiles_by_mrr,
        "selection_top_profiles_balanced_by_top5": profiles_balanced_by_top5,
        "selection_top_profiles_balanced_by_mrr": profiles_balanced_by_mrr,
    }


def annotate_profile_source_balance(
    profile: dict,
    model_scores: np.ndarray,
    union: dict,
    generated_rank: np.ndarray,
    inputs: ExportInputSet,
    source_views: list[dict],
) -> dict:
    annotated = dict(profile)
    policy = scored_union_policy_from_profile(profile)
    per_source = evaluate_scored_union_policy_by_source(
        model_scores,
        union,
        generated_rank,
        inputs,
        policy,
        source_views,
    )
    annotated["selection_per_source"] = per_source
    annotated["selection_source_balance"] = source_balance_report(per_source)
    return annotated


def source_balance_report(per_source: dict[str, dict]) -> dict:
    if not per_source:
        return {
            "source_count": 0,
            "all_sources_non_regressing": False,
            "min_top5_all_target_gain_vs_static": 0.0,
            "min_mrr_all_target_gain_vs_static": 0.0,
        }
    top5_gains = [
        profile["top5_all_target_gain_vs_static"]
        for profile in per_source.values()
    ]
    mrr_gains = [
        profile["mrr_all_target_gain_vs_static"]
        for profile in per_source.values()
    ]
    return {
        "source_count": len(per_source),
        "all_sources_non_regressing": all(
            top5_gain >= 0.0 and mrr_gain >= 0.0
            for top5_gain, mrr_gain in zip(top5_gains, mrr_gains)
        ),
        "min_top5_all_target_gain_vs_static": min(top5_gains),
        "min_mrr_all_target_gain_vs_static": min(mrr_gains),
    }


def source_balanced_selection_key(profile: dict, primary: str) -> tuple:
    balance = profile["selection_source_balance"]
    aggregate_improves = (
        profile["top5_all_target_gain_vs_static"] > 0.0
        and profile["mrr_all_target_gain_vs_static"] > 0.0
    )
    if primary == "top5":
        primary_value = profile["top5_all_targets"]
        secondary_value = profile["mrr_all_targets"]
        primary_floor = balance["min_top5_all_target_gain_vs_static"]
        secondary_floor = balance["min_mrr_all_target_gain_vs_static"]
    elif primary == "mrr":
        primary_value = profile["mrr_all_targets"]
        secondary_value = profile["top5_all_targets"]
        primary_floor = balance["min_mrr_all_target_gain_vs_static"]
        secondary_floor = balance["min_top5_all_target_gain_vs_static"]
    else:
        raise ValueError(f"unsupported balanced primary metric: {primary}")
    return (
        aggregate_improves and balance["all_sources_non_regressing"],
        primary_floor,
        secondary_floor,
        primary_value,
        secondary_value,
        profile["top10_all_targets"],
    )


def split_selected_policy_report(
    selection_profile: dict,
    model_scores: np.ndarray,
    generated_rank: np.ndarray,
    eval_indexes: np.ndarray,
    eval_union: dict,
    eval_inputs: ExportInputSet,
    eval_static: dict,
    selection: str,
    eval_source_views: list[dict],
) -> dict:
    policy = scored_union_policy_from_profile(selection_profile)
    eval_model_scores = model_scores[eval_indexes]
    eval_generated_rank = generated_rank[eval_indexes]
    eval_profile = evaluate_scored_union_policy_profile(
        eval_model_scores,
        eval_union,
        eval_generated_rank,
        eval_inputs,
        eval_static,
        policy,
    )
    eval_per_source = evaluate_scored_union_policy_by_source(
        eval_model_scores,
        eval_union,
        eval_generated_rank,
        eval_inputs,
        policy,
        eval_source_views,
    )
    return {
        "selection": selection,
        "selection_profile": selection_profile,
        "eval_profile": eval_profile,
        "eval_per_source": eval_per_source,
        "accepted_for_packaging": (
            eval_profile["top5_all_target_gain_vs_static"] > 0.0
            and eval_profile["mrr_all_target_gain_vs_static"] > 0.0
        ),
        "accepted_for_packaging_all_eval_sources": all(
            source_profile["top5_all_target_gain_vs_static"] >= 0.0
            and source_profile["mrr_all_target_gain_vs_static"] >= 0.0
            for source_profile in eval_per_source.values()
        ),
    }


def evaluate_scored_union_policy_by_source(
    model_scores: np.ndarray,
    union: dict,
    generated_rank: np.ndarray,
    inputs: ExportInputSet,
    policy: dict,
    source_views: list[dict] | None = None,
) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for view in source_views or build_source_eval_views(inputs, union):
        indexes = view["indexes"]
        source_inputs = view["inputs"]
        source_static = view["static"]
        source_profile = evaluate_scored_union_policy_profile(
            model_scores[indexes],
            view["union"],
            generated_rank[indexes],
            source_inputs,
            source_static,
            policy,
        )
        reports[view["source_name"]] = {
            "eligible_targets": source_profile["eligible_targets"],
            "total_targets": source_profile["total_targets"],
            "static_top5_all_targets": source_static["top5_all_targets"],
            "static_top10_all_targets": source_static["top10_all_targets"],
            "static_mrr_all_targets": source_static["mrr_all_targets"],
            "top1_all_targets": source_profile["top1_all_targets"],
            "top5_all_targets": source_profile["top5_all_targets"],
            "top10_all_targets": source_profile["top10_all_targets"],
            "mrr_all_targets": source_profile["mrr_all_targets"],
            "top5_all_target_gain_vs_static": source_profile[
                "top5_all_target_gain_vs_static"
            ],
            "top10_all_target_gain_vs_static": source_profile[
                "top10_all_target_gain_vs_static"
            ],
            "mrr_all_target_gain_vs_static": source_profile[
                "mrr_all_target_gain_vs_static"
            ],
        }
    return reports


def build_source_eval_views(inputs: ExportInputSet, union: dict) -> list[dict]:
    views: list[dict] = []
    for source_id, source_name in enumerate(inputs.source_names):
        indexes = np.flatnonzero(inputs.source_ids == source_id)
        if indexes.size == 0:
            continue
        source_inputs = subset_inputs(inputs, indexes)
        views.append(
            {
                "source_name": source_name,
                "indexes": indexes,
                "inputs": source_inputs,
                "union": sliced_union(union, indexes),
                "static": evaluate_static_pool(source_inputs),
            }
        )
    return views


def evaluate_scored_union_policy_profile(
    model_scores: np.ndarray,
    union: dict,
    generated_rank: np.ndarray,
    inputs: ExportInputSet,
    static_pool: dict,
    policy: dict,
) -> dict:
    adjusted_scores = scored_union_adjusted_scores(
        model_scores,
        union,
        generated_rank,
        policy,
    )
    profile = metrics_for_union_scores(
        adjusted_scores,
        union["label_positions"],
        union["valid"],
        inputs,
    )
    profile.update(policy)
    for cutoff in REPORT_CUTOFFS:
        profile[f"top{cutoff}_all_target_gain_vs_static"] = (
            profile[f"top{cutoff}_all_targets"]
            - static_pool[f"top{cutoff}_all_targets"]
        )
    profile["mrr_all_target_gain_vs_static"] = (
        profile["mrr_all_targets"] - static_pool["mrr_all_targets"]
    )
    return profile


def scored_union_adjusted_scores(
    model_scores: np.ndarray,
    union: dict,
    generated_rank: np.ndarray,
    policy: dict,
) -> np.ndarray:
    is_static = union["is_static"]
    static_rank = union["static_rank"].astype(np.float32)
    is_generated = generated_rank < 999
    generated_rank_f32 = generated_rank.astype(np.float32)
    locked_bonus = np.zeros_like(model_scores, dtype=np.float32)
    locked_prefix = int(policy["locked_static_prefix"])
    if locked_prefix > 0:
        locked_bonus[is_static & (static_rank < locked_prefix)] = 1.0e6
    return (
        model_scores
        + locked_bonus
        + float(policy["static_bonus"]) * is_static.astype(np.float32)
        - float(policy["static_rank_penalty"]) * np.where(is_static, static_rank, 0.0)
        + float(policy["static_log_count_scale"])
        * np.where(is_static, np.log1p(union["static_count"]), 0.0)
        + float(policy["static_source_bonus"])
        * np.where(is_static, union["static_source_order"], 0.0)
        - float(policy["generated_penalty"]) * (~is_static).astype(np.float32)
        + float(policy["overlap_bonus"]) * (is_static & is_generated).astype(np.float32)
        - float(policy["generated_rank_penalty"])
        * np.where(is_generated, generated_rank_f32, 0.0)
    )


def scored_union_policy(
    locked_static_prefix: int,
    static_bonus: float,
    static_rank_penalty: float,
    generated_penalty: float,
    overlap_bonus: float,
    generated_rank_penalty: float,
    static_log_count_scale: float,
    static_source_bonus: float,
) -> dict:
    return {
        "locked_static_prefix": int(locked_static_prefix),
        "static_bonus": float(static_bonus),
        "static_rank_penalty": float(static_rank_penalty),
        "generated_penalty": float(generated_penalty),
        "overlap_bonus": float(overlap_bonus),
        "generated_rank_penalty": float(generated_rank_penalty),
        "static_log_count_scale": float(static_log_count_scale),
        "static_source_bonus": float(static_source_bonus),
    }


def scored_union_policy_from_profile(profile: dict) -> dict:
    return {field: profile[field] for field in SCORED_UNION_POLICY_FIELDS}


def sliced_union(union: dict, indexes: np.ndarray) -> dict:
    return {
        "ids": union["ids"][indexes],
        "valid": union["valid"][indexes],
        "is_static": union["is_static"][indexes],
        "static_rank": union["static_rank"][indexes],
        "static_count": union["static_count"][indexes],
        "static_source_order": union["static_source_order"][indexes],
        "label_positions": union["label_positions"][indexes],
        "candidate_counts": union["candidate_counts"][indexes],
    }


def exported_union_model_scores(
    union: dict,
    token_id_rows: np.ndarray,
    topk_score_rows: np.ndarray,
    candidate_score_rows: np.ndarray,
    inputs: ExportInputSet,
) -> np.ndarray:
    scores = np.full(union["ids"].shape, -1.0e9, dtype=np.float32)
    for row_index in range(inputs.size):
        generated_scores: dict[int, float] = {}
        for token_id, score in zip(token_id_rows[row_index], topk_score_rows[row_index]):
            token_id = int(token_id)
            if token_id > 2:
                generated_scores[token_id] = max(
                    generated_scores.get(token_id, -1.0e9),
                    float(score),
                )

        for column_index, token_id in enumerate(union["ids"][row_index]):
            if not union["valid"][row_index, column_index]:
                continue
            token_id = int(token_id)
            candidate_scores = []
            static_rank = int(union["static_rank"][row_index, column_index])
            if 0 <= static_rank < candidate_score_rows.shape[1]:
                candidate_scores.append(float(candidate_score_rows[row_index, static_rank]))
            generated_score = generated_scores.get(token_id)
            if generated_score is not None:
                candidate_scores.append(generated_score)
            if candidate_scores:
                scores[row_index, column_index] = max(candidate_scores)
    return scores


def exported_union_feature_matrix(
    union: dict,
    token_id_rows: np.ndarray,
    topk_score_rows: np.ndarray,
    candidate_score_rows: np.ndarray,
    inputs: ExportInputSet,
) -> np.ndarray:
    model_scores = exported_union_model_scores(
        union,
        token_id_rows,
        topk_score_rows,
        candidate_score_rows,
        inputs,
    )
    generated_rank = generated_rank_matrix(union, token_id_rows)

    is_static = union["is_static"].astype(np.float32)
    is_generated = (generated_rank < 999).astype(np.float32)
    static_rank = union["static_rank"].astype(np.float32)
    generated_rank_f32 = generated_rank.astype(np.float32)
    features = np.stack(
        [
            np.where(union["valid"], model_scores, 0.0),
            is_static,
            np.where(union["is_static"], -static_rank, 0.0),
            is_generated,
            np.where(generated_rank < 999, -generated_rank_f32, 0.0),
            (is_static * is_generated).astype(np.float32),
        ],
        axis=2,
    ).astype(np.float32)
    return features


def generated_rank_matrix(union: dict, token_id_rows: np.ndarray) -> np.ndarray:
    generated_rank = np.full(union["ids"].shape, 999, dtype=np.int16)
    for row_index in range(union["ids"].shape[0]):
        ranks: dict[int, int] = {}
        for rank, token_id in enumerate(token_id_rows[row_index].tolist()):
            token_id = int(token_id)
            if token_id > 2 and token_id not in ranks:
                ranks[token_id] = rank
        for column_index, token_id in enumerate(union["ids"][row_index].tolist()):
            if union["valid"][row_index, column_index]:
                rank = ranks.get(int(token_id))
                if rank is not None:
                    generated_rank[row_index, column_index] = rank
    return generated_rank


def evaluate_learned_linear_union_policy(
    token_id_rows: np.ndarray,
    topk_score_rows: np.ndarray,
    candidate_score_rows: np.ndarray,
    inputs: ExportInputSet,
    epochs: int,
    learning_rate: float,
    l2: float,
    max_pairs: int,
    eval_mod: int,
    eval_remainder: int,
    seed: int,
) -> dict:
    union = build_union_candidate_matrix(token_id_rows, inputs)
    if union["ids"].shape[0] == 0:
        return {"enabled": False, "reason": "empty union"}
    eval_mod = max(2, int(eval_mod))
    eval_remainder = int(eval_remainder) % eval_mod
    indexes = np.arange(inputs.size)
    eval_indexes = indexes[indexes % eval_mod == eval_remainder]
    train_indexes = indexes[indexes % eval_mod != eval_remainder]
    if train_indexes.size == 0 or eval_indexes.size == 0:
        return {"enabled": False, "reason": "empty train/eval split"}

    features = exported_union_feature_matrix(
        union,
        token_id_rows,
        topk_score_rows,
        candidate_score_rows,
        inputs,
    )
    weights, training_report = train_pairwise_linear_union_weights(
        features,
        union["valid"],
        union["label_positions"],
        train_indexes,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        max_pairs=max_pairs,
        seed=seed,
    )
    scores = features @ weights
    eval_inputs = subset_inputs(inputs, eval_indexes)
    train_inputs = subset_inputs(inputs, train_indexes)
    eval_metrics = metrics_for_union_scores(
        scores[eval_indexes],
        union["label_positions"][eval_indexes],
        union["valid"][eval_indexes],
        eval_inputs,
    )
    train_metrics = metrics_for_union_scores(
        scores[train_indexes],
        union["label_positions"][train_indexes],
        union["valid"][train_indexes],
        train_inputs,
    )
    eval_static = evaluate_static_pool(eval_inputs)
    train_static = evaluate_static_pool(train_inputs)
    for prefix, metrics, static_metrics in (
        ("eval", eval_metrics, eval_static),
        ("train", train_metrics, train_static),
    ):
        for cutoff in REPORT_CUTOFFS:
            metrics[f"top{cutoff}_gain_vs_static"] = (
                metrics[f"top{cutoff}"] - static_metrics[f"top{cutoff}"]
            )
        metrics["mrr_gain_vs_static"] = metrics["mrr"] - static_metrics["mrr"]
        metrics["split"] = prefix
    return {
        "enabled": True,
        "candidate_scorer": "exported_graph_linear_fusion",
        "feature_names": list(LEARNED_UNION_FEATURE_NAMES),
        "weights": [float(value) for value in weights.tolist()],
        "train_size": int(train_indexes.size),
        "eval_size": int(eval_indexes.size),
        "eval_mod": eval_mod,
        "eval_remainder": eval_remainder,
        "training": training_report,
        "train_static_pool": pick_rank_metrics(train_static),
        "eval_static_pool": pick_rank_metrics(eval_static),
        "train": train_metrics,
        "eval": eval_metrics,
    }


def train_pairwise_linear_union_weights(
    features: np.ndarray,
    valid: np.ndarray,
    label_positions: np.ndarray,
    train_indexes: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    pair_diffs: list[np.ndarray] = []
    positive_rows = 0
    total_available_pairs = 0
    pair_budget = max(1, int(max_pairs)) if max_pairs > 0 else None
    rng = np.random.default_rng(seed)

    positive_row_items: list[tuple[int, int, np.ndarray]] = []
    for row_index in train_indexes.tolist():
        label_position = int(label_positions[row_index])
        if label_position < 0:
            continue
        negative_positions = np.nonzero(valid[row_index])[0]
        negative_positions = negative_positions[negative_positions != label_position]
        if negative_positions.size == 0:
            continue
        positive_rows += 1
        total_available_pairs += int(negative_positions.size)
        positive_row_items.append((row_index, label_position, negative_positions))

    if pair_budget is None or total_available_pairs <= pair_budget:
        for row_index, label_position, negative_positions in positive_row_items:
            positive = features[row_index, label_position]
            for negative_position in negative_positions.tolist():
                pair_diffs.append(positive - features[row_index, negative_position])
    else:
        pairs_per_row = max(1, pair_budget // max(1, len(positive_row_items)))
        remaining = pair_budget
        for row_index, label_position, negative_positions in positive_row_items:
            if remaining <= 0:
                break
            row_budget = min(int(negative_positions.size), pairs_per_row, remaining)
            if row_budget <= 0:
                continue
            if row_budget < negative_positions.size:
                sampled_positions = rng.choice(
                    negative_positions,
                    size=row_budget,
                    replace=False,
                )
            else:
                sampled_positions = negative_positions
            positive = features[row_index, label_position]
            for negative_position in sampled_positions.tolist():
                pair_diffs.append(positive - features[row_index, negative_position])
            remaining -= row_budget
    if not pair_diffs:
        return np.zeros((features.shape[2],), dtype=np.float32), {
            "positive_rows": 0,
            "pair_count": 0,
            "total_available_pair_count": total_available_pairs,
            "max_pairs": pair_budget,
            "epochs": 0,
            "final_loss": None,
        }

    torch.manual_seed(seed)
    diffs = torch.from_numpy(np.stack(pair_diffs).astype(np.float32))
    weights = torch.zeros((features.shape[2],), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        weights[0] = 1.0
        weights[1] = 1.0
        weights[2] = 0.25
        weights[3] = -1.0
        weights[4] = 0.01
        weights[5] = 0.5
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    batch_size = min(8192, diffs.shape[0])
    generator = torch.Generator().manual_seed(seed)
    final_loss = 0.0
    epochs = max(1, int(epochs))
    for _ in range(epochs):
        permutation = torch.randperm(diffs.shape[0], generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, diffs.shape[0], batch_size):
            indexes = permutation[start : start + batch_size]
            batch = diffs[indexes]
            optimizer.zero_grad(set_to_none=True)
            margins = batch @ weights
            loss = torch.nn.functional.softplus(-margins).mean()
            if l2 > 0:
                loss = loss + float(l2) * (weights * weights).mean()
            loss.backward()
            optimizer.step()
            batch_seen = int(indexes.numel())
            seen += batch_seen
            loss_sum += float(loss.detach()) * batch_seen
        final_loss = loss_sum / max(1, seen)
    return weights.detach().numpy().astype(np.float32), {
        "positive_rows": positive_rows,
        "pair_count": int(diffs.shape[0]),
        "total_available_pair_count": total_available_pairs,
        "max_pairs": pair_budget,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "final_loss": final_loss,
    }


def subset_inputs(inputs: ExportInputSet, indexes: np.ndarray) -> ExportInputSet:
    index_list = indexes.tolist()
    return ExportInputSet(
        contexts=inputs.contexts[indexes],
        candidate_ids=inputs.candidate_ids[indexes],
        candidate_rows=[inputs.candidate_rows[index] for index in index_list],
        candidate_counts=inputs.candidate_counts[indexes],
        candidate_source_order=inputs.candidate_source_order[indexes],
        labels=inputs.labels[indexes],
        source_ids=inputs.source_ids[indexes],
        source_names=inputs.source_names,
        total_targets=int(indexes.size),
        eligible_targets=int(indexes.size),
    )


def pick_rank_metrics(report: dict) -> dict:
    return {
        "top1": report["top1"],
        "top3": report["top3"],
        "top5": report["top5"],
        "top10": report["top10"],
        "mrr": report["mrr"],
    }


def build_union_candidate_matrix(token_id_rows: np.ndarray, inputs: ExportInputSet) -> dict:
    union_rows: list[list[int]] = []
    static_rank_rows: list[list[int]] = []
    static_count_rows: list[list[float]] = []
    static_source_order_rows: list[list[float]] = []
    label_positions: list[int] = []
    max_len = 0
    for row_index, (label, neural_row, static_row) in enumerate(
        zip(
            inputs.labels.tolist(),
            token_id_rows.tolist(),
            inputs.candidate_rows,
        )
    ):
        seen = set()
        union_row: list[int] = []
        static_ranks: list[int] = []
        static_counts: list[float] = []
        static_source_orders: list[float] = []
        for rank, token_id in enumerate(static_row):
            token_id = int(token_id)
            if token_id not in seen:
                seen.add(token_id)
                union_row.append(token_id)
                static_ranks.append(rank)
                static_counts.append(float(inputs.candidate_counts[row_index, rank]))
                static_source_orders.append(
                    float(inputs.candidate_source_order[row_index, rank])
                )
        for token_id in neural_row:
            token_id = int(token_id)
            if token_id > 2 and token_id not in seen:
                seen.add(token_id)
                union_row.append(token_id)
                static_ranks.append(-1)
                static_counts.append(0.0)
                static_source_orders.append(0.0)
        try:
            label_positions.append(union_row.index(int(label)))
        except ValueError:
            label_positions.append(-1)
        union_rows.append(union_row)
        static_rank_rows.append(static_ranks)
        static_count_rows.append(static_counts)
        static_source_order_rows.append(static_source_orders)
        max_len = max(max_len, len(union_row))

    ids = np.full((inputs.size, max_len), 0, dtype=np.int64)
    valid = np.zeros((inputs.size, max_len), dtype=bool)
    is_static = np.zeros((inputs.size, max_len), dtype=bool)
    static_rank = np.full((inputs.size, max_len), 999, dtype=np.int16)
    static_count = np.zeros((inputs.size, max_len), dtype=np.float32)
    static_source_order = np.zeros((inputs.size, max_len), dtype=np.float32)
    candidate_counts = np.zeros(inputs.size, dtype=np.int32)
    for row_index, (
        union_row,
        static_ranks,
        static_counts,
        static_source_orders,
    ) in enumerate(
        zip(
            union_rows,
            static_rank_rows,
            static_count_rows,
            static_source_order_rows,
        )
    ):
        row_len = len(union_row)
        candidate_counts[row_index] = row_len
        ids[row_index, :row_len] = union_row
        valid[row_index, :row_len] = True
        for candidate_index, rank in enumerate(static_ranks):
            if rank >= 0:
                is_static[row_index, candidate_index] = True
                static_rank[row_index, candidate_index] = rank
                static_count[row_index, candidate_index] = static_counts[candidate_index]
                static_source_order[row_index, candidate_index] = static_source_orders[
                    candidate_index
                ]
    return {
        "ids": ids,
        "valid": valid,
        "is_static": is_static,
        "static_rank": static_rank,
        "static_count": static_count,
        "static_source_order": static_source_order,
        "label_positions": np.asarray(label_positions, dtype=np.int64),
        "candidate_counts": candidate_counts,
    }


@torch.no_grad()
def score_union_candidates(
    model: NextWordLm,
    contexts: np.ndarray,
    candidate_ids: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    model.eval()
    for start in range(0, contexts.shape[0], batch_size):
        end = min(contexts.shape[0], start + batch_size)
        scores = score_candidate_pool(
            model,
            torch.from_numpy(contexts[start:end]),
            torch.from_numpy(candidate_ids[start:end]),
        )
        chunks.append(scores.numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def metrics_for_union_scores(
    scores: np.ndarray,
    label_positions: np.ndarray,
    valid: np.ndarray,
    inputs: ExportInputSet,
) -> dict:
    masked_scores = np.where(valid, scores, -1.0e9)
    valid_rows = label_positions >= 0
    ranks = np.zeros(inputs.size, dtype=np.int32)
    if np.any(valid_rows):
        row_indexes = np.nonzero(valid_rows)[0]
        label_scores = masked_scores[row_indexes, label_positions[valid_rows]]
        ranks[valid_rows] = (
            masked_scores[row_indexes] > label_scores[:, None]
        ).sum(axis=1) + 1

    hits = Counter()
    reciprocal_rank_sum = 0.0
    for rank in ranks.tolist():
        if rank > 0:
            reciprocal_rank_sum += 1.0 / rank
        for cutoff in REPORT_CUTOFFS:
            if rank > 0 and rank <= cutoff:
                hits[cutoff] += 1
    return rank_report(hits, reciprocal_rank_sum, inputs)


def merged_candidate_row(
    static_row: list[int],
    neural_row: list[int],
    visible_candidates: int,
    locked_static_prefix: int,
) -> list[int]:
    merged: list[int] = []
    locked = min(locked_static_prefix, len(static_row), visible_candidates)
    for token_id in static_row[:locked]:
        append_unique_token(merged, int(token_id), visible_candidates)
    for token_id in neural_row:
        append_unique_token(merged, int(token_id), visible_candidates)
        if len(merged) >= visible_candidates:
            return merged
    for token_id in static_row[locked:]:
        append_unique_token(merged, int(token_id), visible_candidates)
        if len(merged) >= visible_candidates:
            return merged
    return merged


def append_unique_token(output: list[int], token_id: int, limit: int) -> None:
    if len(output) < limit and token_id not in output:
        output.append(token_id)


def collect_topk_onnx_ids(session: ort.InferenceSession, inputs: ExportInputSet) -> np.ndarray:
    rows = [
        session.run(["token_ids"], {"contexts": inputs.contexts[index : index + 1]})[0]
        for index in range(inputs.size)
    ]
    return np.concatenate(rows, axis=0)


def collect_topk_coreml_ids(mlmodel, inputs: ExportInputSet) -> np.ndarray:
    rows = [
        mlmodel.predict({"contexts": inputs.contexts[index : index + 1].astype(np.int32)})[
            "token_ids"
        ]
        for index in range(inputs.size)
    ]
    return np.concatenate(rows, axis=0)


def collect_combined_onnx_outputs(
    session: ort.InferenceSession,
    inputs: ExportInputSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topk_scores = []
    token_ids = []
    candidate_scores = []
    for index in range(inputs.size):
        scores, ids, static_scores = session.run(
            ["topk_scores", "token_ids", "candidate_scores"],
            {
                "contexts": inputs.contexts[index : index + 1],
                "candidate_ids": inputs.candidate_ids[index : index + 1],
            },
        )
        topk_scores.append(scores)
        token_ids.append(ids)
        candidate_scores.append(static_scores)
    return (
        np.concatenate(topk_scores, axis=0),
        np.concatenate(token_ids, axis=0),
        np.concatenate(candidate_scores, axis=0),
    )


def collect_combined_coreml_outputs(
    mlmodel,
    inputs: ExportInputSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topk_scores = []
    token_ids = []
    candidate_scores = []
    for index in range(inputs.size):
        prediction = mlmodel.predict(
            {
                "contexts": inputs.contexts[index : index + 1].astype(np.int32),
                "candidate_ids": inputs.candidate_ids[index : index + 1].astype(np.int32),
            }
        )
        topk_scores.append(prediction["topk_scores"])
        token_ids.append(prediction["token_ids"])
        candidate_scores.append(prediction["candidate_scores"])
    return (
        np.concatenate(topk_scores, axis=0),
        np.concatenate(token_ids, axis=0),
        np.concatenate(candidate_scores, axis=0),
    )


def rank_report(hits: Counter, reciprocal_rank_sum: float, inputs: ExportInputSet) -> dict:
    eligible = max(1, inputs.size)
    total = max(1, inputs.total_targets)
    report = {
        "eligible_targets": inputs.size,
        "total_targets": inputs.total_targets,
        "mrr": reciprocal_rank_sum / eligible,
        "mrr_all_targets": reciprocal_rank_sum / total,
    }
    for cutoff in REPORT_CUTOFFS:
        report[f"top{cutoff}"] = hits[cutoff] / eligible
        report[f"top{cutoff}_all_targets"] = hits[cutoff] / total
    return report


def batch_window(index: int, batch_size: int, sample_size: int) -> tuple[int, int]:
    start = (index * batch_size) % sample_size
    end = min(start + batch_size, sample_size)
    if end - start < batch_size:
        start = 0
        end = batch_size
    return start, end


def make_session(path: Path, intra_op_threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def make_coreml_model(path: Path, compute_unit: str):
    import coremltools as ct

    compute_units = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    try:
        selected = compute_units[compute_unit.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported Core ML compute unit: {compute_unit}") from error
    return ct.models.MLModel(str(path), compute_units=selected)


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def parse_positive_int_list(raw: str, flag: str) -> tuple[int, ...]:
    values = parse_int_list(raw, flag)
    if any(value < 1 for value in values):
        raise SystemExit(f"{flag} values must be at least 1")
    return values


def parse_nonnegative_int_list(raw: str, flag: str) -> tuple[int, ...]:
    values = parse_int_list(raw, flag)
    if any(value < 0 for value in values):
        raise SystemExit(f"{flag} values must be non-negative")
    return values


def parse_float_list(raw: str, flag: str) -> tuple[float, ...]:
    values = tuple(
        float(value)
        for value in raw.split(",")
        if value.strip()
    )
    if not values:
        raise SystemExit(f"{flag} must contain at least one float")
    return values


def parse_int_list(raw: str, flag: str) -> tuple[int, ...]:
    values = tuple(
        int(value)
        for value in raw.split(",")
        if value.strip()
    )
    if not values:
        raise SystemExit(f"{flag} must contain at least one integer")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/autosuggest/corpus"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantized-output", type=Path)
    parser.add_argument("--coreml-output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--native-fixtures", type=Path,
                        help="Write up to 4096 verified input cases for the native Apple benchmark")
    parser.add_argument(
        "--export-kind",
        choices=("candidate-scorer", "full-vocab-topk", "full-vocab-topk-scorer"),
        default="candidate-scorer",
    )
    parser.add_argument("--pool-k", type=int, default=16)
    parser.add_argument("--top-k-output", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-sentences-per-source", type=int)
    parser.add_argument("--max-examples-per-source", type=int, default=1_000)
    parser.add_argument("--benchmark-iterations", type=int, default=2_000)
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--compare-samples", type=int, default=256)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument(
        "--merge-visible-candidates",
        default="5,10",
        help="Comma-separated visible suggestion counts for full-vocab top-k merge evaluation.",
    )
    parser.add_argument(
        "--merge-locked-static-prefixes",
        default="1",
        help="Comma-separated static prefix lengths for full-vocab top-k merge evaluation.",
    )
    parser.add_argument(
        "--scored-union-static-bonuses",
        default="0,0.5,1,2,4",
        help="Comma-separated static membership bonuses for scored union policy evaluation.",
    )
    parser.add_argument(
        "--scored-union-static-rank-penalties",
        default="0,0.25,0.5,1,2,4",
        help="Comma-separated static rank penalties for scored union policy evaluation.",
    )
    parser.add_argument(
        "--scored-union-generated-penalties",
        default="0,2,4,8",
        help="Comma-separated generated-only penalties for scored union policy evaluation.",
    )
    parser.add_argument(
        "--scored-union-overlap-bonuses",
        default="0",
        help="Comma-separated bonuses for candidates present in both static and generated pools.",
    )
    parser.add_argument(
        "--scored-union-generated-rank-penalties",
        default="0",
        help="Comma-separated rank penalties for generated candidates.",
    )
    parser.add_argument(
        "--scored-union-static-log-count-scales",
        default="0",
        help="Comma-separated scales for log1p(static candidate count).",
    )
    parser.add_argument(
        "--scored-union-static-source-bonuses",
        default="0",
        help="Comma-separated bonuses per static n-gram source order.",
    )
    parser.add_argument(
        "--scored-union-locked-static-prefixes",
        default="1,2,3",
        help="Comma-separated locked static prefixes for scored union policy evaluation.",
    )
    parser.add_argument("--scored-union-top-profiles", type=int, default=20)
    parser.add_argument(
        "--scored-union-selection-eval-mod",
        type=int,
        default=5,
        help=(
            "Modulo split for scored-union policy selection. Values below 2 disable "
            "split selection; default holds out 1/mod rows for independent eval."
        ),
    )
    parser.add_argument(
        "--scored-union-selection-eval-remainder",
        type=int,
        default=0,
        help="Remainder held out for scored-union policy eval.",
    )
    parser.add_argument("--learned-union-epochs", type=int, default=0)
    parser.add_argument("--learned-union-learning-rate", type=float, default=0.05)
    parser.add_argument("--learned-union-l2", type=float, default=0.0001)
    parser.add_argument(
        "--learned-union-max-pairs",
        type=int,
        default=2_000_000,
        help=(
            "Maximum pairwise ranking examples for learned union training. "
            "Use 0 to keep every available pair."
        ),
    )
    parser.add_argument("--learned-union-eval-mod", type=int, default=5)
    parser.add_argument("--learned-union-eval-remainder", type=int, default=0)
    parser.add_argument("--learned-union-seed", type=int, default=13)
    parser.add_argument("--coreml-target", default="ios17", choices=("ios16", "ios17", "ios18"))
    parser.add_argument("--coreml-precision", default="float16", choices=("float16", "float32"))
    parser.add_argument(
        "--coreml-compute-unit",
        default="all",
        choices=("all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"),
    )
    parser.add_argument(
        "--rank-penalties",
        default="0,0.25,0.5,0.75,1,1.5,2,3,4,6,8",
        help="Comma-separated static-rank penalties for ONNX reranking evaluation.",
    )
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    if args.pool_k < 1:
        raise SystemExit("--pool-k must be at least 1")
    if args.top_k_output < 1:
        raise SystemExit("--top-k-output must be at least 1")
    rank_penalties = tuple(
        float(value)
        for value in args.rank_penalties.split(",")
        if value.strip()
    )
    if not rank_penalties:
        rank_penalties = DEFAULT_RANK_PENALTIES
    merge_visible_candidates = parse_positive_int_list(
        args.merge_visible_candidates,
        "--merge-visible-candidates",
    )
    merge_locked_static_prefixes = parse_nonnegative_int_list(
        args.merge_locked_static_prefixes,
        "--merge-locked-static-prefixes",
    )
    scored_union_static_bonuses = parse_float_list(
        args.scored_union_static_bonuses,
        "--scored-union-static-bonuses",
    )
    scored_union_static_rank_penalties = parse_float_list(
        args.scored_union_static_rank_penalties,
        "--scored-union-static-rank-penalties",
    )
    scored_union_generated_penalties = parse_float_list(
        args.scored_union_generated_penalties,
        "--scored-union-generated-penalties",
    )
    scored_union_overlap_bonuses = parse_float_list(
        args.scored_union_overlap_bonuses,
        "--scored-union-overlap-bonuses",
    )
    scored_union_generated_rank_penalties = parse_float_list(
        args.scored_union_generated_rank_penalties,
        "--scored-union-generated-rank-penalties",
    )
    scored_union_static_log_count_scales = parse_float_list(
        args.scored_union_static_log_count_scales,
        "--scored-union-static-log-count-scales",
    )
    scored_union_static_source_bonuses = parse_float_list(
        args.scored_union_static_source_bonuses,
        "--scored-union-static-source-bonuses",
    )
    scored_union_locked_static_prefixes = parse_nonnegative_int_list(
        args.scored_union_locked_static_prefixes,
        "--scored-union-locked-static-prefixes",
    )
    data_provenance = evaluation_provenance(args.artifact, args.corpus_dir)
    checkpoint_provenance = None
    if data_provenance["status"] == "partition_verified":
        # This exporter runs ranking-policy comparisons; test must remain reserved
        # for the separate frozen evaluator, even when exporting a fixed graph.
        load_partition(args.corpus_dir, allowed=("validation",))
        checkpoint_provenance = neural_training_provenance(
            args.artifact, args.corpus_dir.parent / "train", args.corpus_dir,
        )
    if args.skip_sentences_per_source is None:
        args.skip_sentences_per_source = 0 if checkpoint_provenance else 100_000
    checkpoint_hash = sha256_file(args.checkpoint)
    model, config = load_model(args.checkpoint, provenance=checkpoint_provenance,
                               retrieval_model=args.artifact)
    lm = NgramLm(args.artifact)
    if config["vocab_size"] != lm.vocab_size:
        raise ValueError("checkpoint vocabulary size differs from retrieval artifact")
    context_window = int(config["context_window"])
    if args.export_kind == "candidate-scorer":
        export_onnx(model, args.output, context_window, args.pool_k, args.opset)
    elif args.export_kind == "full-vocab-topk":
        export_topk_onnx(model, args.output, context_window, args.top_k_output, args.opset)
    else:
        export_combined_onnx(
            model,
            args.output,
            context_window,
            args.pool_k,
            args.top_k_output,
            args.opset,
        )
    coreml_path = None
    if args.coreml_output is not None:
        coreml_path = args.coreml_output
        if args.export_kind == "candidate-scorer":
            export_coreml(
                model,
                coreml_path,
                context_window,
                args.pool_k,
                args.coreml_target,
                args.coreml_precision,
            )
        elif args.export_kind == "full-vocab-topk":
            export_topk_coreml(
                model,
                coreml_path,
                context_window,
                args.top_k_output,
                args.coreml_target,
                args.coreml_precision,
            )
        else:
            export_combined_coreml(
                model,
                coreml_path,
                context_window,
                args.pool_k,
                args.top_k_output,
                args.coreml_target,
                args.coreml_precision,
            )

    quantized_path = None
    if not args.no_quantize:
        quantized_path = args.quantized_output or args.output.with_suffix(".int8.onnx")
        quantize_onnx(args.output, quantized_path)

    inputs = collect_real_inputs(
        lm,
        args.corpus_dir,
        context_window,
        args.pool_k,
        args.skip_sentences_per_source,
        args.max_examples_per_source,
    )
    session = make_session(args.output, args.intra_op_threads)
    report = {
        "data_provenance": data_provenance,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": str(args.checkpoint),
        "artifact": str(args.artifact),
        "model": {
            "architecture": config["architecture"],
            "context_window": context_window,
            "embedding_dim": int(config["embedding_dim"]),
            "hidden_dim": int(config["hidden_dim"]),
            "parameter_count": parameter_count(model),
            "fp32_parameter_bytes": parameter_count(model) * 4,
        },
        "export": {
            "onnx": str(args.output),
            "onnx_bytes": args.output.stat().st_size,
            "opset": args.opset,
            "kind": args.export_kind,
            "pool_k": args.pool_k,
            "top_k_output": args.top_k_output,
            "fixed_batch": 1,
        },
        "verification": {},
        "quality": {
            "static_pool": evaluate_static_pool(inputs),
        },
        "benchmark": {},
    }
    if args.export_kind == "candidate-scorer":
        report["verification"]["onnx_vs_pytorch"] = compare_with_pytorch(
            model,
            session,
            inputs,
            args.compare_samples,
        )
        report["quality"]["onnx_rerank"] = evaluate_onnx_rerank(
            session,
            inputs,
            rank_penalties,
            lock_first=False,
        )
        report["quality"]["onnx_rerank_locked_first"] = evaluate_onnx_rerank(
            session,
            inputs,
            rank_penalties,
            lock_first=True,
        )
        report["benchmark"]["onnx"] = benchmark_onnx(
            session,
            inputs,
            args.benchmark_iterations,
            args.benchmark_batch_size,
        )
    elif args.export_kind == "full-vocab-topk":
        report["verification"]["onnx_vs_pytorch"] = compare_topk_onnx_with_pytorch(
            model,
            session,
            inputs,
            args.compare_samples,
            args.top_k_output,
        )
        onnx_token_ids = collect_topk_onnx_ids(session, inputs)
        report["quality"]["onnx_topk"] = evaluate_topk_candidates(
            onnx_token_ids,
            inputs,
        )
        report["quality"]["onnx_merged"] = evaluate_topk_merge_policies(
            onnx_token_ids,
            inputs,
            merge_visible_candidates,
            merge_locked_static_prefixes,
        )
        report["quality"]["onnx_scored_union"] = evaluate_scored_union_policies(
            model,
            onnx_token_ids,
            inputs,
            scored_union_static_bonuses,
            scored_union_static_rank_penalties,
            scored_union_generated_penalties,
            scored_union_locked_static_prefixes,
            overlap_bonuses=scored_union_overlap_bonuses,
            generated_rank_penalties=scored_union_generated_rank_penalties,
            static_log_count_scales=scored_union_static_log_count_scales,
            static_source_bonuses=scored_union_static_source_bonuses,
            top_profiles=args.scored_union_top_profiles,
        )
        report["benchmark"]["onnx"] = benchmark_topk_onnx(
            session,
            inputs,
            args.benchmark_iterations,
            args.benchmark_batch_size,
        )
    else:
        report["verification"]["onnx_vs_pytorch"] = compare_combined_onnx_with_pytorch(
            model,
            session,
            inputs,
            args.compare_samples,
            args.top_k_output,
        )
        onnx_topk_scores, onnx_token_ids, onnx_candidate_scores = collect_combined_onnx_outputs(
            session,
            inputs,
        )
        report["quality"]["onnx_topk"] = evaluate_topk_candidates(
            onnx_token_ids,
            inputs,
        )
        report["quality"]["onnx_merged"] = evaluate_topk_merge_policies(
            onnx_token_ids,
            inputs,
            merge_visible_candidates,
            merge_locked_static_prefixes,
        )
        report["quality"]["onnx_scored_union"] = evaluate_exported_scored_union_policies(
            onnx_token_ids,
            onnx_topk_scores,
            onnx_candidate_scores,
            inputs,
            scored_union_static_bonuses,
            scored_union_static_rank_penalties,
            scored_union_generated_penalties,
            scored_union_locked_static_prefixes,
            overlap_bonuses=scored_union_overlap_bonuses,
            generated_rank_penalties=scored_union_generated_rank_penalties,
            static_log_count_scales=scored_union_static_log_count_scales,
            static_source_bonuses=scored_union_static_source_bonuses,
            top_profiles=args.scored_union_top_profiles,
        )
        report["quality"]["onnx_scored_union_split"] = (
            evaluate_exported_scored_union_split_selection(
                onnx_token_ids,
                onnx_topk_scores,
                onnx_candidate_scores,
                inputs,
                scored_union_static_bonuses,
                scored_union_static_rank_penalties,
                scored_union_generated_penalties,
                scored_union_locked_static_prefixes,
                overlap_bonuses=scored_union_overlap_bonuses,
                generated_rank_penalties=scored_union_generated_rank_penalties,
                static_log_count_scales=scored_union_static_log_count_scales,
                static_source_bonuses=scored_union_static_source_bonuses,
                top_profiles=args.scored_union_top_profiles,
                eval_mod=args.scored_union_selection_eval_mod,
                eval_remainder=args.scored_union_selection_eval_remainder,
            )
        )
        if args.learned_union_epochs > 0:
            report["quality"]["onnx_learned_linear_union"] = (
                evaluate_learned_linear_union_policy(
                    onnx_token_ids,
                    onnx_topk_scores,
                    onnx_candidate_scores,
                    inputs,
                    epochs=args.learned_union_epochs,
                    learning_rate=args.learned_union_learning_rate,
                    l2=args.learned_union_l2,
                    max_pairs=args.learned_union_max_pairs,
                    eval_mod=args.learned_union_eval_mod,
                    eval_remainder=args.learned_union_eval_remainder,
                    seed=args.learned_union_seed,
                )
            )
        report["benchmark"]["onnx"] = benchmark_combined_onnx(
            session,
            inputs,
            args.benchmark_iterations,
            args.benchmark_batch_size,
        )
    if quantized_path is not None:
        quantized_session = make_session(quantized_path, args.intra_op_threads)
        report["export"]["quantized_onnx"] = str(quantized_path)
        report["export"]["quantized_onnx_bytes"] = quantized_path.stat().st_size
        if args.export_kind == "candidate-scorer":
            report["verification"]["quantized_vs_pytorch"] = compare_with_pytorch(
                model,
                quantized_session,
                inputs,
                args.compare_samples,
            )
            report["quality"]["quantized_rerank"] = evaluate_onnx_rerank(
                quantized_session,
                inputs,
                rank_penalties,
                lock_first=False,
            )
            report["quality"]["quantized_rerank_locked_first"] = evaluate_onnx_rerank(
                quantized_session,
                inputs,
                rank_penalties,
                lock_first=True,
            )
            report["benchmark"]["quantized_onnx"] = benchmark_onnx(
                quantized_session,
                inputs,
                args.benchmark_iterations,
                args.benchmark_batch_size,
            )
        elif args.export_kind == "full-vocab-topk":
            report["verification"]["quantized_vs_pytorch"] = (
                compare_topk_onnx_with_pytorch(
                    model,
                    quantized_session,
                    inputs,
                    args.compare_samples,
                    args.top_k_output,
                )
            )
            quantized_token_ids = collect_topk_onnx_ids(quantized_session, inputs)
            report["quality"]["quantized_topk"] = evaluate_topk_candidates(
                quantized_token_ids,
                inputs,
            )
            report["quality"]["quantized_merged"] = evaluate_topk_merge_policies(
                quantized_token_ids,
                inputs,
                merge_visible_candidates,
                merge_locked_static_prefixes,
            )
            report["quality"]["quantized_scored_union"] = evaluate_scored_union_policies(
                model,
                quantized_token_ids,
                inputs,
                scored_union_static_bonuses,
                scored_union_static_rank_penalties,
                scored_union_generated_penalties,
                scored_union_locked_static_prefixes,
                overlap_bonuses=scored_union_overlap_bonuses,
                generated_rank_penalties=scored_union_generated_rank_penalties,
                static_log_count_scales=scored_union_static_log_count_scales,
                static_source_bonuses=scored_union_static_source_bonuses,
                top_profiles=args.scored_union_top_profiles,
            )
            report["benchmark"]["quantized_onnx"] = benchmark_topk_onnx(
                quantized_session,
                inputs,
                args.benchmark_iterations,
                args.benchmark_batch_size,
            )
        else:
            report["verification"]["quantized_vs_pytorch"] = (
                compare_combined_onnx_with_pytorch(
                    model,
                    quantized_session,
                    inputs,
                    args.compare_samples,
                    args.top_k_output,
                )
            )
            (
                quantized_topk_scores,
                quantized_token_ids,
                quantized_candidate_scores,
            ) = collect_combined_onnx_outputs(quantized_session, inputs)
            report["quality"]["quantized_topk"] = evaluate_topk_candidates(
                quantized_token_ids,
                inputs,
            )
            report["quality"]["quantized_merged"] = evaluate_topk_merge_policies(
                quantized_token_ids,
                inputs,
                merge_visible_candidates,
                merge_locked_static_prefixes,
            )
            report["quality"]["quantized_scored_union"] = (
                evaluate_exported_scored_union_policies(
                    quantized_token_ids,
                    quantized_topk_scores,
                    quantized_candidate_scores,
                    inputs,
                    scored_union_static_bonuses,
                    scored_union_static_rank_penalties,
                    scored_union_generated_penalties,
                    scored_union_locked_static_prefixes,
                    overlap_bonuses=scored_union_overlap_bonuses,
                    generated_rank_penalties=scored_union_generated_rank_penalties,
                    static_log_count_scales=scored_union_static_log_count_scales,
                    static_source_bonuses=scored_union_static_source_bonuses,
                    top_profiles=args.scored_union_top_profiles,
                )
            )
            report["quality"]["quantized_scored_union_split"] = (
                evaluate_exported_scored_union_split_selection(
                    quantized_token_ids,
                    quantized_topk_scores,
                    quantized_candidate_scores,
                    inputs,
                    scored_union_static_bonuses,
                    scored_union_static_rank_penalties,
                    scored_union_generated_penalties,
                    scored_union_locked_static_prefixes,
                    overlap_bonuses=scored_union_overlap_bonuses,
                    generated_rank_penalties=scored_union_generated_rank_penalties,
                    static_log_count_scales=scored_union_static_log_count_scales,
                    static_source_bonuses=scored_union_static_source_bonuses,
                    top_profiles=args.scored_union_top_profiles,
                    eval_mod=args.scored_union_selection_eval_mod,
                    eval_remainder=args.scored_union_selection_eval_remainder,
                )
            )
            if args.learned_union_epochs > 0:
                report["quality"]["quantized_learned_linear_union"] = (
                    evaluate_learned_linear_union_policy(
                        quantized_token_ids,
                        quantized_topk_scores,
                        quantized_candidate_scores,
                        inputs,
                        epochs=args.learned_union_epochs,
                        learning_rate=args.learned_union_learning_rate,
                        l2=args.learned_union_l2,
                        max_pairs=args.learned_union_max_pairs,
                        eval_mod=args.learned_union_eval_mod,
                        eval_remainder=args.learned_union_eval_remainder,
                        seed=args.learned_union_seed,
                    )
                )
            report["benchmark"]["quantized_onnx"] = benchmark_combined_onnx(
                quantized_session,
                inputs,
                args.benchmark_iterations,
                args.benchmark_batch_size,
            )
    if coreml_path is not None:
        coreml_model = make_coreml_model(coreml_path, args.coreml_compute_unit)
        report["export"]["coreml"] = str(coreml_path)
        report["export"]["coreml_bytes"] = directory_size(coreml_path)
        report["export"]["coreml_target"] = args.coreml_target
        report["export"]["coreml_precision"] = args.coreml_precision
        report["export"]["coreml_compute_unit"] = args.coreml_compute_unit
        if args.export_kind == "candidate-scorer":
            report["verification"]["coreml_vs_pytorch"] = compare_coreml_with_pytorch(
                model,
                coreml_model,
                inputs,
                args.compare_samples,
            )
            report["benchmark"]["coreml"] = benchmark_coreml(
                coreml_model,
                inputs,
                args.benchmark_iterations,
                args.benchmark_batch_size,
            )
        elif args.export_kind == "full-vocab-topk":
            report["verification"]["coreml_vs_pytorch"] = compare_topk_coreml_with_pytorch(
                model,
                coreml_model,
                inputs,
                args.compare_samples,
                args.top_k_output,
            )
            coreml_token_ids = collect_topk_coreml_ids(coreml_model, inputs)
            report["quality"]["coreml_topk"] = evaluate_topk_candidates(
                coreml_token_ids,
                inputs,
            )
            report["quality"]["coreml_merged"] = evaluate_topk_merge_policies(
                coreml_token_ids,
                inputs,
                merge_visible_candidates,
                merge_locked_static_prefixes,
            )
            report["quality"]["coreml_scored_union"] = evaluate_scored_union_policies(
                model,
                coreml_token_ids,
                inputs,
                scored_union_static_bonuses,
                scored_union_static_rank_penalties,
                scored_union_generated_penalties,
                scored_union_locked_static_prefixes,
                overlap_bonuses=scored_union_overlap_bonuses,
                generated_rank_penalties=scored_union_generated_rank_penalties,
                static_log_count_scales=scored_union_static_log_count_scales,
                static_source_bonuses=scored_union_static_source_bonuses,
                top_profiles=args.scored_union_top_profiles,
            )
            report["benchmark"]["coreml"] = benchmark_topk_coreml(
                coreml_model,
                inputs,
                args.benchmark_iterations,
                args.benchmark_batch_size,
            )
        else:
            report["verification"]["coreml_vs_pytorch"] = compare_combined_coreml_with_pytorch(
                model,
                coreml_model,
                inputs,
                args.compare_samples,
                args.top_k_output,
            )
            coreml_topk_scores, coreml_token_ids, coreml_candidate_scores = (
                collect_combined_coreml_outputs(coreml_model, inputs)
            )
            report["quality"]["coreml_topk"] = evaluate_topk_candidates(
                coreml_token_ids,
                inputs,
            )
            report["quality"]["coreml_merged"] = evaluate_topk_merge_policies(
                coreml_token_ids,
                inputs,
                merge_visible_candidates,
                merge_locked_static_prefixes,
            )
            report["quality"]["coreml_scored_union"] = evaluate_exported_scored_union_policies(
                coreml_token_ids,
                coreml_topk_scores,
                coreml_candidate_scores,
                inputs,
                scored_union_static_bonuses,
                scored_union_static_rank_penalties,
                scored_union_generated_penalties,
                scored_union_locked_static_prefixes,
                overlap_bonuses=scored_union_overlap_bonuses,
                generated_rank_penalties=scored_union_generated_rank_penalties,
                static_log_count_scales=scored_union_static_log_count_scales,
                static_source_bonuses=scored_union_static_source_bonuses,
                top_profiles=args.scored_union_top_profiles,
            )
            report["quality"]["coreml_scored_union_split"] = (
                evaluate_exported_scored_union_split_selection(
                    coreml_token_ids,
                    coreml_topk_scores,
                    coreml_candidate_scores,
                    inputs,
                    scored_union_static_bonuses,
                    scored_union_static_rank_penalties,
                    scored_union_generated_penalties,
                    scored_union_locked_static_prefixes,
                    overlap_bonuses=scored_union_overlap_bonuses,
                    generated_rank_penalties=scored_union_generated_rank_penalties,
                    static_log_count_scales=scored_union_static_log_count_scales,
                    static_source_bonuses=scored_union_static_source_bonuses,
                    top_profiles=args.scored_union_top_profiles,
                    eval_mod=args.scored_union_selection_eval_mod,
                    eval_remainder=args.scored_union_selection_eval_remainder,
                )
            )
            if args.learned_union_epochs > 0:
                report["quality"]["coreml_learned_linear_union"] = (
                    evaluate_learned_linear_union_policy(
                        coreml_token_ids,
                        coreml_topk_scores,
                        coreml_candidate_scores,
                        inputs,
                        epochs=args.learned_union_epochs,
                        learning_rate=args.learned_union_learning_rate,
                        l2=args.learned_union_l2,
                        max_pairs=args.learned_union_max_pairs,
                        eval_mod=args.learned_union_eval_mod,
                        eval_remainder=args.learned_union_eval_remainder,
                        seed=args.learned_union_seed,
                    )
                )
            report["benchmark"]["coreml"] = benchmark_combined_coreml(
                coreml_model,
                inputs,
                args.benchmark_iterations,
                args.benchmark_batch_size,
            )
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed during export; use an immutable snapshot")
    if args.native_fixtures:
        if inputs.size == 0:
            raise ValueError("cannot create native fixtures from an empty collection")
        indexes = np.linspace(0, inputs.size - 1, min(inputs.size, 4096), dtype=np.int64)
        fixtures = {
            "version": 1, "vocabularySize": lm.vocab_size,
            "contexts": inputs.contexts[indexes].tolist(),
            "candidates": inputs.candidate_ids[indexes].tolist(),
            "targets": inputs.labels[indexes].tolist(),
            "targetScope": "known-vocabulary validation subset; not all-target accuracy",
            "checkpointSHA256": checkpoint_hash, "retrievalSHA256": sha256_file(args.artifact),
        }
        args.native_fixtures.parent.mkdir(parents=True, exist_ok=True)
        args.native_fixtures.write_text(json.dumps(fixtures, separators=(",", ":")) + "\n", encoding="utf-8")
        report["native_fixtures"] = {"path": str(args.native_fixtures), "cases": len(indexes),
                                     "sha256": sha256_file(args.native_fixtures)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
