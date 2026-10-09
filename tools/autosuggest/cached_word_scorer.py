"""Reference host for measuring complete-word decoding with exported KV graphs.

This is a Python evaluation host, not the production keyboard runtime. Prefill
is shared by the beam; every continuation owns its logical cache. Calls accept
and return named NumPy arrays, independently of Core ML or ONNX. Each request
starts fresh, so model, field and cursor changes cannot reuse stale state.
"""

from __future__ import annotations
import numpy as np

from tools.autosuggest.subword_cache import cache_shape


class CachedWordScorer:
    def __init__(self, config, prefill, step, *, cache_dtype=np.float32, max_beam=16):
        if not 1 <= max_beam <= 64 or np.dtype(cache_dtype) not in (
            np.dtype(np.float16), np.dtype(np.float32)
        ):
            raise ValueError("invalid cache host limits")
        self.config = config
        self.prefill = prefill
        self.step = step
        self.cache_dtype = np.dtype(cache_dtype)
        self.max_beam = max_beam
        self.states = {}
        self.initial = None
        self.depth = None
        self.model_calls = 0

    def checked(self, result):
        logits = np.asarray(result["logits"])
        keys = np.asarray(result["next_keys"])
        values = np.asarray(result["next_values"])
        if (
            logits.shape != (1, self.config.vocab_size)
            or keys.shape != cache_shape(self.config)
            or values.shape != keys.shape
            or any(a.dtype not in (np.dtype(np.float16), np.dtype(np.float32)) for a in (logits, keys, values))
            or not all(np.isfinite(a).all() for a in (logits, keys, values))
        ):
            raise ValueError("invalid cache graph output")
        # Backends may recycle output buffers on their next call. Independent
        # copies prevent a sibling branch from silently changing a parent's KV.
        # Core ML's Python bridge may promote a declared FP16 output to FP32.
        # Normalize explicitly to the next graph's input contract; reject
        # overflow rather than allowing a finite FP32 value to become Inf.
        with np.errstate(over="ignore", invalid="ignore"):
            keys = keys.astype(self.cache_dtype, copy=True)
            values = values.astype(self.cache_dtype, copy=True)
        if not np.isfinite(keys).all() or not np.isfinite(values).all():
            raise ValueError("cache output overflows input dtype")
        return logits.copy(), keys, values

    def begin(self, ids):
        self.states = {}
        self.initial = None
        self.depth = None
        self.model_calls = 0
        if (
            not ids or len(ids) > self.config.sequence_length
            or any(type(i) is not int or not 0 <= i < self.config.vocab_size for i in ids)
        ):
            raise ValueError("invalid prefill tokens")
        padded = np.zeros((1, self.config.sequence_length), dtype=np.int32)
        padded[0, :len(ids)] = ids
        logits, keys, values = self.checked(self.prefill({
            "tokens": padded,
            "last_index": np.array([len(ids) - 1], dtype=np.int32),
        }))
        self.length = len(ids)
        self.initial = logits
        self.states = {(): (keys, values)}
        self.depth = 0
        self.model_calls = 1

    def score(self, sequences):
        if self.depth is None or not 1 <= len(sequences) <= self.max_beam:
            raise ValueError("inactive request or invalid beam size")
        paths = [tuple(s) for s in sequences]
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate beam path")
        if self.initial is not None:
            if paths != [()]:
                raise ValueError("first score must request the prefill logits")
            logits, self.initial = self.initial, None
            return logits
        position = self.length + self.depth
        if position >= self.config.sequence_length:
            raise ValueError("cache capacity exceeded; a new prefill is required")
        if any(
            len(p) != self.depth + 1 or p[:-1] not in self.states
            or type(p[-1]) is not int or not 0 <= p[-1] < self.config.vocab_size
            for p in paths
        ):
            raise ValueError("beam parent, token or depth mismatch")
        next_states = {}
        scores = []
        for path in paths:
            keys, values = self.states[path[:-1]]
            logits, next_keys, next_values = self.checked(self.step({
                "token": np.array([[path[-1]]], dtype=np.int32),
                "position": np.array([position], dtype=np.int32),
                "key_cache": keys,
                "value_cache": values,
            }))
            next_states[path] = (next_keys, next_values)
            scores.append(logits)
            self.model_calls += 1
        self.states = next_states
        self.depth += 1
        return np.concatenate(scores, axis=0)

    def close(self):
        self.states = {}
        self.initial = None
        self.depth = None
