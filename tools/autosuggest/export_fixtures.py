"""Validation-only export fixtures that cannot cross document padding/boundaries."""

from __future__ import annotations

import numpy as np

from tools.corpus.provenance import sha256_file


class ValidationSequences:
    def __init__(self, root, receipt, sequence_length):
        if (root / ".building").exists():
            raise ValueError("incomplete validation data")
        path = root / "validation.u16"
        if sha256_file(path) != receipt["partitions"]["validation"]["sha256"]:
            raise ValueError("validation data integrity failure")
        self.stream = np.memmap(path, dtype="<u2", mode="r")
        self.blocks = None
        self.eligible = {}
        if receipt.get("kind") == "obadh-document-training-data":
            if receipt["sequence_length"] != sequence_length:
                raise ValueError("document/model context mismatch")
            records = [
                r for r in receipt["files"] if r["path"] == "validation.targets.u16"
            ]
            if len(records) != 1:
                raise ValueError("missing validation target counts")
            record = records[0]
            counts_path = root / record["path"]
            if (
                counts_path.stat().st_size != record["bytes"]
                or sha256_file(counts_path) != record["sha256"]
            ):
                raise ValueError("validation counts integrity failure")
            self.counts = np.memmap(counts_path, dtype="<u2", mode="r")
            self.blocks = self.stream.reshape(-1, sequence_length + 1)
            if (
                len(self.counts) != len(self.blocks)
                or np.any(self.counts < 1)
                or np.any(self.counts > sequence_length)
            ):
                raise ValueError("invalid document validation dimensions")

    def sample(self, rng, *, maximum_prefix, continuation):
        if maximum_prefix < 1 or continuation < 1:
            raise ValueError("positive prefix and continuation required")
        if self.blocks is None:
            source = self.stream
            valid = len(source)
        else:
            if continuation not in self.eligible:
                self.eligible[continuation] = np.flatnonzero(
                    self.counts >= continuation
                )
            eligible = self.eligible[continuation]
            if not len(eligible):
                raise ValueError("no validation block can fit continuation")
            block = int(rng.choice(eligible))
            source = self.blocks[block]
            valid = int(self.counts[block]) + 1
        upper = min(maximum_prefix, valid - continuation)
        if upper < 1:
            raise ValueError("insufficient validation context")
        length = int(rng.integers(1, upper + 1))
        start = int(rng.integers(0, valid - length - continuation + 1))
        sequence = np.asarray(
            source[start : start + length + continuation], dtype=np.int32
        )
        if np.any(sequence == 0):
            raise ValueError("validation fixture contains padding as text")
        return sequence, length
