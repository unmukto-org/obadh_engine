"""Mix verified teacher labels across derivations of the same corpus split."""

import gzip
import json
from collections import defaultdict
import numpy as np
import torch

from tools.correction.data import PairData


class KindBalancedPairData(PairData):
    """Error-kind sampling with the original identity curriculum.

    The default square-root weighting increases exposure to rare tasks without
    equalizing categories of very different sizes. Exponent zero explicitly
    requests uniform categories for a controlled ablation. Validation keeps its
    original sampling distribution.
    """

    def __init__(self, root, exponent=.5):
        if exponent not in (0., .5):
            raise ValueError("unsupported error-kind sampling exponent")
        super().__init__(root)
        pools = defaultdict(list)
        arrays = self.arrays["train"]
        count = 0
        for i, row in enumerate(records(self.root, "train")):
            if i >= len(arrays["identity"]) or row["mode"] != arrays["mode"][i] or bool(row["source_text"] == row["target"]) != bool(arrays["identity"][i]):
                raise ValueError("base metadata/array alignment mismatch")
            if not arrays["identity"][i]:
                pools[row["kind"]].append(i)
            count += 1
        if count != len(arrays["identity"]) or not pools:
            raise ValueError("base metadata/array count mismatch")
        self.kind_names = sorted(pools)
        self.error_kinds = [np.asarray(pools[k], dtype=np.int64) for k in self.kind_names]
        weights = np.asarray([len(pool) for pool in self.error_kinds], dtype=np.float64) ** exponent
        self.kind_probabilities = torch.tensor(weights / weights.sum(), dtype=torch.float64)
        self.sampling_contract = dict(strategy="error-kind-counts-to-exponent", exponent=exponent, kinds=[
            dict(kind=name, rows=len(pool), probability=float(probability))
            for name, pool, probability in zip(self.kind_names, self.error_kinds, self.kind_probabilities)
        ])

    def sample(self, split, batch, identity_fraction, generator):
        if split != "train":
            return super().sample(split, batch, identity_fraction, generator)
        clean = round(batch * identity_fraction)
        pool = self.pools[split][1]
        chosen = [pool[torch.randint(len(pool), (clean,), generator=generator).numpy()]]
        kinds = torch.multinomial(self.kind_probabilities, batch - clean, replacement=True, generator=generator).numpy() if batch > clean else np.empty(0, dtype=np.int64)
        for k, pool in enumerate(self.error_kinds):
            count = int((kinds == k).sum())
            chosen.append(pool[torch.randint(len(pool), (count,), generator=generator).numpy()])
        indices = np.concatenate(chosen)
        return {name: torch.from_numpy(np.array(values[indices], dtype=np.float32 if name == "weight" else np.int64)) for name, values in self.arrays[split].items() if name != "identity"}


def records(root, split):
    with gzip.open(root / (split + ".jsonl.gz"), "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError("wrong split in correction evidence")
            yield row


def verify_lineage(base, teacher):
    if teacher.receipt.get("teacher_validated") is not True or any(teacher.receipt[k] != base.receipt[k] for k in ("corpus_id", "tokenizer_sha256", "vocab_size", "sequence_length")):
        raise ValueError("teacher data lineage mismatch")
    validation = list(records(base.root, "validation"))
    held_text = {r["clean_sha256"] for r in validation}
    held_work = {r["work_id"] for r in validation}
    rows = list(records(teacher.root, "train"))
    if any(r["clean_sha256"] in held_text or r["work_id"] in held_work for r in rows):
        raise ValueError("teacher training overlaps held-out correction sources")
    return rows


class TeacherPairData(PairData):
    def __init__(self, root, base):
        super().__init__(root)
        rows = verify_lineage(base, self)
        if len(rows) != len(self.arrays["train"]["identity"]):
            raise ValueError("teacher metadata/array row mismatch")
        kinds = sorted({r["kind"] for r in rows if r["source_text"] != r["target"]})
        self.error_kinds = [np.asarray([i for i, r in enumerate(rows) if r["kind"] == kind and r["source_text"] != r["target"]], dtype=np.int64) for kind in kinds]
        if not self.error_kinds or any(bool(r["source_text"] == r["target"]) != bool(self.arrays["train"]["identity"][i]) for i, r in enumerate(rows)):
            raise ValueError("teacher metadata/identity mismatch")

    def sample(self, split, batch, identity_fraction, generator):
        if split != "train":
            return super().sample(split, batch, identity_fraction, generator)
        clean = round(batch * identity_fraction)
        pool = self.pools[split][1]
        chosen = [pool[torch.randint(len(pool), (clean,), generator=generator).numpy()]]
        kinds = torch.randint(len(self.error_kinds), (batch - clean,), generator=generator).numpy()
        for k, pool in enumerate(self.error_kinds):
            count = int((kinds == k).sum())
            chosen.append(pool[torch.randint(len(pool), (count,), generator=generator).numpy()])
        indices = np.concatenate(chosen)
        return {name: torch.from_numpy(np.array(values[indices], dtype=np.float32 if name == "weight" else np.int64)) for name, values in self.arrays[split].items() if name != "identity"}
