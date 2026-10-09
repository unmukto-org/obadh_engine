"""Verify corpus partitions and bind derived artifacts to their training inputs.

Legacy corpora remain readable, but cannot acquire verified evaluation status.
All digests cover bytes, including compressed files; manifests use canonical JSON
for their dataset identity. No machine-specific absolute paths enter that identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

PARTITIONS_KIND = "obadh-corpus-partitions"
PARTITION_KIND = "obadh-corpus-partition"
VERSION = 1
SPLITS = ("train", "validation", "test")


def digest_json(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_files(root: Path, files: list[dict]) -> None:
    if not isinstance(files, list) or not files:
        raise ValueError(f"partition has no sentence files: {root}")
    declared: set[str] = set()
    for item in files:
        name = item["path"]
        relative = PurePosixPath(name)
        if (
            len(relative.parts) != 2
            or relative.parts[0] != "sentences"
            or not name.endswith(".tsv.gz")
            or "\\" in name
            or name in declared
        ):
            raise ValueError(f"invalid or duplicate sentence path: {name}")
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"sentence path escapes partition: {name}")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"sentence file digest mismatch: {path}")
        declared.add(name)
    actual = {
        path.relative_to(root).as_posix() for path in (root / "sentences").iterdir()
    }
    if actual != declared:
        raise ValueError(f"sentence file inventory mismatch: {root}")


def load_partition(root: Path, *, allowed: tuple[str, ...] = SPLITS) -> dict | None:
    if (root / ".building").exists() or (root.parent / ".building").exists():
        raise ValueError(f"corpus publication is incomplete: {root}")
    path = root / "manifest.json"
    if not path.exists():
        # A partition missing its manifest must never silently become legacy data.
        parent = root.parent / "manifest.json"
        if parent.exists() and read_object(parent).get("artifact") == PARTITIONS_KIND:
            raise ValueError(f"missing partition manifest: {path}")
        return None
    manifest = read_object(path)
    if manifest.get("artifact") == PARTITIONS_KIND:
        raise ValueError("use an explicit train, validation, or test directory")
    if manifest.get("artifact") != PARTITION_KIND:
        parent_path = root.parent / "manifest.json"
        if (
            parent_path.exists()
            and read_object(parent_path).get("artifact") == PARTITIONS_KIND
        ):
            raise ValueError(f"invalid partition manifest kind: {path}")
        return None
    if manifest.get("version") != VERSION or manifest.get("split") not in allowed:
        raise ValueError(f"unsupported version or forbidden corpus split: {path}")
    parent = read_object(root.parent / "manifest.json")
    if parent.get("artifact") != PARTITIONS_KIND or parent.get("version") != VERSION:
        raise ValueError(f"invalid partition-set manifest: {root.parent}")
    dataset_id = parent.get("dataset_id")
    payload = {key: value for key, value in parent.items() if key != "dataset_id"}
    if dataset_id != digest_json(payload) or manifest.get("dataset_id") != dataset_id:
        raise ValueError("corpus dataset identity mismatch")
    receipt = parent["splits"][manifest["split"]]
    if (
        manifest.get("files") != receipt["files"]
        or manifest.get("statistics") != receipt["statistics"]
    ):
        raise ValueError("partition manifest differs from partition-set receipt")
    verify_files(root, manifest["files"])
    return {
        "dataset_id": dataset_id,
        "split": manifest["split"],
        "manifest_sha256": sha256_file(path),
    }


def training_provenance(root: Path) -> dict | None:
    return load_partition(root, allowed=("train",))


def verify_artifact_training(path: Path, expected: dict | None = None) -> dict | None:
    sidecar = path.with_suffix(".manifest.json")
    manifest = read_object(sidecar) if sidecar.exists() else {}
    provenance = manifest.get("training_corpus")
    if expected is not None and provenance != expected:
        raise ValueError(f"artifact was not built from this training partition: {path}")
    if provenance is not None:
        if provenance.get("split") != "train":
            raise ValueError(f"artifact provenance is not a training split: {path}")
        if manifest.get("artifact_sha256") != sha256_file(path):
            raise ValueError(f"derived artifact digest mismatch: {path}")
    return provenance


def evaluation_provenance(model: Path, corpus: Path) -> dict:
    evaluation = load_partition(corpus, allowed=("validation", "test"))
    training = verify_artifact_training(model)
    if evaluation is None:
        return {"status": "unverified", "reason": "legacy evaluation corpus"}
    if training is None or training.get("dataset_id") != evaluation["dataset_id"]:
        raise ValueError(
            "evaluation requires an artifact trained on the same partition set"
        )
    # Verify the recorded training receipt and bytes as well as the eval corpus.
    actual_training = training_provenance(corpus.parent / "train")
    if training != actual_training:
        raise ValueError(
            "model training receipt differs from the verified train partition"
        )
    return {
        "status": "partition_verified",
        "training_corpus": training,
        "evaluation_corpus": evaluation,
        "model_sha256": sha256_file(model),
        "scope": "document groups and exact NFC sentence overlap; not near-duplicate detection",
    }


def neural_training_provenance(
    model: Path, training: Path, validation: Path | None
) -> dict:
    train = training_provenance(training)
    if train is None:
        if validation is not None and load_partition(validation) is not None:
            raise ValueError(
                "cannot mix legacy training data with a verified validation split"
            )
        return {"status": "unverified", "reason": "legacy training corpus"}
    if validation is None:
        raise ValueError("partitioned neural training requires --validation-corpus-dir")
    valid = load_partition(validation, allowed=("validation",))
    if valid is None or train["dataset_id"] != valid["dataset_id"]:
        raise ValueError(
            "training and validation must belong to the same partition set"
        )
    verify_artifact_training(model, expected=train)
    return {
        "status": "partition_verified",
        "training_corpus": train,
        "validation_corpus": valid,
    }


def verify_checkpoint_provenance(
    checkpoint: dict, provenance: dict, model: Path
) -> None:
    if provenance["status"] != "partition_verified":
        return
    report = checkpoint.get("report", {})
    if report.get("data_provenance") != provenance:
        raise ValueError(
            "checkpoint has missing or incompatible train/validation provenance"
        )
    if report.get("artifact", {}).get("sha256") != sha256_file(model):
        raise ValueError(
            "checkpoint retrieval artifact/token vocabulary identity mismatch"
        )
