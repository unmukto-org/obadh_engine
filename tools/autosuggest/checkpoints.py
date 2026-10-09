"""Atomic, resumable training checkpoints shared by offline model tools."""

from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile

import torch


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Keep the previous checkpoint intact until the replacement is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_rng(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict, device: torch.device) -> None:
    if state["device_type"] != device.type:
        raise ValueError("exact training resume requires the same device type")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA device count differs from this training allocation"
            )
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
