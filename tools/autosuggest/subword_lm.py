"""Portable compact causal subword transformer and verified corpus preparation.

Byte-level BPE retains all UTF-8 input without a word-vocabulary OOV reset.
Gated feed-forward blocks, rotary positions, tied embeddings and causal SDPA
keep the student small. Research checkpoints are not release model packages.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

from tools.corpus.provenance import load_partition, sha256_file, write_json, digest_json
from tools.autosuggest.eval_ngram_lm import iter_eval_sentence_tokens


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    width: int = 256
    layers: int = 4
    heads: int = 4
    ff_width: int = 768
    sequence_length: int = 128


def make_model(config):
    import torch
    from torch import nn
    from torch.nn import functional as F

    if config.width % config.heads or (config.width // config.heads) % 2:
        raise ValueError("rotary attention requires an even head dimension")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d = config.width
            self.attn_norm = nn.LayerNorm(d, bias=False)
            self.ff_norm = nn.LayerNorm(d, bias=False)
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.attn_out = nn.Linear(d, d, bias=False)
            self.gate_up = nn.Linear(d, 2 * config.ff_width, bias=False)
            self.down = nn.Linear(config.ff_width, d, bias=False)

        def forward(self, x, cos, sin):
            b, t, d = x.shape
            q, k, v = (
                self.qkv(self.attn_norm(x))
                .view(b, t, 3, config.heads, d // config.heads)
                .unbind(2)
            )

            def rotary(a):
                a = a.transpose(1, 2)
                left, right = a.chunk(2, dim=-1)
                return a * cos.to(a.dtype) + torch.cat((-right, left), dim=-1) * sin.to(
                    a.dtype
                )

            q, k = rotary(q), rotary(k)
            a = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True)
            x = x + self.attn_out(a.transpose(1, 2).reshape(b, t, d))
            gate, up = self.gate_up(self.ff_norm(x)).chunk(2, dim=-1)
            return x + self.down(F.silu(gate) * up)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.embedding = nn.Embedding(config.vocab_size, config.width)
            self.blocks = nn.ModuleList(Block() for _ in range(config.layers))
            self.norm = nn.LayerNorm(config.width, bias=False)
            frequency = 1.0 / (
                10000
                ** (
                    torch.arange(0, config.width // config.heads, 2).float()
                    / (config.width // config.heads)
                )
            )
            angle = torch.outer(torch.arange(config.sequence_length).float(), frequency)
            angle = torch.cat((angle, angle), dim=-1)
            self.register_buffer("cos", angle.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin", angle.sin()[None, None, :, :], persistent=False)
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(module.weight, std=0.02)
            for block in self.blocks:
                nn.init.normal_(
                    block.attn_out.weight, std=0.02 / math.sqrt(2 * config.layers)
                )
                nn.init.normal_(
                    block.down.weight, std=0.02 / math.sqrt(2 * config.layers)
                )

        def forward(self, ids):
            x = self.embedding(ids)
            t = ids.shape[1]
            for block in self.blocks:
                x = block(x, self.cos[:, :, :t], self.sin[:, :, :t])
            return self.norm(x) @ self.embedding.weight.T

    return Model()


def prepare(args):
    import numpy as np
    from tokenizers import pre_tokenizers, trainers

    train = load_partition(args.corpus / "train", allowed=("train",))
    val = load_partition(args.corpus / "validation", allowed=("validation",))
    if not train or not val or train["dataset_id"] != val["dataset_id"]:
        raise ValueError("verified matching partitions required")
    if args.output.exists():
        raise ValueError("refusing to replace prepared subword data")
    args.output.mkdir(parents=True)
    (args.output / ".building").touch()

    def texts(split, maximum):
        for _, tokens in iter_eval_sentence_tokens(
            args.corpus / split,
            sources=None,
            skip_sentences_per_source=0,
            max_sentences_per_source=maximum,
        ):
            yield " ".join(tokens)

    from tools.autosuggest.subword_tokenizer import make_tokenizer

    tokenizer = make_tokenizer(args.pretokenizer)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        min_frequency=2,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        texts("train", args.tokenizer_sentences), trainer=trainer
    )
    tokenizer.save(str(args.output / "tokenizer.json"))
    counts = {}
    for split in ("train", "validation"):
        token_count = 0
        sentence_count = 0
        buffer = []
        maximum = args.sentences if split == "train" else args.validation_sentences
        with (args.output / f"{split}.u16").open("wb") as out:

            def flush():
                nonlocal token_count, sentence_count
                for encoded in tokenizer.encode_batch(buffer, add_special_tokens=False):
                    ids = [1] + encoded.ids + [2]
                    if 3 in ids:
                        raise ValueError("byte vocabulary unexpectedly produced UNK")
                    out.write(np.asarray(ids, dtype="<u2").tobytes())
                    token_count += len(ids)
                    sentence_count += 1
                buffer.clear()

            for text in texts(split, maximum):
                buffer.append(text)
                if len(buffer) == 1024:
                    flush()
            if buffer:
                flush()
        counts[split] = {
            "tokens": token_count,
            "sentences": sentence_count,
            "sha256": sha256_file(args.output / f"{split}.u16"),
        }
    receipt = {
        "version": 1,
        "kind": "obadh-subword-training-data",
        "dataset_id": train["dataset_id"],
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": sha256_file(args.output / "tokenizer.json"),
        "special_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3},
        "encoding": "little-endian-uint16",
        "policy": {
            "max_train_sentences_per_source": args.sentences,
            "max_validation_sentences_per_source": args.validation_sentences,
            "tokenizer_train_sentences_per_source": args.tokenizer_sentences,
            "pretokenizer": args.pretokenizer,
            "normalization": "NFC",
            "packed_sentences": "BOS/text/EOS; causal cross-sentence context",
        },
        "partitions": counts,
        "source_manifests": {
            s: sha256_file(args.corpus / s / "manifest.json")
            for s in ("train", "validation")
        },
    }
    write_json(args.output / "manifest.json", receipt)
    (args.output / ".building").unlink()
    print(json.dumps(receipt, indent=2))


def train(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tools.autosuggest.checkpoints import (
        atomic_torch_save,
        capture_rng,
        restore_rng,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this training profile")
    if (args.data / ".building").exists():
        raise ValueError("incomplete training data")
    receipt = json.loads((args.data / "manifest.json").read_text())
    for split, record in receipt["partitions"].items():
        if split not in ("train", "validation") or record["sha256"] != sha256_file(
            args.data / f"{split}.u16"
        ):
            raise ValueError("data integrity failure")
    if receipt["tokenizer_sha256"] != sha256_file(args.data / "tokenizer.json"):
        raise ValueError("tokenizer integrity failure")
    documents = None
    if receipt.get("kind") == "obadh-document-training-data":
        from tools.autosuggest.document_data import DocumentData

        if args.literary_data or args.literary_fraction:
            raise ValueError("document data already owns domain/author sampling")
        documents = DocumentData(
            args.data,
            tokenizer_sha256=receipt["tokenizer_sha256"],
            sequence_length=args.sequence_length,
        )
    literary = None
    if args.literary_data:
        from tools.autosuggest.literary_data import LiteraryData

        literary = LiteraryData(
            args.literary_data,
            tokenizer_sha256=receipt["tokenizer_sha256"],
            sequence_length=args.sequence_length,
        )
    if not 0 <= args.literary_fraction <= 1 or (
        args.literary_fraction > 0 and literary is None
    ):
        raise ValueError("literary fraction requires compatible data")
    config = ModelConfig(
        vocab_size=receipt["vocab_size"],
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        ff_width=args.ff_width,
        sequence_length=args.sequence_length,
    )
    contract = {
        "config": asdict(config),
        "data_id": digest_json(receipt),
        "script_sha256": sha256_file(Path(__file__)),
        "batch": args.batch,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "warmup": args.warmup,
        "eval_batches": args.eval_batches,
        "eval_every": args.eval_every,
        "initialization_sha256": sha256_file(args.initialize_from)
        if args.initialize_from
        else None,
        "literary_data_id": literary.receipt["data_id"] if literary else None,
        "literary_fraction": args.literary_fraction,
        "document_loader_sha256": sha256_file(
            Path(__file__).with_name("document_data.py")
        )
        if documents
        else None,
        "literary_loader_sha256": sha256_file(
            Path(__file__).with_name("literary_data.py")
        )
        if literary
        else None,
    }
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    model = make_model(config).to(device)
    if args.initialize_from:
        initial = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        if initial["config"] != asdict(config):
            raise ValueError("initialization architecture mismatch")
        if (
            initial["data_provenance"]["tokenizer_sha256"]
            != receipt["tokenizer_sha256"]
        ):
            raise ValueError("initialization tokenizer mismatch")
        model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    data = {
        s: np.memmap(args.data / f"{s}.u16", dtype="<u2", mode="r")
        for s in ("train", "validation")
    }
    if any(len(x) < config.sequence_length + 2 for x in data.values()):
        raise ValueError("insufficient sequence data")
    step = 0
    best = float("inf")
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["contract"] != contract:
            raise ValueError("resume contract mismatch")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best = checkpoint["best"]
        history = checkpoint["history"]
        restore_rng(checkpoint["rng"], device)

    def batch(split, generator=None):
        if documents is not None:
            values = torch.from_numpy(
                documents.sample(split, args.batch, generator)
            ).to(device)
            return values[:, :-1], values[:, 1:]
        literary_count = round(args.batch * args.literary_fraction)
        base_count = args.batch - literary_count
        starts = torch.randint(
            0,
            len(data[split]) - config.sequence_length - 1,
            (base_count,),
            generator=generator,
        )
        values = (
            np.stack(
                [
                    np.asarray(
                        data[split][int(i) : int(i) + config.sequence_length + 1],
                        dtype=np.int64,
                    )
                    for i in starts
                ]
            )
            if base_count
            else np.empty((0, config.sequence_length + 1), dtype=np.int64)
        )
        if literary_count:
            values = np.concatenate(
                (values, literary.sample(split, literary_count, generator))
            )
        values = torch.from_numpy(values).to(device)
        return values[:, :-1], values[:, 1:]

    if not args.resume and any(
        (args.output / n).exists() for n in ("latest.pt", "best.pt", "report.json")
    ):
        raise ValueError("refusing to overwrite an existing run without --resume")
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, status):
        atomic_torch_save(
            {
                "version": 1,
                "contract": contract,
                "data_provenance": receipt,
                "literary_provenance": literary.receipt if literary else None,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng(device),
                "step": step,
                "best": best,
                "history": history,
                "status": status,
                "config": asdict(config),
            },
            args.output / name,
        )

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        generator = torch.Generator().manual_seed(20261005)
        for _ in range(args.eval_batches):
            x, y = batch("validation", generator)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.flatten(0, 1), y.flatten(), ignore_index=0
                )
            losses.append(float(loss))
        model.train()
        return sum(losses) / len(losses)

    started = time.monotonic()
    last_save = started
    start_step = step
    status = "completed"
    model.train()
    save("latest.pt", "running")
    while step < args.steps:
        if time.time() >= args.stop_epoch:
            status = "time_limit"
            break
        if args.max_steps_this_run and step - start_step >= args.max_steps_this_run:
            status = "step_limit"
            break
        progress = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        scale = (
            min(1.0, (step + 1) / max(1, args.warmup))
            if step < args.warmup
            else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        x, y = batch("train")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=0)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % 50 == 0:
            print(
                json.dumps(
                    {
                        "event": "subword_train",
                        "step": step,
                        "loss": float(loss.detach()),
                        "seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            validation = evaluate()
            history.append({"step": step, "validation_token_loss": validation})
            if validation < best:
                best = validation
                save("best.pt", "running")
            print(
                json.dumps(
                    {
                        "event": "subword_validation",
                        "step": step,
                        "token_loss": validation,
                    }
                ),
                flush=True,
            )
            save("latest.pt", "running")
            last_save = time.monotonic()
        elif time.monotonic() - last_save >= 300:
            save("latest.pt", "running")
            last_save = time.monotonic()
    save("latest.pt", status)
    # Explicitly reload to exercise checkpoint durability even in a two-step smoke.
    reloaded = torch.load(
        args.output / "latest.pt", map_location="cpu", weights_only=False
    )
    if reloaded["step"] != step:
        raise ValueError("checkpoint reload failed")
    report = {
        "status": status,
        "step": step,
        "tokens_seen": step * args.batch * config.sequence_length,
        "history": history,
        "config": asdict(config),
        "parameters": sum(x.numel() for x in model.parameters()),
        "contract": contract,
        "seconds_this_run": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "scope": "subword language modeling; word-level quality requires separate evaluation",
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument(
        "--pretokenizer",
        choices=("unicode-marks-v1", "gpt2-regex-v0"),
        default="unicode-marks-v1",
    )
    a.add_argument("--vocab", type=int, default=8192)
    a.add_argument("--sentences", type=int, default=1000000)
    a.add_argument("--validation-sentences", type=int, default=6000)
    a.add_argument("--tokenizer-sentences", type=int, default=100000)
    a = sub.add_parser("train")
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--resume", type=Path)
    a.add_argument(
        "--initialize-from",
        type=Path,
        help="Start a new schedule with compatible weights and a fresh optimizer; preserve this argument when resuming",
    )
    a.add_argument("--literary-data", type=Path)
    a.add_argument("--literary-fraction", type=float, default=0.0)
    for name, default in [
        ("width", 256),
        ("layers", 4),
        ("heads", 4),
        ("ff-width", 768),
        ("sequence-length", 128),
        ("batch", 128),
        ("steps", 6000),
        ("warmup", 300),
        ("seed", 17),
        ("eval-batches", 20),
        ("eval-every", 500),
        ("max-steps-this-run", 0),
    ]:
        a.add_argument("--" + name, type=int, default=default)
    a.add_argument("--learning-rate", type=float, default=0.0006)
    a.add_argument("--stop-epoch", type=float, required=True)
    args = p.parse_args()
    for name, value in vars(args).items():
        if (
            isinstance(value, int)
            and name not in ("seed", "max_steps_this_run")
            and value < 1
        ):
            p.error(f"{name} must be positive")
    if hasattr(args, "vocab") and not 260 <= args.vocab <= 65535:
        p.error("vocab must fit uint16 and contain the byte alphabet")
    if hasattr(args, "max_steps_this_run") and args.max_steps_this_run < 0:
        p.error("max-steps-this-run must be nonnegative")
    if hasattr(args, "learning_rate") and (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.stop_epoch)
    ):
        p.error("invalid learning rate or deadline")
    {"prepare": prepare, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
