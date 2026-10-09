"""Resumable pretrained byte-level accuracy reference, not a mobile artifact."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from tools.autosuggest.checkpoints import atomic_torch_save, capture_rng, restore_rng
from tools.corpus.provenance import digest_json, sha256_file, write_json
from tools.correction.byte_data import ByteData, PREFIX, encode, decode, target_weights


def collate(rows, device):
    inputs = [encode(PREFIX[r["mode"]] + r["source"]) for r in rows]
    targets = [encode(r["target"]) for r in rows]
    source_length, target_length = max(map(len, inputs)), max(map(len, targets))
    source = torch.tensor([r + [0] * (source_length - len(r)) for r in inputs], device=device)
    labels = torch.tensor([r + [-100] * (target_length - len(r)) for r in targets], device=device)
    return dict(input_ids=source, attention_mask=source.ne(0), labels=labels)


def load_model(pretrained):
    from transformers import T5ForConditionalGeneration
    # FP32 master weights and Adam states; CUDA autocast uses native BF16.
    model = T5ForConditionalGeneration.from_pretrained(pretrained, local_files_only=True, dtype=torch.float32)
    # Transformers 5.18 repurposes the old tie flag and introduces a separate
    # decoder scaling flag. Validate actual parameter sharing and arithmetic,
    # rather than interpreting the renamed configuration as model behavior.
    original = json.loads((pretrained / "config.json").read_text())
    scale_outputs = getattr(model.config, "scale_decoder_outputs", model.config.tie_word_embeddings)
    shared = model.shared.weight.data_ptr()
    if original.get("tie_word_embeddings") is not False or scale_outputs or shared == model.lm_head.weight.data_ptr():
        raise ValueError("ByT5 reference requires its original untied, unscaled output projection")
    if any(stack.embed_tokens.weight.data_ptr() != shared for stack in (model.encoder, model.decoder)):
        raise ValueError("ByT5 input embeddings must remain shared")
    return model


@torch.inference_mode()
def evaluate(model, bank_path, output, limit, checkpoint_hash, step, batch_size=16):
    from tools.autosuggest.compare_proofreaders import normalize, protected
    from tools.correction.evaluate import measure
    bank = [json.loads(line) for line in bank_path.read_text().splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    (output / ".building").touch()
    contract = dict(checkpoint_sha256=checkpoint_hash, step=step, bank_sha256=digest_json(bank), byte_limit=limit,
                    batch=batch_size,
                    script_sha256=sha256_file(Path(__file__)), decoder_sha256=sha256_file(Path(__file__).with_name("byte_data.py")),
                    precision="bf16 autocast, FP32 master weights", decoding="greedy UTF-8, bounded EOS, protected-span fallback", calibrated=False)
    write_json(output / "contract.json", contract)
    was_training = model.training
    model.eval()
    rows = []
    started = time.monotonic()
    with (output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for start in range(0, len(bank), batch_size):
            batch = [dict(r, source=normalize(r["source"]), target=normalize(r["target"])) for r in bank[start:start + batch_size]]
            encoded = [encode(PREFIX[r["mode"]] + r["source"]) for r in batch]
            active = [i for i, tokens in enumerate(encoded) if len(tokens) <= limit]
            outputs = {}
            if active:
                length = max(len(encoded[i]) for i in active)
                source = torch.tensor([encoded[i] + [0] * (length - len(encoded[i])) for i in active], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Permit the full declared capacity, independent of references.
                    tokens = model.generate(input_ids=source, attention_mask=source.ne(0), max_new_tokens=limit,
                                            do_sample=False, num_beams=1, use_cache=True,
                                            pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
                                            suppress_tokens=[0, 2] + list(range(259, model.config.vocab_size)))
                outputs = {i: decode(ids) for i, ids in zip(active, tokens.tolist())}
            for i, row in enumerate(batch):
                raw, reason = outputs.get(i, (None, "capacity_exceeded"))
                if raw is not None and protected(raw) != protected(row["source"]):
                    reason = "protected_span_changed"
                result = dict(row, raw_answer=raw, answer=row["source"] if reason else raw,
                              fallback_reason=reason, capacity_exceeded=i not in active, minimum_token_probability=None)
                rows.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(dict(event="byte_evaluate", step=step, rows=len(rows), total=len(bank))), flush=True)
    report = dict(completed=True, contract=contract, groups=measure(rows), seconds=time.monotonic() - started,
                  rows_sha256=sha256_file(output / "rows.jsonl"), scope="Development diagnostics, not representative human typing accuracy")
    write_json(output / "report.json", report)
    (output / ".building").unlink()
    model.train(was_training)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "pretrained", "output", "bank"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--initialize-from", type=Path, help="Start a new schedule from a compatible trained reference")
    p.add_argument("--family-weights", default="base=.7,lexical=.2,teacher=.1")
    p.add_argument("--initial-identity", type=float, default=.4)
    p.add_argument("--final-identity", type=float, default=.65)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--edit-weight", type=float, default=1.)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--max-steps-this-run", type=int)
    p.add_argument("--stop-epoch", type=float, required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--evaluate-checkpoint", type=Path)
    p.add_argument("--evaluate-output", type=Path)
    args = p.parse_args()
    try:
        parts = [item.split("=") for item in args.family_weights.split(",")]
        family_weights = {k: float(v) for k, v in parts}
        if len(parts) != len(family_weights) or not family_weights or any(not math.isfinite(v) or v <= 0 for v in family_weights.values()):
            raise ValueError("invalid family weights")
    except (ValueError, TypeError):
        p.error("family weights must be unique NAME=positive-weight entries")
    if not 0 < args.initial_identity < 1 or not 0 < args.final_identity < 1:
        p.error("identity fractions must lie strictly between zero and one")
    if min(args.steps, args.batch, args.accumulate, args.warmup, args.eval_every, args.eval_batches) < 1 or args.learning_rate <= 0 or not 1 <= args.edit_weight <= 32:
        p.error("invalid schedule")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(8)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if args.evaluate_checkpoint:
        if not args.evaluate_output:
            p.error("evaluation output required")
        state = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        model = load_model(args.pretrained)
        model.load_state_dict(state["state_dict"], strict=True)
        model.to(device)
        evaluate(model, args.bank, args.evaluate_output, state["contract"]["byte_limit"], sha256_file(args.evaluate_checkpoint), state["step"], args.batch)
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("existing training requires explicit resume")
    data = ByteData(args.data)
    model = load_model(args.pretrained).to(device)
    model.config.use_cache = False
    pretrained_files = {p.name: sha256_file(p) for p in sorted(args.pretrained.glob("*")) if p.is_file()}
    if args.initialize_from:
        initial = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        if initial.get("kind") != "obadh-byte-accuracy-reference" or initial["contract"]["pretrained_files"] != pretrained_files or initial["contract"]["byte_limit"] != data.receipt["byte_limit"]:
            raise ValueError("incompatible byte reference initialization")
        model.load_state_dict(initial["state_dict"], strict=True)
        del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(.9, .999), weight_decay=.01, fused=True)
    rng = random.Random(args.seed + 1)
    contract = dict(kind="obadh-byte-accuracy-reference", data_id=data.receipt["data_id"], byte_limit=data.receipt["byte_limit"],
                    pretrained_files=pretrained_files, initialization_sha256=sha256_file(args.initialize_from) if args.initialize_from else None,
                    source_sha256={n: sha256_file(Path(__file__).with_name(n)) for n in ("byte_train.py", "byte_data.py")},
                    checkpoints_sha256=sha256_file(Path(__file__).parents[1] / "autosuggest/checkpoints.py"),
                    config=model.config.to_dict(), steps=args.steps, batch=args.batch, accumulate=args.accumulate,
                    warmup=args.warmup, learning_rate=args.learning_rate, seed=args.seed, eval_every=args.eval_every, eval_batches=args.eval_batches,
                    edit_weight=args.edit_weight,
                    family_sampling=family_weights, error_sampling="uniform kinds within family",
                    identity_curriculum=dict(initial=args.initial_identity, final=args.final_identity, transition_fraction=.8),
                    bank_sha256=sha256_file(args.bank), torch=torch.__version__)
    step, best, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("resume contract mismatch")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["sampling_rng"])
        restore_rng(state["rng"], device)
        step, best, history = state["step"], state["best"], state["history"]
        del state
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, status):
        atomic_torch_save(dict(kind="obadh-byte-accuracy-reference", contract=contract, state_dict=model.state_dict(),
                              optimizer=optimizer.state_dict(), sampling_rng=rng.getstate(), rng=capture_rng(device),
                              step=step, best=best, history=history, status=status), args.output / name)

    def loss_for(rows):
        batch = collate(rows, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch, use_cache=False)
        if args.edit_weight == 1:
            return output.loss
        weights = [target_weights(r["source"], r["target"], args.edit_weight) for r in rows]
        length = batch["labels"].shape[1]
        weight = torch.tensor([w + [0.] * (length - len(w)) for w in weights], device=device)
        loss = F.cross_entropy(output.logits.float().transpose(1, 2), batch["labels"], reduction="none", ignore_index=-100)
        return (loss * weight).sum() / weight.sum()

    @torch.no_grad()
    def validation():
        model.eval()
        generator = random.Random(20261005)
        values = [float(loss_for(data.sample(generator, args.batch, .5, "validation"))) for _ in range(args.eval_batches)]
        model.train()
        return sum(values) / len(values)

    started = last_save = time.monotonic()
    start_step = step
    status = "completed"
    model.train()
    save("latest.pt", "running")
    while step < args.steps:
        if time.time() >= args.stop_epoch:
            status = "time_limit"
            break
        if args.max_steps_this_run is not None and step - start_step >= args.max_steps_this_run:
            status = "step_limit"
            break
        progress = max(0., (step - args.warmup) / max(1, args.steps - args.warmup))
        scale = (step + 1) / args.warmup if step < args.warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        optimizer.zero_grad(set_to_none=True)
        losses = []
        fraction = args.initial_identity if step < .8 * args.steps else args.final_identity
        for _ in range(args.accumulate):
            loss = loss_for(data.sample(rng, args.batch, fraction, family_weights=family_weights))
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite loss")
            (loss / args.accumulate).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % 25 == 0 or step <= 2:
            print(json.dumps(dict(event="byte_train", step=step, loss=sum(losses) / len(losses), gradient_norm=float(norm),
                                  seconds=time.monotonic() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated())), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val = validation()
            history.append(dict(step=step, validation_cross_entropy=val))
            if val < best:
                best = val
                save("best.pt", "running")
            save("latest.pt", "running")
            last_save = time.monotonic()
            print(json.dumps(dict(event="byte_validation", **history[-1])), flush=True)
        elif time.monotonic() - last_save >= 300:
            save("latest.pt", "running")
            last_save = time.monotonic()
    save("latest.pt", status)
    reloaded = torch.load(args.output / "latest.pt", map_location="cpu", weights_only=False)
    if reloaded["step"] != step or reloaded["contract"] != contract:
        raise ValueError("checkpoint reload failure")
    del reloaded
    report = dict(status=status, step=step, history=history, best=best, contract=contract,
                  parameters=sum(p.numel() for p in model.parameters()), seconds_this_run=time.monotonic() - started,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(), checkpoint_reload_verified=True)
    write_json(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "contract"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
