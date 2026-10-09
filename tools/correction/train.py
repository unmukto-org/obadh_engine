"""Recoverable task-specific correction training with an explicit copy curriculum."""

from dataclasses import asdict
import argparse
import json
import math
from pathlib import Path
import time
import torch

from tools.autosuggest.checkpoints import atomic_torch_save, capture_rng, restore_rng
from tools.corpus.provenance import sha256_file, write_json
from tools.correction.data import PairData
from tools.correction.model import CorrectionConfig, Corrector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "initialize-from", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--initialize-kind", choices=("language", "correction"), default="language")
    p.add_argument("--architecture", choices=("recurrent", "edit"), default="recurrent")
    p.add_argument("--teacher-data", type=Path)
    p.add_argument("--teacher-fraction", type=float, default=.5)
    p.add_argument("--base-error-sampling", choices=("rows", "sqrt-kinds", "uniform-kinds"), default="rows")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=.0003)
    p.add_argument("--encoder-rate", type=float, default=.00005)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--max-steps-this-run", type=int)
    p.add_argument("--stop-epoch", type=float, required=True)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()
    if args.architecture == "edit" and args.initialize_kind != "correction":
        p.error("edit training requires correction encoder initialization")
    if min(args.steps, args.batch, args.eval_every, args.eval_batches, args.warmup) < 1 or not 0 < args.encoder_rate <= args.learning_rate or not 0 < args.teacher_fraction < 1 or (args.max_steps_this_run is not None and args.max_steps_this_run < 1):
        p.error("invalid training schedule")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this profile")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("existing run requires explicit resume")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    from tools.correction.distillation_sampling import TeacherPairData, KindBalancedPairData
    data = PairData(args.data) if args.base_error_sampling == "rows" else KindBalancedPairData(args.data, exponent=.5 if args.base_error_sampling == "sqrt-kinds" else 0.)
    teacher = TeacherPairData(args.teacher_data, data) if args.teacher_data else None
    initial = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
    if initial["data_provenance"]["tokenizer_sha256"] != data.receipt["tokenizer_sha256"]:
        raise ValueError("initialization tokenizer mismatch")
    if args.architecture == "edit":
        from tools.correction.edit_model import EditConfig, EditCorrector
        config = EditConfig(**{k: v for k, v in initial["config"].items() if k != "decoder_layers"})
    else:
        config = CorrectionConfig(**initial["config"])
    if config.sequence_length != data.receipt["sequence_length"] or config.vocab_size != data.receipt["vocab_size"]:
        raise ValueError("initialization/data dimensions mismatch")
    contract = dict(
        config=asdict(config), data_id=data.receipt["data_id"], initialization_sha256=sha256_file(args.initialize_from),
        initialization_kind=args.initialize_kind,
        architecture=args.architecture,
        base_sampling=getattr(data, "sampling_contract", dict(strategy="uniform-error-rows")),
        teacher_data_id=teacher.receipt["data_id"] if teacher else None,
        teacher_fraction=args.teacher_fraction if teacher else None,
        teacher_sampling="uniform error kinds, explicit identity fraction, same-corpus work/text disjointness" if teacher else None,
        steps=args.steps, batch=args.batch, seed=args.seed, warmup=args.warmup,
        learning_rate=args.learning_rate, encoder_rate=args.encoder_rate,
        eval_every=args.eval_every, eval_batches=args.eval_batches,
        identity_curriculum={"initial": .35, "final": .65, "transition_fraction": .8},
        source_sha256={name: sha256_file(Path(__file__).with_name(name)) for name in ("train.py", "model.py", "data.py", "distillation_sampling.py") + (("edit_model.py", "edit_policy.py") if args.architecture == "edit" else ())},
        encoder_helpers_sha256={name: sha256_file(Path(__file__).parents[1] / "autosuggest" / name) for name in ("subword_lm.py", "subword_cache.py", "checkpoints.py")},
    )
    model = EditCorrector(config) if args.architecture == "edit" else Corrector(config)
    if args.architecture == "edit":
        model.initialize(initial)
    elif args.initialize_kind == "language":
        model.initialize_encoder(initial)
    else:
        if initial.get("kind") != "obadh-contextual-corrector" or initial["config"] != asdict(config):
            raise ValueError("invalid correction initialization")
        model.load_state_dict(initial["state_dict"], strict=True)
    del initial
    device = torch.device("cuda")
    model.to(device)
    parameters = list(model.named_parameters())
    optimizer = torch.optim.AdamW([
        dict(params=[p for n, p in parameters if n.startswith("encoder.")], lr=args.encoder_rate, base_lr=args.encoder_rate),
        dict(params=[p for n, p in parameters if not n.startswith("encoder.")], lr=args.learning_rate, base_lr=args.learning_rate),
    ], betas=(.9, .95), weight_decay=.01, fused=True)
    sampler = torch.Generator().manual_seed(args.seed + 1)
    step, best, history = 0, float("inf"), []
    alignment_counts = {s: dict(supported=0, unsupported=0) for s in ("train", "validation")}
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("resume contract mismatch")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.set_state(state["sampling_rng"])
        restore_rng(state["rng"], device)
        step, best, history = state["step"], state["best"], state["history"]
        alignment_counts = state.get("alignment_counts", alignment_counts)
        del state
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, status):
        atomic_torch_save(dict(
            version=1, kind="obadh-contextual-edit-corrector" if args.architecture == "edit" else "obadh-contextual-corrector", contract=contract,
            data_provenance=data.receipt, config=asdict(config), state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(), rng=capture_rng(device), sampling_rng=sampler.get_state(),
            step=step, best=best, history=history, status=status,
            alignment_counts=alignment_counts,
        ), args.output / name)

    def loss_for(split, generator, fraction):
        if teacher and split == "train":
            teacher_count = round(args.batch * args.teacher_fraction)
            base = data.sample(split, args.batch - teacher_count, fraction, generator)
            distilled = teacher.sample(split, teacher_count, fraction, generator)
            batch = {k: torch.cat((base[k], distilled[k])) for k in base}
        else:
            batch = data.sample(split, args.batch, fraction, generator)
        if args.architecture == "edit":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, supported, unsupported = model.loss(batch)
            alignment_counts[split]["supported"] += supported
            alignment_counts[split]["unsupported"] += unsupported
            return loss
        batch = {k: v.to(device) for k, v in batch.items()}
        previous = torch.cat((torch.ones((args.batch, 1), dtype=torch.long, device=device), batch["target"][:, :-1]), dim=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, attention, gate = model(batch["source"], batch["mode"], previous)
            logp = model.target_log_prob(logits, attention, gate, batch["source"], batch["target"])
            weight = batch["weight"]
            loss = -(logp * weight).sum() / weight.sum()
        return loss

    @torch.no_grad()
    def evaluate():
        model.eval()
        generator = torch.Generator().manual_seed(20261005)
        values = [float(loss_for("validation", generator, .5)) for _ in range(args.eval_batches)]
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
        if args.max_steps_this_run and step - start_step >= args.max_steps_this_run:
            status = "step_limit"
            break
        progress = max(0., (step - args.warmup) / max(1, args.steps - args.warmup))
        scale = (step + 1) / args.warmup if step < args.warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
        fraction = .35 if step < .8 * args.steps else .65
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for("train", sampler, fraction)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite correction loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % 50 == 0:
            print(json.dumps(dict(event="correction_train", step=step, loss=float(loss.detach()), identity_fraction=fraction, seconds=time.monotonic() - started)), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            validation = evaluate()
            history.append(dict(step=step, weighted_validation_loss=validation))
            if validation < best:
                best = validation
                save("best.pt", "running")
            save("latest.pt", "running")
            last_save = time.monotonic()
            print(json.dumps(dict(event="correction_validation", **history[-1])), flush=True)
        elif time.monotonic() - last_save >= 300:
            save("latest.pt", "running")
            last_save = time.monotonic()
    save("latest.pt", status)
    state = torch.load(args.output / "latest.pt", map_location="cpu", weights_only=False)
    if state["step"] != step or state["contract"] != contract:
        raise ValueError("checkpoint reload failure")
    report = dict(status=status, step=step, parameters=sum(p.numel() for p in model.parameters()), history=history, best=best, contract=contract, alignment_counts=alignment_counts, peak_cuda_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(), torch=torch.__version__, seconds_this_run=time.monotonic() - started, scope="Synthetic correction training with optional separately identified teacher agreement; evaluation on native typing remains required")
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
