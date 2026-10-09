"""Generate resumable, explicitly synthetic Bangla dialogue candidates offline.

Structural filters are not human linguistic approval. Rows remain candidates
until reviewed and evaluated in a controlled training ablation. No benchmark
text, user typing, private messages, or external serving endpoint is used.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import time
import unicodedata

from tools.corpus.provenance import digest_json, sha256_file, write_json

TOPICS = [
    "বাস ধরার সময়",
    "ট্রেনের টিকিট",
    "রান্নার বাজার",
    "বন্ধুর জন্মদিন",
    "বৃষ্টিতে যাতায়াত",
    "বিদ্যুৎ চলে যাওয়া",
    "ইন্টারনেটের সমস্যা",
    "অনলাইন ক্লাস",
    "স্কুলের বাড়ির কাজ",
    "পরীক্ষার প্রস্তুতি",
    "লাইব্রেরির বই ফেরত",
    "ক্রিকেট খেলা",
    "ফুটবল ম্যাচ দেখা",
    "সিনেমা দেখতে যাওয়া",
    "গান শোনা",
    "বই কেনা",
    "দোকানের খোলার সময়",
    "পোশাকের মাপ",
    "জুতার রং",
    "পণ্য ফেরত দেওয়া",
    "বাসার চাবি",
    "হারানো ছাতা",
    "বাড়িতে অতিথি",
    "পিকনিকের প্রস্তুতি",
    "ছুটির পরিকল্পনা",
    "বাড়ি ফেরার সময়",
    "শিশুকে স্কুল থেকে আনা",
    "কাজের মিটিং",
    "অফিসে দেরি",
    "কাজ ভাগ করে নেওয়া",
    "প্রকল্পের সময়সীমা",
    "কম্পিউটারের সমস্যা",
    "ফোনের চার্জ",
    "ছবি পাঠানো",
    "ঠিকানা জানতে চাওয়া",
    "রাস্তা চিনিয়ে দেওয়া",
    "দোকানে অর্ডার",
    "খাবার ডেলিভারি",
    "চা খাওয়ার বিরতি",
    "দুপুরের খাবার",
    "রাতের খাবার",
    "ব্যায়ামের সময়",
    "ডাক্তারের অ্যাপয়েন্টমেন্টের সময়",
    "ওষুধের দোকান খোলা আছে কি না",
    "ভ্রমণের ব্যাগ গোছানো",
    "হোটেলের বুকিং",
    "আত্মীয়ের সঙ্গে দেখা",
    "পাড়ার অনুষ্ঠান",
    "ঈদের শুভেচ্ছা",
    "পূজার ছুটির পরিকল্পনা",
    "নতুন চাকরির খবর",
    "বাসা বদলানো",
    "বাড়িভাড়া নিয়ে কথা",
    "মেরামতের লোকের সময়",
    "টিউশনের সময়",
    "শিক্ষকের সঙ্গে সাক্ষাৎ",
    "একসঙ্গে পড়াশোনা",
    "বন্ধুর খোঁজ নেওয়া",
    "মন খারাপের কথা শোনা",
    "দুঃখ প্রকাশ করা",
    "ধন্যবাদ জানানো",
    "সাহায্য চাইতে দ্বিধা",
    "পরিকল্পনা বদলানো",
    "ভুল বোঝাবুঝি পরিষ্কার করা",
    "সময় মনে করিয়ে দেওয়া",
    "বাগানের গাছ",
    "পোষা বিড়ালের খাবার",
    "বিয়ের নিমন্ত্রণ",
    "শীতের পোশাক",
    "গরমে পানীয়",
    "হাঁটতে যাওয়া",
    "রেসিপি জানতে চাওয়া",
    "রান্নাঘরের জিনিস",
    "পরীক্ষার ফল",
    "কলেজে ভর্তি সম্পর্কে সময় জানা",
    "প্রিন্ট করানোর দোকান",
    "দরকারি কাগজ পাঠানো",
    "ভিডিও কলে যোগ দেওয়া",
    "খেলার মাঠে দেখা",
    "দৈনন্দিন ছোটখাটো ভালো খবর",
]
RELATIONS = [
    "দুই বন্ধু",
    "ভাই ও বোন",
    "মা ও প্রাপ্তবয়স্ক সন্তান",
    "দুই সহকর্মী",
    "প্রতিবেশী",
    "ক্রেতা ও দোকানদার",
    "সহপাঠী",
    "দুই আত্মীয়",
]
INTENTS = [
    "সময় বা তথ্য জানতে চাওয়া",
    "প্রস্তাবে সম্মতি জানানো",
    "ভদ্রভাবে না বলা ও বিকল্প দেওয়া",
    "একটি ভুল ধারণা পরিষ্কার করা",
    "পরিকল্পনা বদলানো",
    "ছোট একটি অনুরোধ করা",
    "একটি খবর জানানো",
    "কথার বিস্তারিত জানতে চাওয়া",
]
SYSTEM = 'Write original, natural Bangla mobile-message conversations. Use modern conversational language, never literary narration. Do not imitate an author. These are fictional everyday exchanges, not professional advice. Output exactly one JSON object with the key "turns", whose value is an array of message strings. No markdown, explanations, speaker labels, or extra keys. Preserve requested English words as English rather than transliterating them.'


def request(index, seed):
    rng = random.Random((seed << 32) + index)
    topic = TOPICS[index % len(TOPICS)]
    relationship = rng.choice(RELATIONS)
    intent = rng.choice(INTENTS)
    region = rng.choice(["বাংলাদেশের সাধারণ চলিত বাংলা", "পশ্চিমবঙ্গের সাধারণ চলিত বাংলা"])
    tone = rng.choice(["বন্ধুত্বপূর্ণ ও স্বাভাবিক", "ভদ্র ও সংক্ষিপ্ত", "দ্রুত লেখা ছোট বার্তা"])
    english = (
        rng.choice(["okay", "thanks", "sorry", "please"])
        if rng.random() < 0.2
        else None
    )
    prompt = f"বিষয়: {topic}। সম্পর্ক: {relationship}। কথার উদ্দেশ্য: {intent}। ভাষাভঙ্গি: {region}; {tone}। ৪ থেকে ৬টি পালা লিখুন। প্রতিটি পালা ১ থেকে ২৫ শব্দের হবে। অন্তত একটি স্বাভাবিক ছোট উত্তর সর্বোচ্চ ৩ শব্দের হবে। প্রশ্ন ও যতিচিহ্ন স্বাভাবিকভাবে ব্যবহার করুন। আজকের কথোপকথনের নমুনা {index + 1}; নতুন পরিস্থিতি ও নতুন বাক্য লিখুন।"
    if english:
        prompt += (
            f" অন্তত একবার এই English শব্দটি অক্ষর অপরিবর্তিত রেখে ব্যবহার করুন: {english}।"
        )
    return {
        "topic": topic,
        "relationship": relationship,
        "intent": intent,
        "region": region,
        "tone": tone,
        "required_english": english,
        "prompt": prompt,
    }


def validate(raw, required_english=None):
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None, "invalid_json"
    if (
        not isinstance(obj, dict)
        or set(obj) != {"turns"}
        or not isinstance(obj["turns"], list)
        or not 4 <= len(obj["turns"]) <= 6
    ):
        return None, "invalid_structure"
    turns = obj["turns"]
    if any(
        not isinstance(t, str) or not t.strip() or len(t) > 400 or "\n" in t
        for t in turns
    ):
        return None, "invalid_turn"
    turns = [unicodedata.normalize("NFC", t.strip()) for t in turns]
    if (
        any(len(t.split()) > 30 for t in turns)
        or min(len(t.split()) for t in turns) > 3
    ):
        return None, "verbosity"
    joined = "\n".join(turns)
    letters = [c for c in joined if unicodedata.category(c).startswith("L")]
    if (
        not letters
        or sum("\u0980" <= c <= "\u09ff" for c in letters) / len(letters) < 0.8
    ):
        return None, "language_ratio"
    if required_english and not re.search(
        r"\b" + re.escape(required_english) + r"\b", joined, re.I
    ):
        return None, "english_not_preserved"
    if any(re.match(r"^(?:[AB১২12]|ব্যক্তি\s*[১২12])\s*:", t) for t in turns):
        return None, "speaker_labels"
    if any(unicodedata.category(c) == "Cc" and c != "\n" for c in joined):
        return None, "control_characters"
    return turns, None


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--stop-epoch", type=float, required=True)
    args = p.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    contract = {
        "model": selection["model"],
        "revision": selection["revision"],
        "script_sha256": sha256_file(Path(__file__)),
        "batch": args.batch,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 20,
        "thinking": False,
        "scope": __doc__,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.output / "candidates.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(
        "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS candidates(id INTEGER PRIMARY KEY, scenario TEXT, raw TEXT, turns TEXT, reason TEXT, digest TEXT);"
    )
    db.execute("CREATE INDEX IF NOT EXISTS candidate_digests ON candidates(digest)")
    previous = db.execute('SELECT value FROM metadata WHERE key="contract"').fetchone()
    if previous and json.loads(previous[0]) != contract:
        raise ValueError("generation resume contract mismatch")
    if not previous:
        db.execute(
            "INSERT INTO metadata VALUES(?,?)",
            ("contract", json.dumps(contract, sort_keys=True)),
        )
        db.commit()
    done = {row[0] for row in db.execute("SELECT id FROM candidates")}
    torch.set_num_threads(8)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False, padding_side="left"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).eval()
    status = "completed"
    for start in range(0, args.count, args.batch):
        indices = list(range(start, min(args.count, start + args.batch)))
        if all(i in done for i in indices):
            continue
        if any(i in done for i in indices):
            raise ValueError(
                "partial batch found; count must extend along batch boundaries"
            )
        if time.time() >= args.stop_epoch:
            status = "time_limit"
            break
        scenarios = [request(i, args.seed) for i in indices]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": s["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for s in scenarios
        ]
        inputs = tokenizer(
            prompts, padding=True, return_tensors="pt", add_special_tokens=False
        ).to("cuda")
        torch.manual_seed(args.seed + start)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=20,
                use_cache=True,
            )
        answers = tokenizer.batch_decode(
            output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        with db:
            for index, scenario, raw in zip(indices, scenarios, answers):
                turns, reason = validate(raw, scenario["required_english"])
                digest = digest_json(turns) if turns else None
                if (
                    digest
                    and db.execute(
                        "SELECT 1 FROM candidates WHERE digest=?", (digest,)
                    ).fetchone()
                ):
                    reason = "duplicate_conversation"
                    turns = None
                db.execute(
                    "INSERT INTO candidates VALUES(?,?,?,?,?,?)",
                    (
                        index,
                        json.dumps(scenario, ensure_ascii=False),
                        raw,
                        json.dumps(turns, ensure_ascii=False) if turns else None,
                        reason,
                        digest,
                    ),
                )
        counts = {
            str(reason or "structurally_accepted"): n
            for reason, n in db.execute(
                "SELECT reason,COUNT(*) FROM candidates GROUP BY reason"
            )
        }
        report = {
            "contract": contract,
            "attempted": sum(counts.values()),
            "counts": counts,
            "seconds_this_run": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "status": "completed" if start + len(indices) >= args.count else "running",
            "admitted_for_training": False,
        }
        write_json(args.output / "report.json", report)
        print(json.dumps(report), flush=True)
    counts = {
        str(reason or "structurally_accepted"): n
        for reason, n in db.execute(
            "SELECT reason,COUNT(*) FROM candidates GROUP BY reason"
        )
    }
    write_json(
        args.output / "report.json",
        {
            "contract": contract,
            "attempted": sum(counts.values()),
            "counts": counts,
            "seconds_this_run": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "status": status,
            "admitted_for_training": False,
        },
    )
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()


if __name__ == "__main__":
    main()
