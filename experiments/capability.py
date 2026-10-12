"""Measure benchmark accuracy in a plain prompt and inside an evasion episode.

    python experiments/capability.py --task mmlu --n-questions 1000 [--adapter r1_sigmoid_mse]
    python experiments/capability.py --task mmlu_pro --n-questions 1000 [--adapter r1_sigmoid_mse]
    python experiments/capability.py --task triviaqa --n-questions 1000 [--adapter r1_sigmoid_mse]
    python experiments/capability.py --task gsm8k [--adapter r1_sigmoid_mse]

Reads  data/splits/, outputs/<model>/adapters/<adapter>/, the benchmark from the HF Hub
Writes experiments/outputs/capability/<model>_<adapter>_<task>_single.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.activations import MODELS, load_model, model_dir  # noqa: E402
from core.concepts import ALL_CONCEPTS, held_out_concepts, same_family  # noqa: E402
from core.episodes import (  # noqa: E402
    ALL_CELLS,
    build_episode,
    content,
    eval_prefills,
    load_pools,
    with_imitate,
)
from core.probes import seed_for  # noqa: E402

HERE = Path(__file__).resolve().parent
LETTERS = "ABCD"

GSM8K_SHOTS = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
        "in May. How many clips did Natalia sell altogether in April and May?",
        "In April she sold 48 clips. In May she sold 48 / 2 = 24 clips. Altogether she sold "
        "48 + 24 = 72 clips. The answer is 72.",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
        "babysitting. How much did she earn?",
        "Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned 50 * 0.2 = $10. "
        "The answer is 10.",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the "
        "money she needs. Her parents decided to give her $15 for that purpose, and her "
        "grandparents twice as much as her parents. How much more money does Betty need?",
        "Betty has 100 / 2 = $50. Her grandparents gave 15 * 2 = $30. In total she has "
        "50 + 15 + 30 = $95. She still needs 100 - 95 = $5. The answer is 5.",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does "
        "he write a year?",
        "Each time he writes 3 * 2 = 6 pages. Twice a week that is 6 * 2 = 12 pages. In a year "
        "that is 12 * 52 = 624 pages. The answer is 624.",
    ),
]
GSM8K_LEAD = (
    "Solve the math problem. Show brief reasoning and end with "
    "'The answer is <number>'."
)


def mmlu_block(subject: str, shots: list, q: str, choices: list) -> str:
    """The whole 5-shot prompt as one text block, ending on the unanswered question."""

    def one(question, ch, ans):
        s = (
            question
            + "\n"
            + "".join(f"{LETTERS[i]}. {c}\n" for i, c in enumerate(ch))
            + "Answer:"
        )
        return s + (f" {LETTERS[ans]}\n\n" if ans is not None else "")

    head = (
        f"The following are multiple choice questions (with answers) about "
        f"{subject.replace('_', ' ')}.\n\n"
    )
    return (
        head
        + "".join(one(e["question"], e["choices"], e["answer"]) for e in shots)
        + one(q, choices, None)
    )


def gsm8k_single_block(question: str) -> str:
    """Four worked examples and the real question, packed into one block."""
    body = "".join(f"{GSM8K_LEAD}\n{q}\n{a}\n\n" for q, a in GSM8K_SHOTS)
    return body + f"{GSM8K_LEAD}\n{question}"


TRIVIA_LEAD = (
    "Answer the trivia question with the shortest correct answer. "
    "Give only the answer, no explanation."
)
TRIVIA_SHOTS = [
    ("Which planet is known as the Red Planet?", "Mars"),
    ("Who wrote the play Macbeth?", "William Shakespeare"),
    ("What is the capital city of Japan?", "Tokyo"),
    ("In which year did the Berlin Wall fall?", "1989"),
]


def trivia_block(question: str) -> str:
    """Closed-book TriviaQA prompt. The four shots set the answer format."""
    shots = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in TRIVIA_SHOTS)
    return f"{TRIVIA_LEAD}\n\n{shots}Q: {question}\nA:"


def mmlu_pro_block(shots: list, q: str, choices: list) -> str:
    """MMLU-Pro prompt: the MMLU layout with up to ten options."""
    letters = "ABCDEFGHIJ"

    def one(question, ch, ans):
        body = (
            question + "\n" + "".join(f"{letters[i]}. {c}\n" for i, c in enumerate(ch))
        )
        return body + "Answer:" + (f" {letters[ans]}\n\n" if ans is not None else "")

    head = (
        "The following are multiple choice questions. Answer with a single letter.\n\n"
    )
    return (
        head
        + "".join(one(e["question"], e["options"], e["answer_index"]) for e in shots)
        + one(q, choices, None)
    )


def episode_messages(
    concept: str, pools: dict, idx: int, n_demo_pairs: int, seed: int
) -> list:
    """Build a standard episode. Return its messages and its effective target."""
    peers = [
        c for c in held_out_concepts() if c != concept and not same_family(concept, c)
    ]
    row = eval_prefills(pools[concept])[idx % 100]
    rng = random.Random(seed_for("capability", seed, concept, idx))
    cell = next(x for x in ALL_CELLS if x.c_label == idx % 2 and x.prefill_dir == 1)
    ep = build_episode(
        concept,
        cell,
        topic_demos=content(pools[concept]),
        other_demos={c: content(pools[c]) for c in peers},
        topic_prefills=[row],
        other_prefills={c: eval_prefills(pools[c]) for c in peers},
        n_demo_pairs=n_demo_pairs,
        rng=rng,
        system_prompt=None,
    )
    imitate = (1 - ep.cell.c_label) if idx % 2 == 0 else ep.cell.c_label
    ep = with_imitate(ep, imitate)
    return list(ep.messages), ep.effective_target


def build(
    task: str,
    item: dict,
    arm: str,
    concept: str,
    pools: dict,
    idx: int,
    n_demo_pairs: int,
    seed: int,
) -> tuple[list, int | None]:
    """Messages for one question under one arm.

    In the episode arm the benchmark block goes into the instruction turn, not a new user turn."""
    target = None
    msgs: list = []
    if arm == "episode":
        msgs, target = episode_messages(concept, pools, idx, n_demo_pairs, seed)

    block = (
        mmlu_block(item["subject"], item["shots"], item["question"], item["choices"])
        if task == "mmlu"
        else mmlu_pro_block(item["shots"], item["question"], item["choices"])
        if task == "mmlu_pro"
        else trivia_block(item["question"])
        if task == "triviaqa"
        else gsm8k_single_block(item["question"])
    )
    if msgs:
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n\n" + block}
    else:
        msgs = [{"role": "user", "content": block}]
    return msgs, target


def norm_trivia(text: str) -> str:
    """TriviaQA normalisation: lowercase, and remove articles and punctuation."""
    import re

    t = str(text).strip().split("\n")[0].lower()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())


def extract_number(text: str) -> float | None:
    m = re.findall(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text, flags=re.I)
    if not m:
        m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    try:
        return float(m[-1].replace(",", "")) if m else None
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--adapter", default=None, help="omit for the base model")
    ap.add_argument(
        "--task", choices=["mmlu", "mmlu_pro", "gsm8k", "triviaqa"], required=True
    )
    ap.add_argument("--n-questions", type=int, default=0, help="0 = the full split")
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/capability")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    pools = load_pools(args.splits, ALL_CONCEPTS)
    ho = held_out_concepts()

    if args.task == "mmlu":
        ds = load_dataset("cais/mmlu", "all")
        dev: dict[str, list] = {}
        for e in ds["dev"]:
            dev.setdefault(e["subject"], []).append(e)
        test = ds["test"]
        n = len(test) if args.n_questions <= 0 else min(args.n_questions, len(test))
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(test), size=n, replace=False)
        items = [
            {
                "subject": test[int(i)]["subject"],
                "question": test[int(i)]["question"],
                "choices": test[int(i)]["choices"],
                "gold": test[int(i)]["answer"],
                "shots": dev[test[int(i)]["subject"]][:5],
            }
            for i in pick
        ]
    elif args.task == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
        test = ds["test"]
        val = list(ds["validation"])
        rng = np.random.default_rng(args.seed)
        n = len(test) if args.n_questions <= 0 else min(args.n_questions, len(test))
        pick = rng.choice(len(test), size=n, replace=False)
        by_cat: dict[str, list] = {}
        for e in val:
            by_cat.setdefault(e["category"], []).append(e)
        items = [
            {
                "subject": test[int(i)]["category"],
                "question": test[int(i)]["question"],
                "choices": test[int(i)]["options"],
                "gold": test[int(i)]["answer_index"],
                "shots": by_cat.get(test[int(i)]["category"], [])[:5],
            }
            for i in pick
        ]
    elif args.task == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["validation"]
        n = len(ds) if args.n_questions <= 0 else min(args.n_questions, len(ds))
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(ds), size=n, replace=False)
        items = [
            {
                "question": ds[int(i)]["question"],
                "gold": ds[int(i)]["answer"]["value"],
                "aliases": ds[int(i)]["answer"]["normalized_aliases"],
            }
            for i in pick
        ]
    else:
        ds = load_dataset("openai/gsm8k", "main")["test"]
        n = len(ds) if args.n_questions <= 0 else min(args.n_questions, len(ds))
        items = [
            {
                "question": ds[i]["question"],
                "gold": float(
                    ds[i]["answer"].split("####")[-1].strip().replace(",", "")
                ),
            }
            for i in range(n)
        ]

    mdir = model_dir(args.outputs, args.model)
    model, tok = load_model(MODELS[args.model]["hf_name"])
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, mdir / "adapters" / args.adapter)
        model = model.merge_and_unload()
    model.eval()
    tok.padding_side = "left"
    LETTERS_PRO = "ABCDEFGHIJ"
    lset = LETTERS_PRO if args.task == "mmlu_pro" else LETTERS
    letter_ids = [tok.encode(f" {c}", add_special_tokens=False)[-1] for c in lset]

    print(
        f"model={args.model} adapter={args.adapter or 'base'} task={args.task} "
        f"n={len(items)}",
        flush=True,
    )

    results = {}
    t0 = time.time()
    for arm in ("bare", "episode"):
        prompts, targets = [], []
        for i, it in enumerate(items):
            msgs, tgt = build(
                args.task,
                it,
                arm,
                ho[i % len(ho)],
                pools,
                i,
                args.n_demo_pairs,
                args.seed,
            )
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts.append(
                text + ("The answer is" if args.task in ("mmlu", "mmlu_pro") else "")
            )
            targets.append(tgt)

        preds: list = []
        for s in range(0, len(prompts), args.batch_size):
            chunk = prompts[s : s + args.batch_size]
            enc = tok(
                chunk, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            with torch.no_grad():
                if args.task in ("mmlu", "mmlu_pro"):
                    logits = model(**enc).logits[:, -1, :]
                    preds.extend(
                        int(np.argmax([logits[b, i].item() for i in letter_ids]))
                        for b in range(len(chunk))
                    )
                else:
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                    for b in range(len(chunk)):
                        gen = out[b, enc["input_ids"].shape[1] :]
                        txt = tok.decode(gen, skip_special_tokens=True)
                        preds.append(
                            txt if args.task == "triviaqa" else extract_number(txt)
                        )
            if s % (args.batch_size * 20) == 0:
                print(
                    f"  [{arm}] {s + len(chunk)}/{len(prompts)}  "
                    f"{(time.time() - t0) / 60:.0f} min",
                    flush=True,
                )

        golds = [it["gold"] for it in items]
        if args.task == "triviaqa":
            ok = [
                norm_trivia(p) in {norm_trivia(g)} | set(it.get("aliases", []))
                if p
                else False
                for p, g, it in zip(preds, golds, items)
            ]
        elif args.task in ("mmlu", "mmlu_pro"):
            ok = [p is not None and p == g for p, g in zip(preds, golds)]
        else:
            ok = [p is not None and abs(p - g) < 1e-4 for p, g in zip(preds, golds)]
        entry = {"accuracy": float(np.mean(ok)), "n": len(ok)}
        if arm == "episode":
            for name, want in (("suppress", 0), ("elevate", 1)):
                sel = [o for o, t in zip(ok, targets) if t == want]
                entry[f"accuracy_{name}"] = float(np.mean(sel)) if sel else float("nan")
                entry[f"n_{name}"] = len(sel)
        results[arm] = entry
        print(
            f"  {arm:8s} accuracy {entry['accuracy']:.4f}"
            + (
                ""
                if arm == "bare"
                else f"   (suppress {entry['accuracy_suppress']:.4f} n={entry['n_suppress']}, "
                f"elevate {entry['accuracy_elevate']:.4f} n={entry['n_elevate']})"
            ),
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.adapter or 'base'}_{args.task}_single"
    payload = {
        "model": args.model,
        "adapter": args.adapter or "base",
        "task": args.task,
        "shots_format": "single",
        "n_questions": len(items),
        "n_demo_pairs": args.n_demo_pairs,
        "seed": args.seed,
        "episode_concepts": "held-out, cycled by question index",
        "arms": results,
        "delta": results["episode"]["accuracy"] - results["bare"]["accuracy"],
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (args.out / f"{tag}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"\nbare {results['bare']['accuracy']:.4f} -> episode "
        f"{results['episode']['accuracy']:.4f}   delta {payload['delta']:+.4f}"
    )
    print(f"wrote {args.out / f'{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
