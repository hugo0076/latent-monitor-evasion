"""Split the 500 sentences of each concept into 150 probe-train and 350 neurofeedback (not fitted).

    python 01_split_pools.py

Reads  data/raw/{concept}.jsonl
Writes data/splits/probe_train/{concept}.jsonl   150 per concept, to fit the probes
       data/splits/nf/{concept}.jsonl            350 per concept, never used to fit a probe

The nf sentences supply all probe evaluation and all episode content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.concepts import ALL_CONCEPTS

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw", type=Path, default=HERE / "data/raw")
    ap.add_argument("--out", type=Path, default=HERE / "data/splits")
    ap.add_argument("--probe-train-n", type=int, default=150)
    ap.add_argument("--nf-n", type=int, default=350)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    needed = args.probe_train_n + args.nf_n
    for split in ("probe_train", "nf"):
        (args.out / split).mkdir(parents=True, exist_ok=True)

    for concept in ALL_CONCEPTS:
        src = args.raw / f"{concept}.jsonl"
        rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
        if len(rows) < needed:
            raise ValueError(
                f"{concept}: need {needed} sentences, {src} has {len(rows)}"
            )
        for i, row in enumerate(rows):
            row["item_id"] = f"{concept}:{i:04d}"

        idx = np.arange(len(rows))
        np.random.default_rng(args.seed).shuffle(idx)
        halves = {
            "probe_train": [rows[i] for i in idx[: args.probe_train_n]],
            "nf": [rows[i] for i in idx[args.probe_train_n : needed]],
        }
        for split, items in halves.items():
            path = args.out / split / f"{concept}.jsonl"
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items)
            )

    print(
        f"{len(ALL_CONCEPTS)} concepts split {args.probe_train_n}/{args.nf_n} "
        f"(seed {args.seed}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
