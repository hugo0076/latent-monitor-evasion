"""Measure how much the edit direction changes when only the monitored tokens change.

    python experiments/monitored_token_independence.py

Reads  data/splits/, outputs/llama-3.1-8b/probes/base/, outputs/llama-3.1-8b/adapters/<adapter>/
Writes experiments/outputs/monitored_token_independence/<model>.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.activations import (  # noqa: E402
    MODELS,
    load_model,
    model_dir,
    probes_dir,
    response_activations,
)
from core.concepts import (  # noqa: E402
    ALL_CONCEPTS,
    held_out_concepts,
    same_family,
)
from core.episodes import (  # noqa: E402
    ALL_CELLS,
    VerdictTable,
    build_episode,
    content,
    eval_prefills,
    load_pools,
    with_imitate,
)
from core.probes import seed_for  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = "llama-3.1-8b"


def pooled_edit(model, tok, layer, up_msgs, dn_msgs, prefill) -> np.ndarray:
    """dh for one prefill: mean-pooled elevate response minus mean-pooled suppress response."""
    hu = response_activations(model, tok, layer, up_msgs, prefill).mean(0)
    hd = response_activations(model, tok, layer, dn_msgs, prefill).mean(0)
    return (hu - hd).astype(np.float64)


def mean_pairwise_cos(dh: np.ndarray) -> float:
    """Mean off-diagonal cosine among the rows of dh [N, D]."""
    u = dh / (np.linalg.norm(dh, axis=1, keepdims=True) + 1e-12)
    c = u @ u.T
    n = len(u)
    return float((c.sum() - np.trace(c)) / (n * (n - 1)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--n-prefills",
        type=int,
        default=40,
        help="monitored sentences per concept (pairwise cos is over these)",
    )
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--adapter",
        default="r1_sigmoid_mse",
        help="adapter run name, or 'base' for the base model",
    )
    ap.add_argument("--label-source", default="logreg")
    args = ap.parse_args()

    ho = held_out_concepts()
    n_fwd = len(ho) * args.n_prefills * 2 * 2
    print(
        f"{len(ho)} concepts x {args.n_prefills} prefills x 2 arms x 2 modes = "
        f"{n_fwd:,} forwards (~{n_fwd / 8.6 / 60:.0f} min)"
    )

    mdir = model_dir(OUT, MODEL)
    pdir = probes_dir(OUT, MODEL, source="base")
    pools = load_pools(ROOT / "data/splits", ALL_CONCEPTS)
    verdicts = (
        None
        if args.label_source == "topic"
        else VerdictTable(pdir, pools, kind=args.label_source)
    )
    layer = MODELS[MODEL]["layer"]

    model, tok = load_model(MODELS[MODEL]["hf_name"])
    if args.adapter != "base":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, mdir / "adapters" / args.adapter)
        model = model.merge_and_unload()
    model.eval()

    t0 = time.time()
    per_concept = {"strict": {}, "loose": {}}
    for c in ho:
        peers = [x for x in ho if x != c and not same_family(c, x)]
        v = verdicts.for_concept(c) if verdicts else None
        rows = eval_prefills(pools[c])[: args.n_prefills]

        def demos_kwargs():
            return dict(
                topic_demos=content(pools[c]),
                other_demos={m: content(pools[m]) for m in peers},
                topic_prefills=eval_prefills(pools[c]),
                other_prefills={m: eval_prefills(pools[m]) for m in peers},
                n_demo_pairs=args.n_demo_pairs,
                system_prompt=None,
                verdicts=v,
            )

        cell = next(x for x in ALL_CELLS if x.c_label == 1 and x.prefill_dir == 1)
        fixed = build_episode(
            c,
            cell,
            rng=random.Random(seed_for("mti_ctx", args.seed, c)),
            **demos_kwargs(),
        )
        up_msgs = with_imitate(fixed, cell.c_label).messages
        dn_msgs = with_imitate(fixed, 1 - cell.c_label).messages
        dh_strict = np.stack(
            [
                pooled_edit(model, tok, layer, up_msgs, dn_msgs, row["sentence"])
                for row in rows
            ]
        )

        dh_loose = []
        for t, row in enumerate(rows):
            rng = random.Random(seed_for("eval", args.seed, c, row["item_id"], t))
            lcell = next(
                x for x in ALL_CELLS if x.c_label == t % 2 and x.prefill_dir == 1
            )
            ep = build_episode(
                c,
                lcell,
                topic_prefills=[row],
                rng=rng,
                **{k: v2 for k, v2 in demos_kwargs().items() if k != "topic_prefills"},
            )
            up = with_imitate(ep, ep.cell.c_label).messages
            dn = with_imitate(ep, 1 - ep.cell.c_label).messages
            dh_loose.append(pooled_edit(model, tok, layer, up, dn, row["sentence"]))
        dh_loose = np.stack(dh_loose)

        per_concept["strict"][c] = mean_pairwise_cos(dh_strict)
        per_concept["loose"][c] = mean_pairwise_cos(dh_loose)
        print(
            f"  {c:26s} strict {per_concept['strict'][c]:.3f}  "
            f"loose {per_concept['loose'][c]:.3f}  [{(time.time() - t0) / 60:.0f}m]",
            flush=True,
        )

    def summary(mode):
        vals = np.array(list(per_concept[mode].values()))
        rng = np.random.default_rng(0)
        bs = vals[rng.integers(0, len(vals), (2000, len(vals)))].mean(1)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
        }

    d = HERE / "outputs/monitored_token_independence"
    d.mkdir(parents=True, exist_ok=True)
    out = {
        "model": MODEL,
        "adapter": args.adapter,
        "label_source": args.label_source,
        "n_prefills": args.n_prefills,
        "n_demo_pairs": args.n_demo_pairs,
        "concepts": ho,
        "per_concept": per_concept,
        "strict": summary("strict"),
        "loose": summary("loose"),
    }
    (d / f"{MODEL}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"\nstrict {out['strict']['mean']:.3f} +/- "
        f"{(out['strict']['ci95'][1] - out['strict']['ci95'][0]) / 2:.3f}   "
        f"loose {out['loose']['mean']:.3f} +/- "
        f"{(out['loose']['ci95'][1] - out['loose']['ci95'][0]) / 2:.3f}"
    )
    print(f"wrote {d / f'{MODEL}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
