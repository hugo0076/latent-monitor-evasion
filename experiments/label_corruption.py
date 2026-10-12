"""Flip each demonstration tag with probability p and measure the paired effect size of the edit.

    python experiments/label_corruption.py --adapter r1_sigmoid_mse --suffix _fine
    python experiments/label_corruption.py

Reads  data/splits/, outputs/<model>/probes/<probe-source>/, outputs/<model>/adapters/<adapter>/
Writes experiments/outputs/label_corruption/<model>_<adapter>.json  (+ .png)
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
from core.concepts import ALL_CONCEPTS, held_out_concepts, same_family  # noqa: E402
from core.episodes import (  # noqa: E402
    ALL_CELLS,
    build_episode,
    content,
    eval_prefills,
    load_pools,
    with_imitate,
)
from core.probes import MonitorSet, seed_for  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--adapter", default=None, help="omit for the base model")
    ap.add_argument("--probe-source", default="base")
    ap.add_argument("--kind", default="logreg")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/label_corruption")
    ap.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    ap.add_argument("--n-prefills", type=int, default=100)
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--suffix",
        default="",
        help="appended to the output file name",
    )
    args = ap.parse_args()

    concepts = held_out_concepts()
    pools = load_pools(args.splits, ALL_CONCEPTS)
    mdir = model_dir(args.outputs, args.model, args.layer)
    monitors = MonitorSet.load(
        probes_dir(args.outputs, args.model, source=args.probe_source, layer=args.layer)
        / f"{args.kind}.npz"
    )

    model, tok = load_model(MODELS[args.model]["hf_name"])
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, mdir / "adapters" / args.adapter)
        model = model.merge_and_unload()
    model.eval()

    def score(messages, prefill) -> np.ndarray:
        h = response_activations(model, tok, monitors.layer, messages, prefill)
        return monitors.logits(h[None, ...], np.ones((1, len(h)), np.float32))[0]

    def peers(c: str) -> list[str]:
        return [x for x in concepts if x != c and not same_family(c, x)]

    n_fwd = len(args.levels) * len(concepts) * args.n_prefills * 2
    print(
        f"model={args.model} adapter={args.adapter or 'base'} "
        f"{len(concepts)} held-out x {args.n_prefills} prefills x {len(args.levels)} levels "
        f"= {n_fwd:,} forwards",
        flush=True,
    )

    t0 = time.time()
    results = []
    for p in args.levels:
        per_concept = {}
        for c in concepts:
            k = monitors.index[c]
            pj = peers(c)
            diffs = []
            for t, row in enumerate(eval_prefills(pools[c])[: args.n_prefills]):
                rng = random.Random(seed_for("eval", args.seed, c, row["item_id"], t))
                cell = next(
                    x for x in ALL_CELLS if x.c_label == t % 2 and x.prefill_dir == 1
                )
                ep = build_episode(
                    c,
                    cell,
                    topic_demos=content(pools[c]),
                    other_demos={m: content(pools[m]) for m in pj},
                    topic_prefills=[row],
                    other_prefills={m: eval_prefills(pools[m]) for m in pj},
                    n_demo_pairs=args.n_demo_pairs,
                    rng=rng,
                    system_prompt=None,
                    corrupt_p=p,
                )
                up = score(with_imitate(ep, ep.cell.c_label).messages, row["sentence"])[
                    k
                ]
                dn = score(
                    with_imitate(ep, 1 - ep.cell.c_label).messages, row["sentence"]
                )[k]
                diffs.append(up - dn)
            d = np.array(diffs)
            per_concept[c] = {
                "cohens_dz": float(d.mean() / d.std(ddof=1))
                if d.std(ddof=1) > 0
                else 0.0,
                "mean_gap": float(d.mean()),
            }
        dz = float(np.mean([v["cohens_dz"] for v in per_concept.values()]))
        gap = float(np.mean([v["mean_gap"] for v in per_concept.values()]))
        results.append(
            {"p": p, "cohens_dz": dz, "mean_gap": gap, "by_concept": per_concept}
        )
        done = (args.levels.index(p) + 1) * len(concepts) * args.n_prefills * 2
        print(
            f"  p={p:.2f}   paired dz {dz:+7.2f}   mean(up-down) {gap:+7.2f}   "
            f"[{done:,}/{n_fwd:,}, {(time.time() - t0) / 60:.0f} min]",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.adapter or 'base'}{args.suffix}"
    (args.out / f"{tag}.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter or "base",
                "kind": args.kind,
                "probe_source": args.probe_source,
                "concepts": concepts,
                "n_prefills": args.n_prefills,
                "n_demo_pairs": args.n_demo_pairs,
                "seed": args.seed,
                "metric": "paired Cohen's dz of (up - down) on each concept's own probe",
                "levels": results,
            },
            indent=2,
        )
        + "\n"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ps = [r["p"] for r in results]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for c in concepts:
            ax.plot(
                ps,
                [r["by_concept"][c]["cohens_dz"] for r in results],
                color="#bbb",
                lw=1,
                zorder=1,
            )
        ax.plot(
            ps,
            [r["cohens_dz"] for r in results],
            "o-",
            color="#c0392b",
            lw=2.4,
            zorder=3,
            label="mean over 10 held-out",
        )
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvline(0.5, color="k", lw=0.8, ls="--")
        ax.text(
            0.5, ax.get_ylim()[1], " labels carry no information", fontsize=8, va="top"
        )
        ax.set_xlabel("label corruption probability $p$")
        ax.set_ylabel("paired Cohen's $d_z$   (up $-$ down)")
        ax.set_title(f"{tag} — the push follows the labels, not the sentences")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out / f"{tag}.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")

    print(f"\nwrote {args.out / f'{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
