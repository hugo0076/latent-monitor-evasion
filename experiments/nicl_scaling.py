"""Measure how the edit's alignment with w changes with the number of demonstrations.

    python experiments/nicl_scaling.py --adapter r1_sigmoid_mse --compare base

Reads  data/splits/, outputs/<model>/probes/<probe-source>/,
       outputs/<model>/activations/<probe-source>/pooled/, outputs/<model>/adapters/<adapter>/
Writes experiments/outputs/nicl_scaling/<model>.json  (+ <model>.png)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.activations import (  # noqa: E402
    MODELS,
    PROMPT,
    load_model,
    model_dir,
    probes_dir,
    response_activations,
)
from core.concepts import (  # noqa: E402
    ALL_CONCEPTS,
    TRAIN_CONCEPTS,
    held_out_concepts,
    same_family,
)
from core.episodes import (  # noqa: E402
    ALL_CELLS,
    Episode,
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
    ap.add_argument("--adapter", default="r1_sigmoid_mse")
    ap.add_argument("--compare", nargs="*", default=["base"])
    ap.add_argument("--probe-source", default="base")
    ap.add_argument("--kind", default="logreg")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/nicl_scaling")
    ap.add_argument("--demo-pairs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n-prefills", type=int, default=30)
    ap.add_argument(
        "--ref-pairs",
        type=int,
        default=16,
        help="pairs in the fixed reference DoM (16 = the standard episode)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="token cap for response_activations. A prompt longer than the cap raises an\n"
        "error. One demonstration pair is about 256 tokens.",
    )
    ap.add_argument(
        "--suffix",
        default="",
        help="appended to the output file name. All levels are subsets of one draw at\n"
        "max(--demo-pairs), so a different maximum changes the demonstrations at every level.",
    )
    args = ap.parse_args()

    ho = held_out_concepts()
    pools = load_pools(args.splits, ALL_CONCEPTS)
    runs = [args.adapter, *args.compare]

    pdir = probes_dir(args.outputs, args.model, source=args.probe_source)
    bank = MonitorSet.load(pdir / f"{args.kind}.npz")
    names, W = bank.names, bank.params["w"].astype(np.float64)

    S = np.load(pdir / f"scores_{args.kind}.npy")
    conc = np.array(
        [
            json.loads(x)["concept"]
            for x in (pdir / "index.jsonl").read_text().splitlines()
            if x.strip()
        ]
    )

    def peers(c: str) -> list[str]:
        grp = TRAIN_CONCEPTS if c in TRAIN_CONCEPTS else held_out_concepts()
        return [x for x in grp if x != c and not same_family(c, x)]

    pc1 = np.load(pdir / "pca.npz")["components"][0].astype(np.float64)

    ben = {c: S[np.isin(conc, peers(c)), names.index(c)] for c in ho}
    thr = {c: float(np.quantile(ben[c], 0.99)) for c in ho}

    def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
        return float(
            roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
        )

    adir = (
        model_dir(args.outputs, args.model)
        / "activations"
        / args.probe_source
        / "pooled"
    )
    pos_of, POOL = {}, {}
    for c in ALL_CONCEPTS:
        rows = [
            json.loads(x)
            for x in (args.splits / "nf" / f"{c}.jsonl").read_text().splitlines()
            if x.strip()
        ]
        pos_of[c] = {r["item_id"]: 150 + i for i, r in enumerate(rows)}
        POOL[c] = np.load(adir / f"{c}.npy")

    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)  # noqa: E731

    MAX_PAIRS = max(args.demo_pairs)

    def full_episode(c: str, t: int):
        """One draw per (concept, prefill), at the largest demonstration count."""
        pj = [x for x in ho if x != c and not same_family(c, x)]
        row = eval_prefills(pools[c])[t]
        rng = random.Random(seed_for("nicl", args.seed, c, t))
        cell = next(x for x in ALL_CELLS if x.c_label == t % 2 and x.prefill_dir == 1)
        return build_episode(
            c,
            cell,
            topic_demos=content(pools[c]),
            other_demos={m: content(pools[m]) for m in pj},
            topic_prefills=[row],
            other_prefills={m: eval_prefills(pools[m]) for m in pj},
            n_demo_pairs=MAX_PAIRS,
            rng=rng,
            system_prompt=None,
        ), row

    def subset(ep: Episode, m_pairs: int) -> Episode:
        """The first `m_pairs` demonstrations of each class, in their original order.

        A level is then a subset of every larger level. Each class is counted separately,
        because a plain prefix of the shuffled messages is not class-balanced.
        """
        on = off = 0
        keep = []
        for d, item in enumerate(ep.demo_ids):
            if item.split(":")[0] == ep.concept:
                if on < m_pairs:
                    on += 1
                    keep.append(d)
            elif off < m_pairs:
                off += 1
                keep.append(d)
        msgs = [m for d in keep for m in ep.messages[2 * d : 2 * d + 2]] + [
            ep.messages[-1]
        ]
        return Episode(
            messages=msgs,
            prefill=ep.prefill,
            concept=ep.concept,
            cell=ep.cell,
            demo_ids=[ep.demo_ids[i] for i in keep],
            prefill_id=ep.prefill_id,
        )

    def get(ids) -> np.ndarray:
        """Mean stored activation over the given demonstration ids."""
        rows = [POOL[i.split(":")[0]][pos_of[i.split(":")[0]][i]] for i in ids]
        return np.stack(rows).astype(np.float64).mean(0)

    def dom_of(ep) -> np.ndarray:
        """Difference of means of this episode's own demonstrations, from stored activations."""
        on = [d for d in ep.demo_ids if d.split(":")[0] == ep.concept]
        off = [d for d in ep.demo_ids if d.split(":")[0] != ep.concept]
        return get(on) - get(off)

    def dom_pool(c: str) -> np.ndarray:
        """Difference of means of the whole content pool of c against the pools of its peers."""
        pj = [x for x in ho if x != c and not same_family(c, x)]
        on = [r["item_id"] for r in content(pools[c])]
        off = [r["item_id"] for m in pj for r in content(pools[m])]
        return get(on) - get(off)

    out: dict = {"demo_pairs": args.demo_pairs, "by_run": {}}
    t0 = time.time()
    for run in runs:
        model, tok = load_model(MODELS[args.model]["hf_name"])
        if run and run != "base":
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, model_dir(args.outputs, args.model) / "adapters" / run
            )
            model = model.merge_and_unload()
        model.eval()
        layer = MODELS[args.model]["layer"]

        def make_z(mdl, tk):
            def z_of(messages, prefill, k):
                h = response_activations(
                    mdl, tk, layer, messages, prefill, max_length=args.max_length
                )
                return h.mean(0).astype(np.float64), float(
                    bank.logits(h[None], np.ones((1, len(h)), np.float32))[0][k]
                )

            return z_of

        z_of = make_z(model, tok)

        tpr_base = {}
        for c in ho:
            k = names.index(c)
            zb = np.array(
                [
                    z_of(PROMPT, r["sentence"], k)[1]
                    for r in eval_prefills(pools[c])[: args.n_prefills]
                ]
            )
            tpr_base[c] = float((zb > thr[c]).mean())

        draws = {(c, t): full_episode(c, t) for c in ho for t in range(args.n_prefills)}
        dom_ref = {
            (c, t): dom_of(subset(draws[(c, t)][0], args.ref_pairs)) for c, t in draws
        }
        dom_inf = {c: dom_pool(c) for c in ho}

        levels: dict[str, dict] = {}
        for n_pairs in args.demo_pairs:
            cw, cd, dw, nrm, au, ts, dzs = [], [], [], [], [], [], []
            cref, cinf, dref_w, dinf_w = [], [], [], []
            pe_w, pe_ref, pe_inf, pe_dom, pe_norm = [], [], [], [], []
            pe_w_p, pe_ref_p, pe_inf_p, pe_norm_p, pe_ceil = [], [], [], [], []
            for c in ho:
                k = names.index(c)
                w = W[k]
                dhs, doms, drefs, z_up, z_dn = [], [], [], [], []
                for t in range(args.n_prefills):
                    full, row = draws[(c, t)]
                    ep = subset(full, n_pairs)
                    hu, zu = z_of(
                        with_imitate(ep, ep.cell.c_label).messages, row["sentence"], k
                    )
                    hd, zd = z_of(
                        with_imitate(ep, 1 - ep.cell.c_label).messages,
                        row["sentence"],
                        k,
                    )
                    dhs.append(hu - hd)
                    doms.append(dom_of(ep))
                    drefs.append(dom_ref[(c, t)])
                    z_up.append(zu)
                    z_dn.append(zd)
                Dh = np.array(dhs)
                un = lambda M: M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-12)  # noqa
                pe_w.append(float(np.mean(un(Dh) @ unit(w))))
                pe_ref.append(float(np.mean(un(Dh) @ unit(dom_ref[(c, 0)]))))
                pe_inf.append(float(np.mean(un(Dh) @ unit(dom_inf[c]))))
                pe_dom.append(
                    float(
                        np.mean([un(Dh)[q] @ unit(doms[q]) for q in range(len(doms))])
                    )
                )
                pe_norm.append(float(np.linalg.norm(Dh, axis=1).mean()))
                Dp = Dh - np.outer(Dh @ pc1, pc1)
                pe_w_p.append(float(np.mean(un(Dp) @ unit(w))))
                pe_ref_p.append(float(np.mean(un(Dp) @ unit(dom_ref[(c, 0)]))))
                pe_inf_p.append(float(np.mean(un(Dp) @ unit(dom_inf[c]))))
                pe_norm_p.append(float(np.linalg.norm(Dp, axis=1).mean()))
                Dm = np.array(doms)
                pe_ceil.append(float(np.mean(un(Dm) @ unit(w))))
                dh, dom, dref = np.mean(dhs, 0), np.mean(doms, 0), np.mean(drefs, 0)
                dn = np.array(z_dn)
                d = np.array(z_up) - dn
                cw.append(unit(dh) @ unit(w))
                cd.append(unit(dh) @ unit(dom))
                dw.append(unit(dom) @ unit(w))
                cref.append(unit(dh) @ unit(dref))
                cinf.append(unit(dh) @ unit(dom_inf[c]))
                dref_w.append(unit(dref) @ unit(w))
                dinf_w.append(unit(dom_inf[c]) @ unit(w))
                nrm.append(float(np.linalg.norm(dh)))
                au.append(auroc(dn, ben[c]))
                ts.append(float((dn > thr[c]).mean()))
                dzs.append(float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else 0.0)
            levels[str(n_pairs)] = {
                "cos_dh_w": float(np.mean(cw)),
                "cos_dh_dom": float(np.mean(cd)),
                "ceiling_cos_dom_w": float(np.mean(dw)),
                "norm": float(np.mean(nrm)),
                "frac_of_ceiling": float(np.mean(cw) / np.mean(dw)),
                "cos_dh_dom_ref": float(np.mean(cref)),
                "cos_dh_dom_pool": float(np.mean(cinf)),
                "ceiling_ref_w": float(np.mean(dref_w)),
                "ceiling_pool_w": float(np.mean(dinf_w)),
                "evasion_auroc": float(np.mean(au)),
                "tpr_suppressed": float(np.mean(ts)),
                "tpr_baseline": float(np.mean(list(tpr_base.values()))),
                "cohens_dz": float(np.mean(dzs)),
                "pe_cos_dh_w": float(np.mean(pe_w)),
                "pe_cos_dh_dom_ref": float(np.mean(pe_ref)),
                "pe_cos_dh_dom_pool": float(np.mean(pe_inf)),
                "pe_cos_dh_dom_n": float(np.mean(pe_dom)),
                "pe_norm": float(np.mean(pe_norm)),
                "pe_perp_cos_dh_w": float(np.mean(pe_w_p)),
                "pe_perp_cos_dh_dom_ref": float(np.mean(pe_ref_p)),
                "pe_perp_cos_dh_dom_pool": float(np.mean(pe_inf_p)),
                "pe_perp_norm": float(np.mean(pe_norm_p)),
                "pe_ceiling_cos_dom_w": float(np.mean(pe_ceil)),
                "pe_frac_of_ceiling": float(np.mean(pe_w) / np.mean(pe_ceil)),
            }
            v = levels[str(n_pairs)]
            print(
                f"  {run:16s} {2 * n_pairs:3d} demos  ||dh|| {v['norm']:6.2f}  "
                f"cos(dh,w) {v['cos_dh_w']:+.3f} ({100 * v['frac_of_ceiling']:.0f}% ceil)  "
                f"cos(dh,DoM_n) {v['cos_dh_dom']:+.3f}  "
                f"cos(dh,DoM_{2 * args.ref_pairs}) {v['cos_dh_dom_ref']:+.3f}  "
                f"cos(dh,DoM_inf) {v['cos_dh_dom_pool']:+.3f}  | PER-EP "
                f"cos(dh,w) {v['pe_cos_dh_w']:+.3f} "
                f"cos(dh,DoM_32) {v['pe_cos_dh_dom_ref']:+.3f} "
                f"cos(dh,DoM_inf) {v['pe_cos_dh_dom_pool']:+.3f} |dh| {v['pe_norm']:6.2f}  "
                f"dz {v['cohens_dz']:+6.2f}  AUROC {v['evasion_auroc']:.3f}  "
                f"TPR {v['tpr_suppressed']:.3f}/{v['tpr_baseline']:.3f}"
                f"  [{(time.time() - t0) / 60:.0f}m]",
                flush=True,
            )
        out["by_run"][run or "base"] = levels
        del model

    args.out.mkdir(parents=True, exist_ok=True)
    out.update(
        {
            "model": args.model,
            "n_prefills": args.n_prefills,
            "concepts": ho,
            "elapsed_min": round((time.time() - t0) / 60, 1),
        }
    )
    (args.out / f"{args.model}{args.suffix}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(f"\nwrote {args.out / f'{args.model}{args.suffix}.json'}")
    return plot(out, args.out)


def plot(out: dict, outdir: Path) -> int:
    """Plot six metrics against the number of demonstrations."""
    runs, ns = list(out["by_run"]), out["demo_pairs"]
    panels = [
        ("norm", r"push magnitude  $\|\Delta h\|$", None),
        ("cos_dh_w", r"aim  $\cos(\Delta h,\, w)$", "ceiling_cos_dom_w"),
        (
            "cos_dh_dom_ref",
            r"convergence  $\cos(\Delta h,\, \mathrm{DoM}_{32})$",
            "cos_dh_dom_pool",
        ),
        ("cohens_dz", "paired Cohen's $d_z$", None),
        ("evasion_auroc", "evasion AUROC   (low = evaded)", 0.5),
        ("tpr_suppressed", "TPR @ 1% FPR   (low = evaded)", "tpr_baseline"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    for ax, (field, label, ref) in zip(axes, panels):
        for run, colour in zip(runs, ["#d62728", "#1f77b4", "#2ca02c"]):
            ax.plot(
                ns,
                [out["by_run"][run][str(n)][field] for n in ns],
                "-o",
                color=colour,
                ms=4.5,
                lw=1.8,
                label=run,
            )
            if isinstance(ref, str):
                ax.plot(
                    ns,
                    [out["by_run"][run][str(n)][ref] for n in ns],
                    ls=(0, (1, 2)),
                    color=colour,
                    lw=1.3,
                )
        if not isinstance(ref, str) and ref is not None:
            ax.axhline(ref, color="k", ls=(0, (1, 2)), lw=1.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(2 * n) for n in ns])
        ax.set_xlabel("demonstrations")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("held-out concepts, mean over 10")
    axes[1].set_ylabel("dotted = DoM ceiling", fontsize=8)
    axes[2].set_ylabel("dotted = population DoM", fontsize=8)
    axes[-1].set_ylabel("dotted = baseline TPR", fontsize=8)
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{out['model']} — in-context demonstrations vs the push and the outcome",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = outdir / f"{out['model']}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
