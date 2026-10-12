"""Pseudo-directions: does the edit follow a direction defined only by the tags? Needs a GPU.

    python experiments/in_span_directions.py --corrupt-p 0.5 1.0 --suffix _v2
    python experiments/in_span_directions.py --anchors physics archaeology sycophancy astronomy \
        --cov-betas 0.0 --n-per-beta 0 --corrupt-p 0.5 --corrupt-thetas 90 --suffix _torandom

Reads  outputs/<model>/probes/base/{logreg.npz, pca.npz}, outputs/<model>/activations/base/pooled/
Writes experiments/outputs/in_span_directions/<model><suffix>.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
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
from core.concepts import ALL_CONCEPTS  # noqa: E402
from core.episodes import (  # noqa: E402
    ALL_CELLS,
    build_episode,
    content,
    eval_prefills,
    load_pools,
    with_imitate,
)
from core.geometry import unit  # noqa: E402
from core.probes import MonitorSet, seed_for  # noqa: E402

HERE = Path(__file__).resolve().parent
PAIRS = [
    ("physics", "sarcastic"),
    ("astronomy", "urgency_panic"),
    ("archaeology", "imperatives_commands"),
    ("portuguese", "sycophancy"),
]
THETAS = [0, 30, 45, 60, 90]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--runs", nargs="+", default=["base", "r1_sigmoid_mse"])
    ap.add_argument("--probe-source", default="base")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/in_span_directions")
    ap.add_argument("--thetas", type=int, nargs="+", default=THETAS)
    ap.add_argument("--n-reps", type=int, default=100)
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="fraction of the pool taken from each extreme",
    )
    ap.add_argument(
        "--cov-betas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="random directions Sigma^(beta/2) g: 0 is isotropic, 1 follows the "
        "activation covariance",
    )
    ap.add_argument("--n-per-beta", type=int, default=6)
    ap.add_argument("--corrupt-p", type=float, nargs="*", default=[])
    ap.add_argument("--corrupt-thetas", type=int, nargs="+", default=[0, 45])
    ap.add_argument("--corrupt-reps", type=int, default=50)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--shrinkage", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument(
        "--anchors",
        nargs="*",
        default=[],
        help="rotate from each named concept toward a random direction, "
        "not toward a second concept",
    )
    ap.add_argument(
        "--rotate-to",
        nargs="*",
        default=["isotropic", "cov1.0"],
        help="random target families: isotropic, or cov1.0 (activation-shaped). "
        "cos_dom is the primary measure",
    )
    args = ap.parse_args()

    pdir = probes_dir(args.outputs, args.model, source=args.probe_source)
    bank = MonitorSet.load(pdir / "logreg.npz")
    zf = np.load(pdir / "logreg.npz", allow_pickle=True)
    assert np.all(zf["mu"] == 0) and np.all(zf["sd"] == 1), (
        "logreg bank is standardised"
    )
    assert np.abs(zf["p_b"]).max() == 0.0, "logreg bank has a non-zero intercept"
    W = bank.params["w"].astype(np.float64)
    assert np.allclose(np.linalg.norm(W, axis=1), 1.0), (
        "probe directions are not unit-norm"
    )
    pc1 = np.load(pdir / "pca.npz")["components"][0].astype(np.float64)
    pools = load_pools(args.splits, ALL_CONCEPTS)

    adir = (
        model_dir(args.outputs, args.model)
        / "activations"
        / args.probe_source
        / "pooled"
    )
    H, LAB, ITEM, PRE = [], [], [], []
    for c in ALL_CONCEPTS:
        if not (adir / f"{c}.npy").exists() or c not in bank.names:
            continue
        rows = [
            json.loads(x)
            for x in (args.splits / "nf" / f"{c}.jsonl").read_text().splitlines()
            if x.strip()
        ]
        pos = {r["item_id"]: 150 + i for i, r in enumerate(rows)}
        A = np.load(adir / f"{c}.npy").astype(np.float64)
        for it in content(pools[c]):
            if it["item_id"] in pos:
                H.append(A[pos[it["item_id"]]])
                LAB.append(c)
                ITEM.append(it)
        for it in eval_prefills(pools[c]):
            if it["item_id"] in pos:
                PRE.append((A[pos[it["item_id"]]], it))
    H = np.stack(H)
    LAB = np.array(LAB)
    PH = np.stack([h for h, _ in PRE])
    print(
        f"pool: {len(H)} demo sentences, {len(PRE)} prefill candidates, "
        f"{len(set(LAB))} concepts",
        flush=True,
    )

    Sig = None
    if any(b > 0 for b in args.cov_betas) or any(
        k != "isotropic" for k in args.rotate_to
    ):
        X = H - H.mean(0)
        Sig = X.T @ X / len(X)
        Sig = (1 - args.shrinkage) * Sig + args.shrinkage * np.trace(Sig) / len(
            Sig
        ) * np.eye(len(Sig))
        ev, V = np.linalg.eigh(Sig)

    rng = np.random.default_rng(args.seed)
    D = W.shape[1]

    def draw(t, k):
        """The k highest and k lowest sentences by projection onto t, plus a high prefill."""
        p = H @ t
        m = max(k, int(args.margin * len(p)))
        o = np.argsort(p)
        return o[-m:], o[:m]

    def sample_episode(t, rep):
        r = np.random.default_rng(seed_for("inspan2", args.seed, int(rep), float(t[0])))
        hi, lo = draw(t, args.n_demo_pairs)
        ph = r.choice(hi, args.n_demo_pairs, replace=False)
        pl = r.choice(lo, args.n_demo_pairs, replace=False)
        pre_i = int(np.argsort(PH @ t)[-1 - int(r.integers(0, 20))])
        return ph, pl, pre_i

    def build(ph, pl, pre_i, corrupt_p=0.0):
        posi = [ITEM[i] for i in ph]
        negi = [ITEM[i] for i in pl]
        verd = {it["item_id"]: 1 for it in posi} | {it["item_id"]: 0 for it in negi}
        cell = next(x for x in ALL_CELLS if x.c_label == 1 and x.prefill_dir == 1)
        row = PRE[pre_i][1]
        ep = build_episode(
            LAB[ph[0]],
            cell,
            topic_demos=posi,
            other_demos={"neg": negi},
            topic_prefills=[row],
            other_prefills={"neg": [row]},
            n_demo_pairs=args.n_demo_pairs,
            rng=random.Random(seed_for("inspan2b", args.seed, int(pre_i))),
            system_prompt=None,
            verdicts=verd,
            corrupt_p=corrupt_p,
        )
        return ep, row, [it["item_id"] for it in posi], [it["item_id"] for it in negi]

    def dom_of(ph, pl):
        return H[ph].mean(0) - H[pl].mean(0)

    def perp(v):
        return v - (v @ pc1) * pc1

    geo = {}
    if args.anchors:
        grng = np.random.default_rng(args.seed + 7)
        for anchor in args.anchors:
            wa = unit(W[bank.names.index(anchor)])
            for kind in args.rotate_to:
                g = grng.normal(size=D)
                tgt = (
                    unit(g) if kind == "isotropic" else unit(V @ (ev**0.5 * (V.T @ g)))
                )
                e2 = unit(tgt - (tgt @ wa) * wa)
                geo[f"{anchor}|{kind}"] = {
                    "e1": wa,
                    "e2": e2,
                    "cos_wA_wB": float(wa @ tgt),
                    "angle_deg": float(
                        np.degrees(np.arccos(np.clip(float(wa @ tgt), -1, 1)))
                    ),
                    "e2_vs_wB_deg": float(
                        np.degrees(np.arccos(np.clip(float(e2 @ tgt), -1, 1)))
                    ),
                    "target_max_cos_to_bank": float(np.abs(W @ tgt).max()),
                }
    for a, b in (
        []
        if args.anchors
        else [(x, y) for x, y in PAIRS if x in bank.names and y in bank.names]
    ):
        wa, wb = unit(W[bank.names.index(a)]), unit(W[bank.names.index(b)])
        cab = float(wa @ wb)
        e2 = unit(wb - cab * wa)
        geo[f"{a}|{b}"] = {
            "e1": wa,
            "e2": e2,
            "cos_wA_wB": cab,
            "angle_deg": float(np.degrees(np.arccos(np.clip(cab, -1, 1)))),
            "e2_vs_wB_deg": float(
                np.degrees(np.arccos(np.clip(float(e2 @ wb), -1, 1)))
            ),
        }

    randoms = []
    for beta in args.cov_betas:
        for _ in range(args.n_per_beta):
            g = rng.normal(size=D)
            r = unit(g if beta == 0 else V @ (ev ** (beta / 2) * (V.T @ g)))
            randoms.append((f"cov{beta:.2f}", r))

    def maxcos(r):
        cs = np.abs(W @ unit(r))
        return float(cs.max()), bank.names[int(np.argmax(cs))]

    print(
        "\npre-flight -- cos(difference-of-means, t) and the positive-16 composition:"
    )
    prefl = {}
    for key, g in geo.items():
        for th in args.thetas:
            ct, st = np.cos(np.radians(th)), np.sin(np.radians(th))
            t = unit(ct * g["e1"] + st * g["e2"])
            cs, comp, sets = [], Counter(), []
            for rep in range(20):
                ph, pl, _ = sample_episode(t, rep)
                cs.append(float(unit(perp(dom_of(ph, pl))) @ unit(perp(t))))
                comp.update(LAB[ph])
                sets.append(frozenset(ph.tolist()))
            prefl[f"{key}@{th}"] = {
                "cos_dom_t": float(np.mean(cs)),
                "top_concepts": comp.most_common(3),
                "n_distinct_draws": len(set(sets)),
            }
        s0 = frozenset(sample_episode(unit(g["e1"]), 0)[0].tolist())
        s45 = frozenset(
            sample_episode(
                unit(
                    np.cos(np.radians(45)) * g["e1"] + np.sin(np.radians(45)) * g["e2"]
                ),
                0,
            )[0].tolist()
        )
        ov = len(s0 & s45) / len(s0)
        prefl[f"{key}@overlap_0_45"] = ov
        assert ov < 0.9, (
            f"{key}: 0 and 45 degree positives overlap {ov:.0%} -- sweep is degenerate"
        )
        print(
            f"  {key:40s} cos(DoM,t) "
            + " ".join(f"{prefl[f'{key}@{th}']['cos_dom_t']:.3f}" for th in args.thetas)
            + f"   0/45 overlap {ov:.0%}"
        )

    n_a = len(geo) * len(args.thetas) * args.n_reps * 2 * len(args.runs)
    n_r = len(randoms) * args.n_reps * 2 * len(args.runs)
    n_c = (
        len(args.corrupt_p)
        * len(geo)
        * len(args.corrupt_thetas)
        * args.corrupt_reps
        * 2
        * len(args.runs)
    )
    print(
        f"\nrotation {n_a:,} + random {n_r:,} ({len(randoms)} dirs) + corruption {n_c:,} "
        f"= {n_a + n_r + n_c:,} forwards"
    )

    out = {
        "model": args.model,
        "space": "raw residual stream, no standardisation, logreg only",
        "n_reps": args.n_reps,
        "margin": args.margin,
        "pool_size": len(H),
        "chance_floor": float(1 / np.sqrt(D)),
        "pair_geometry": {
            k: {kk: v[kk] for kk in ("cos_wA_wB", "angle_deg", "e2_vs_wB_deg")}
            for k, v in geo.items()
        },
        "anchors": args.anchors,
        "rotate_to": args.rotate_to,
        "preflight": prefl,
        "predictions": {
            "regression": "cos(dh, dom) flat across theta; components rotate as (cos, sin)",
            "retrieval": "cos(dh, dom) dips at intermediate theta; components pin to the "
            "nearer of e1, e2",
            "corruption": "genuine regression collapses under shuffled labels",
            "random_arms": "if steering needs variance along the target, cov1.0 works where "
            "cov0.0 does not",
        },
        "by_run": {},
    }
    t0 = time.time()
    for run in args.runs:
        mdl, tok = load_model(MODELS[args.model]["hf_name"])
        if run != "base":
            from peft import PeftModel

            mdl = PeftModel.from_pretrained(
                mdl, model_dir(args.outputs, args.model) / "adapters" / run
            )
            mdl = mdl.merge_and_unload()
        mdl.eval()

        def push(ep, row, _m=mdl, _t=tok):
            hs = [
                response_activations(
                    _m,
                    _t,
                    bank.layer,
                    with_imitate(ep, im).messages,
                    row["sentence"],
                    max_length=args.max_length,
                ).mean(0)
                for im in (ep.cell.c_label, 1 - ep.cell.c_label)
            ]
            return (hs[0] - hs[1]).astype(np.float64)

        rot, rnd = [], []
        for key, g in geo.items():
            for th in args.thetas:
                ct, st = np.cos(np.radians(th)), np.sin(np.radians(th))
                t = unit(ct * g["e1"] + st * g["e2"])
                for cp in [0.0] + (
                    list(args.corrupt_p) if th in args.corrupt_thetas else []
                ):
                    nrep = args.n_reps if cp == 0 else args.corrupt_reps
                    dhs, doms = [], []
                    for rep in range(nrep):
                        ph, pl, pi = sample_episode(t, rep)
                        ep, row, _, _ = build(ph, pl, pi, cp)
                        dhs.append(push(ep, row))
                        doms.append(dom_of(ph, pl))
                    dhp = np.stack([perp(v) for v in dhs])
                    rh = unit(perp(np.mean(doms, 0)))
                    rec = {
                        "pair": key,
                        "theta": th,
                        "corrupt_p": cp,
                        "norm": float(np.mean(np.linalg.norm(dhs, axis=1))),
                    }
                    for nm, tgt in (
                        ("cos_dom", rh),
                        ("cos_e1", unit(perp(g["e1"]))),
                        ("cos_e2", unit(perp(g["e2"]))),
                    ):
                        v = np.array([float(unit(x) @ tgt) for x in dhp])
                        bb = (
                            np.random.default_rng(0)
                            .choice(v, (4000, len(v)), replace=True)
                            .mean(1)
                        )
                        rec[nm] = float(v.mean())
                        rec[nm + "_ci"] = [
                            float(np.quantile(bb, 0.025)),
                            float(np.quantile(bb, 0.975)),
                        ]
                    rot.append(rec)
                    print(
                        f"  {run} {key[:26]:26s} th={th:3d} p={cp:.2f} "
                        f"cos_dom {rec['cos_dom']:+.3f} e1 {rec['cos_e1']:+.3f} "
                        f"e2 {rec['cos_e2']:+.3f} [{(time.time() - t0) / 60:.0f}m]",
                        flush=True,
                    )

        for nm, r in randoms:
            mc, near = maxcos(r)
            dhs, doms = [], []
            for rep in range(args.n_reps):
                ph, pl, pi = sample_episode(r, rep)
                ep, row, _, _ = build(ph, pl, pi)
                dhs.append(push(ep, row))
                doms.append(dom_of(ph, pl))
            dhp = np.stack([perp(v) for v in dhs])
            rh = unit(perp(np.mean(doms, 0)))
            cr = np.array([float(unit(x) @ unit(perp(r))) for x in dhp])
            cd = np.array([float(unit(x) @ rh) for x in dhp])
            pr = np.array([float(x @ unit(r)) for x in dhs])
            rnd.append(
                {
                    "arm": nm,
                    "max_cos_to_concept": mc,
                    "nearest_concept": near,
                    "cos_target": float(cr.mean()),
                    "cos_dom": float(cd.mean()),
                    "norm": float(np.mean(np.linalg.norm(dhs, axis=1))),
                    "proj_mean": float(pr.mean()),
                    "proj_sd": float(pr.std(ddof=1)),
                    "cohens_d": float(pr.mean() / pr.std(ddof=1))
                    if pr.std(ddof=1)
                    else 0.0,
                    "cos_dom_t_preflight": float(
                        np.mean(
                            [
                                float(
                                    unit(perp(dom_of(*sample_episode(r, q)[:2])))
                                    @ unit(perp(r))
                                )
                                for q in range(10)
                            ]
                        )
                    ),
                }
            )
            print(
                f"  {run} {nm:10s} maxcos {mc:.3f} cos_target {cr.mean():+.4f} "
                f"cos_dom {cd.mean():+.3f} |dh| {rnd[-1]['norm']:.2f} d {rnd[-1]['cohens_d']:+.3f}"
                f" [{(time.time() - t0) / 60:.0f}m]",
                flush=True,
            )
        out["by_run"][run] = {"rotation": rot, "random": rnd}
        del mdl

    args.out.mkdir(parents=True, exist_ok=True)
    out["elapsed_min"] = round((time.time() - t0) / 60, 1)
    (args.out / f"{args.model}{args.suffix}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(f"\nwrote {args.out / f'{args.model}{args.suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
