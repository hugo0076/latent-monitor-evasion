"""Figure 2 (Section 4.1) and the Appendix E.1 table: the base edit, amplified. CPU only.

    python figures/fig_results.py
    python figures/fig_results.py --models llama-3.1-8b

Reads  outputs/<model>/evals/{base,r1_sigmoid_mse}.npz, outputs/<model>/probes/base/
Writes figures/outputs/amplify.{pdf,png}, and figures/outputs/curves.npz for --replot
Prints the per-model TPR by α that the Appendix E.1 table reports.

Each episode's edit Δh = h_up - h_down is scaled by α and applied to the single-turn activations
of the same response, h(α) = h_bare - α Δh. This excludes the in-episode context shift, and
α = 1 is the edit that the model produces. Each episode keeps its own edit, because a mean over
episodes cancels per-episode aim noise.

    on target    concept i's edit, read by concept i's monitor (solid)
    off target   concept i's edit on concept j's content, read by j's monitor, mean over j (dashed)
    random       a random unit direction per episode at the same norm (grey band, 5-95%, all models)
    oracle       the monitor direction at the same norm (only with --oracle)

Bands are 95% bootstrap intervals over the 10 held-out concepts. The diamond is the adapter,
at the α that matches its edit norm and at the organism's measured in-episode TPR or AUROC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from core.activations import MODELS, model_dir, probes_dir  # noqa: E402
from core.concepts import TRAIN_CONCEPTS, held_out_concepts  # noqa: E402

HERE = Path(__file__).resolve().parent
N_BOOT = 2000
SHORT = {
    "llama-3.1-8b": "Llama-3.1-8B",
    "gemma-3-27b": "Gemma-3-27B",
    "llama-3.3-70b": "Llama-3.3-70B",
    "qwen3-14b": "Qwen3-14B",
}
COLOUR = style.MODEL_COLOR
ADAPTER = {
    "llama-3.1-8b": "r1_sigmoid_mse",
    "gemma-3-27b": "r1_sigmoid_mse",
    "qwen3-14b": "r1_sigmoid_mse",
    "llama-3.3-70b": "r1_sigmoid_mse",
}


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def boot(per_concept: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(per_concept)
    d = per_concept[rng.integers(0, n, (N_BOOT, n))].mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def collect(
    model: str,
    kind: str,
    outputs: Path,
    n_random: int,
    alphas: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Every curve for one model, on the shared amplification axis."""
    mdir, pdir = model_dir(outputs, model), probes_dir(outputs, model, source="base")
    z = np.load(mdir / "evals" / "base.npz", allow_pickle=True)
    cs = [str(x) for x in z["concepts"]]
    roster = [str(x) for x in z["probes"]][: len(z[f"probe_{kind}_scale"])]
    W = z[f"probe_{kind}_w"].astype(np.float64)
    THR = z[f"probe_{kind}_threshold"].astype(np.float64)
    SCA = z[f"probe_{kind}_scale"].astype(np.float64)
    D = W.shape[1]

    S = np.load(pdir / f"scores_{kind}.npy")
    idxr = [
        json.loads(x)
        for x in (pdir / "index.jsonl").read_text().splitlines()
        if x.strip()
    ]
    conc = np.array([r["concept"] for r in idxr])
    ho = [c for c in held_out_concepts() if c in cs]

    out = {
        m: {d: np.zeros((len(alphas), len(ho))) for d in ("own", "oracle")}
        for m in ("tpr", "auroc")
    }
    for m in ("tpr", "auroc"):
        out[m]["random"] = np.zeros((len(alphas), len(ho), n_random))
    norms = []

    for ci, c in enumerate(ho):
        i, k = cs.index(c), roster.index(c)
        grp = TRAIN_CONCEPTS if c in TRAIN_CONCEPTS else held_out_concepts()
        ben = S[np.isin(conc, [x for x in grp if x != c and x in cs]), roster.index(c)]
        thr = float(np.quantile(ben, 0.99))
        y = None

        hb = np.asarray(z["h_bare"])[i].astype(np.float64)
        dh = (np.asarray(z["h_up"])[i] - np.asarray(z["h_down"])[i]).astype(np.float64)
        w = W[k]
        norms.append(float(np.linalg.norm(dh.mean(0))))

        z_bare = hb @ w
        p_own = dh @ w
        mags = np.linalg.norm(dh, axis=1)
        G = rng.standard_normal((n_random, len(mags), D))
        p_rand = (unit(G) @ w) * mags
        del G
        y = np.r_[np.ones(len(z_bare)), np.zeros(len(ben))]

        def metrics(raw: np.ndarray) -> tuple[float, float]:
            zz = (raw - THR[k]) / SCA[k]
            return float((zz > thr).mean()), float(roc_auc_score(y, np.r_[zz, ben]))

        for ai, a in enumerate(alphas):
            out["tpr"]["own"][ai, ci], out["auroc"]["own"][ai, ci] = metrics(
                z_bare - a * p_own
            )
            out["tpr"]["oracle"][ai, ci], out["auroc"]["oracle"][ai, ci] = metrics(
                z_bare - a * mags
            )
            for r in range(n_random):
                t, u_ = metrics(z_bare - a * p_rand[r])
                out["tpr"]["random"][ai, ci, r] = t
                out["auroc"]["random"][ai, ci, r] = u_

    # an off-diagonal cell pairs prefill r of the donor edit with prefill r of the target
    pvec = {}
    for c in ho:
        i = cs.index(c)
        pvec[c] = (np.asarray(z["h_up"])[i] - np.asarray(z["h_down"])[i]).astype(
            np.float64
        )
    spec = {
        m: {d: np.zeros((len(alphas), len(ho))) for d in ("diag", "off")}
        for m in ("tpr", "auroc")
    }
    for cj, c in enumerate(ho):
        j, k = cs.index(c), roster.index(c)
        grp = TRAIN_CONCEPTS if c in TRAIN_CONCEPTS else held_out_concepts()
        ben = S[np.isin(conc, [x for x in grp if x != c and x in cs]), roster.index(c)]
        thr = float(np.quantile(ben, 0.99))
        y = np.r_[np.ones(np.asarray(z["h_bare"]).shape[1]), np.zeros(len(ben))]
        zb_ = np.asarray(z["h_bare"])[j].astype(np.float64) @ W[k]
        proj = {ci_name: pvec[ci_name] @ W[k] for ci_name in ho}
        for ai, a in enumerate(alphas):
            offs = {"tpr": [], "auroc": []}
            for ci, ci_name in enumerate(ho):
                zz = (zb_ - a * proj[ci_name] - THR[k]) / SCA[k]
                t = float((zz > thr).mean())
                u_ = float(roc_auc_score(y, np.r_[zz, ben]))
                if ci_name == c:
                    spec["tpr"]["diag"][ai, cj], spec["auroc"]["diag"][ai, cj] = t, u_
                else:
                    offs["tpr"].append(t)
                    offs["auroc"].append(u_)
            spec["tpr"]["off"][ai, cj] = np.mean(offs["tpr"])
            spec["auroc"]["off"][ai, cj] = np.mean(offs["auroc"])

    res = {"ho": ho, "D": D, "norm": float(np.mean(norms)), "curves": out, "spec": spec}

    if model in ADAPTER and (mdir / "evals" / f"{ADAPTER[model]}.npz").exists():
        za = np.load(mdir / "evals" / f"{ADAPTER[model]}.npz", allow_pickle=True)
        acs = [str(x) for x in za["concepts"]]
        za_kinds = [str(x) for x in za["kinds"]]
        za_probes = [str(x) for x in za["probes"]]
        pts = {"tpr": [], "auroc": []}
        an = []
        for c in ho:
            if c not in acs:
                continue
            ia = acs.index(c)
            grp = TRAIN_CONCEPTS if c in TRAIN_CONCEPTS else held_out_concepts()
            ben = S[
                np.isin(conc, [x for x in grp if x != c and x in cs]), roster.index(c)
            ]
            thr = float(np.quantile(ben, 0.99))
            # x: α = adapter edit norm / base edit norm (norms of the mean edit per concept)
            da = (np.asarray(za["h_up"])[ia] - np.asarray(za["h_down"])[ia]).astype(
                np.float64
            )
            an.append(float(np.linalg.norm(da.mean(0))))
            # y is the organism's measured suppress arm, not its edit re-applied to h_bare
            col = next(
                j
                for j, (p, kk) in enumerate(zip(za_probes, za_kinds))
                if p == c and kk == kind
            )
            dn = np.asarray(za["down"])[ia, :, col].astype(np.float64)
            pts["tpr"].append(float((dn > thr).mean()))
            pts["auroc"].append(
                float(
                    roc_auc_score(
                        np.r_[np.ones(len(dn)), np.zeros(len(ben))], np.r_[dn, ben]
                    )
                )
            )
        res["adapter"] = {
            "alpha": float(np.mean(an)) / res["norm"],
            "tpr": np.array(pts["tpr"]),
            "auroc": np.array(pts["auroc"]),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["llama-3.1-8b", "gemma-3-27b", "qwen3-14b", "llama-3.3-70b"],
        choices=sorted(MODELS),
    )
    ap.add_argument("--kind", default="logreg")
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="draw from figures/outputs/curves.npz, computing only the models it lacks",
    )
    ap.add_argument(
        "--oracle",
        action="store_true",
        help="also draw the oracle (monitor-direction) curve",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    style.apply()
    rng = np.random.default_rng(args.seed)
    alphas = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "curves.npz"
    got = {}
    if args.replot and cache.exists():
        raw = np.load(cache, allow_pickle=True)
        got = {k: v for k, v in raw["got"].item().items() if k in args.models}
        alphas = raw["alphas"]
        print(f"replotting from {cache}  ({', '.join(got)})")
    todo = [m for m in args.models if m not in got]
    for m in todo:
        if not (model_dir(args.outputs, m) / "evals" / "base.npz").exists():
            print(f"  (no base eval for {m})")
            continue
        got[m] = collect(m, args.kind, args.outputs, args.n_random, alphas, rng)
        r = got[m]
        print(
            f"{SHORT.get(m, m):16s} ||dh|| {r['norm']:8.2f}  D={r['D']}  "
            f"n_concepts={len(r['ho'])}"
            + (f"  adapter at a={r['adapter']['alpha']:.1f}" if "adapter" in r else "")
        )
        for ai, a in enumerate(alphas):
            c = r["curves"]
            print(
                f"    a={a:6.2f}  own TPR {c['tpr']['own'][ai].mean():.3f}  "
                f"oracle {c['tpr']['oracle'][ai].mean():.3f}  "
                f"rand {c['tpr']['random'][ai].mean():.3f}   |   "
                f"AUROC own {c['auroc']['own'][ai].mean():.3f}  "
                f"oracle {c['auroc']['oracle'][ai].mean():.3f}"
            )

    if todo:
        np.savez(cache, got=got, alphas=alphas)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, met, lab in zip(axes, ("tpr", "auroc"), ("TPR @ 1% FPR", "AUROC")):
        rnd_all = []
        for m, r in got.items():
            col, c, sp = COLOUR.get(m, "#333333"), r["curves"], r["spec"]
            on = sp[met]["diag"].mean(1)
            lo = np.array(
                [boot(sp[met]["diag"][i], rng)[0] for i in range(len(alphas))]
            )
            hi = np.array(
                [boot(sp[met]["diag"][i], rng)[1] for i in range(len(alphas))]
            )
            ax.plot(alphas, on, "o-", color=col, lw=2.1, ms=5, label=SHORT.get(m, m))
            ax.fill_between(alphas, lo, hi, color=col, alpha=0.16)
            ax.plot(alphas, sp[met]["off"].mean(1), "--", color=col, lw=1.5, alpha=0.9)
            if args.oracle:
                ax.plot(
                    alphas, c[met]["oracle"].mean(1), ":", color=col, lw=1.5, alpha=0.75
                )
            rnd_all.append(c[met]["random"].reshape(len(alphas), -1))
            if "adapter" in r:
                ax.plot(
                    [r["adapter"]["alpha"]],
                    [r["adapter"][met].mean()],
                    "D",
                    ms=8,
                    color=col,
                    alpha=0.55,
                    mec="#333333",
                    mew=0.8,
                    zorder=6,
                )
        R = np.concatenate(rnd_all, axis=1)
        ax.fill_between(
            alphas,
            np.quantile(R, 0.05, axis=1),
            np.quantile(R, 0.95, axis=1),
            color="#8c8c8c",
            alpha=0.22,
            lw=0,
            zorder=1,
        )
        ax.axvline(1.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_ylabel(lab)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.22)
    shapes = [("-", 2.1, "o"), ("--", 1.5, None)]
    names = ["on target", "off target (mean over 9)"]
    if args.oracle:
        shapes.append((":", 1.5, None))
        names.append("oracle (monitor direction)")
    handles = [
        plt.Line2D([], [], color="k", ls=ls, lw=lw, marker=mk) for ls, lw, mk in shapes
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="#8c8c8c", alpha=0.30, ec="none"))
    names.append(f"random direction (n={args.n_random}, 5-95%)")
    handles.append(
        plt.Line2D(
            [],
            [],
            color="#888888",
            marker="D",
            ls="",
            ms=6,
            alpha=0.7,
            mec="#333333",
            mew=0.8,
        )
    )
    names.append("rank-1 adapter")
    axes[1].legend(handles, names, fontsize=6.5, loc="lower left")
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17)
    fig.supxlabel(
        r"Amplification of the model's own edit ($\alpha$)",
        fontsize=style.FS["label"],
        y=0.025,
    )
    fig.canvas.draw()
    for ax, letter in zip(axes, "ab"):
        e = ax.get_window_extent()
        x0 = e.x0 / (fig.get_figwidth() * fig.dpi)
        y1 = e.y1 / (fig.get_figheight() * fig.dpi)
        fig.text(
            x0 - 0.052,
            min(y1 + 0.070, 0.998),
            letter,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"amplify.{ext}", bbox_inches="tight")
    print(f"\nwrote {args.out / 'amplify.pdf'} (+.png)")

    print(
        f"\n{'model':16s} {'alpha':>6s} {'on':>7s} {'off':>7s} {'oracle':>7s} {'rand':>7s}"
    )
    for m, r in got.items():
        for ai, a in enumerate(alphas):
            print(
                f"{SHORT.get(m, m):16s} {a:6.2f} {r['spec']['tpr']['diag'][ai].mean():7.3f} "
                f"{r['spec']['tpr']['off'][ai].mean():7.3f} "
                f"{r['curves']['tpr']['oracle'][ai].mean():7.3f} "
                f"{r['curves']['tpr']['random'][ai].mean():7.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
