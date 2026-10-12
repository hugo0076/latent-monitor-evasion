"""Build specificity matrices for the base model, its amplified edit and the organism. CPU only.

    python experiments/grid_compare.py --gains 1 8 32
    python experiments/grid_compare.py --base base_grid20 --lora r1_sigmoid_mse_grid20 \
        --gains 4 6 8 10 12 16 24 32 --out experiments/outputs/grid_compare_grid20

Reads  outputs/<model>/evals/{<base>,<lora>}.npz, outputs/<model>/probes/<probe-source>/
Writes experiments/outputs/grid_compare/<model>.{json,png}
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
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.activations import MODELS, model_dir, probes_dir  # noqa: E402
from core.concepts import TRAIN_CONCEPTS, held_out_concepts, same_family  # noqa: E402

HERE = Path(__file__).resolve().parent
SHORT = {
    "evasiveness_noncommittal": "evasiveness",
    "imperatives_commands": "imperatives",
    "urgency_panic": "urgency",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--base", default="base")
    ap.add_argument("--lora", default="r1_sigmoid_mse")
    ap.add_argument(
        "--gains", type=float, nargs="+", required=True, help="the alpha values"
    )
    ap.add_argument("--probe-source", default="base")
    ap.add_argument("--kind", default="logreg")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/grid_compare")
    args = ap.parse_args()

    mdir = model_dir(args.outputs, args.model)
    pdir = probes_dir(args.outputs, args.model, source=args.probe_source)
    z = {
        r: np.load(mdir / "evals" / f"{r}.npz", allow_pickle=True)
        for r in (args.base, args.lora)
    }
    z0 = z[args.base]
    cs = [str(x) for x in z0["concepts"]]
    grid = [str(x) for x in z0["grid"]]
    roster = [str(x) for x in z0["probes"]][: len(z0[f"probe_{args.kind}_scale"])]
    W = z0[f"probe_{args.kind}_w"].astype(np.float64)
    THR = z0[f"probe_{args.kind}_threshold"].astype(np.float64)
    SC = z0[f"probe_{args.kind}_scale"].astype(np.float64)
    n = len(grid)

    S = np.load(pdir / f"scores_{args.kind}.npy")
    idxr = [
        json.loads(x)
        for x in (pdir / "index.jsonl").read_text().splitlines()
        if x.strip()
    ]
    conc = np.array([r["concept"] for r in idxr])

    def peers(c):
        grp = TRAIN_CONCEPTS if c in TRAIN_CONCEPTS else held_out_concepts()
        return [x for x in grp if x != c and not same_family(c, x) and x in cs]

    ben = {c: S[np.isin(conc, peers(c)), roster.index(c)] for c in grid}
    thr = {c: float(np.quantile(ben[c], 0.99)) for c in grid}

    def matrices(run: str, gain: float | None = None):
        """TPR, AUROC and dz matrices over the grid. `gain` is alpha, or None for the stored arms."""
        zz = z[run]
        hb = np.asarray(zz["h_bare"]).astype(np.float64)
        gu = np.asarray(zz["h_grid_up"]).astype(np.float64)
        gd = np.asarray(zz["h_grid_down"]).astype(np.float64)
        TP, AU, DZ = (np.zeros((n, n)) for _ in range(3))
        for a, ci in enumerate(grid):
            i, k = cs.index(ci), roster.index(ci)
            for b in range(n):
                if gain is None:
                    up, dn = gu[a, b], gd[a, b]
                else:
                    up, dn = hb[i], hb[i] - gain * (gu[a, b] - gd[a, b])
                zu = (up @ W[k] - THR[k]) / SC[k]
                zd = (dn @ W[k] - THR[k]) / SC[k]
                d = zu - zd
                TP[a, b] = float((zd > thr[ci]).mean())
                AU[a, b] = roc_auc_score(
                    np.r_[np.ones(len(zd)), np.zeros(len(ben[ci]))], np.r_[zd, ben[ci]]
                )
                DZ[a, b] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else 0.0
        return {"tpr": TP, "auroc": AU, "dz": DZ}

    lora = matrices(args.lora)
    target = float(np.diag(lora["tpr"]).mean())
    print(
        f"LoRA diagonal TPR = {target:.4f}  (the level the amplified base must match)"
    )

    gains = args.gains
    panels = {f"{args.base} (unamplified)": matrices(args.base, None)}
    for g in gains:
        panels[f"{args.base} a={g} (set)"] = matrices(args.base, g)
    panels[args.lora] = lora

    off = ~np.eye(n, dtype=bool)
    print(f"\n{'panel':30s} {'metric':7s} {'diag':>8s} {'off':>8s} {'ratio':>8s}")
    summary = {}
    for name, M in panels.items():
        summary[name] = {}
        for f in ("tpr", "auroc", "dz"):
            d, o = float(np.diag(M[f]).mean()), float(M[f][off].mean())
            summary[name][f] = {"diag": d, "off": o}
            print(f"{name:30s} {f:7s} {d:8.3f} {o:8.3f} {o / (d + 1e-9):8.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "grid": grid,
                "gains": gains,
                "lora_diag_tpr": target,
                "summary": summary,
                "matrices": {
                    k: {f: v[f].tolist() for f in v} for k, v in panels.items()
                },
            },
            indent=2,
        )
        + "\n"
    )

    lab = [SHORT.get(c, c)[:9] for c in grid]
    fig, axes = plt.subplots(
        3, len(panels), figsize=(4.4 * len(panels), 12.4), squeeze=False
    )
    for ri, (f, title, cmap, vlim) in enumerate(
        [
            ("tpr", "TPR @ 1% FPR", "viridis_r", (0, 1)),
            ("auroc", "evasion AUROC", "viridis_r", (0, 1)),
            ("dz", "paired $d_z$", "magma", None),
        ]
    ):
        vmax = (
            max(np.abs(M[f]).max() for M in panels.values())
            if vlim is None
            else vlim[1]
        )
        vmin = 0 if vlim else -vmax
        for ci_, (name, M) in enumerate(panels.items()):
            ax = axes[ri][ci_]
            im = ax.imshow(M[f], cmap=cmap, vmin=vmin, vmax=vmax)
            for a in range(n):
                ax.add_patch(
                    Rectangle((a - 0.5, a - 0.5), 1, 1, fill=False, ec="k", lw=1.4)
                )
            ax.set_xticks(range(n))
            ax.set_xticklabels(lab, rotation=90, fontsize=6)
            ax.set_yticks(range(n))
            ax.set_yticklabels(lab if ci_ == 0 else [], fontsize=6)
            ax.set_title(f"{name}\n{title}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"{args.model} — design B: content $i$ (rows) x instruction $j$ (cols). "
        f"Amplified base matched to the LoRA's diagonal TPR.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out / f"{args.model}.png", dpi=170, bbox_inches="tight")
    print(f"\nwrote {args.out / f'{args.model}.json'} and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
