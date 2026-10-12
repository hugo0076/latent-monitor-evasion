"""Score the held-out eval arms with a Mahalanobis OOD detector fitted on base-model activations.
CPU only.

    python experiments/ood_mahalanobis.py --model llama-3.1-8b --runs base r1_sigmoid_mse r1_sigmoid_oodsof0.25
    python experiments/ood_mahalanobis.py --model gemma-3-27b --runs base r1_sigmoid_mse

Reads  outputs/<model>/probes/<probe-source>/index.jsonl, outputs/<model>/evals/<run>.npz,
       outputs/<model>/activations/<probe-source>/pooled/
Writes experiments/outputs/ood_mahalanobis/<model>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.activations import MODELS, model_dir, probes_dir  # noqa: E402
from core.concepts import TEST_CONCEPTS  # noqa: E402

HERE = Path(__file__).resolve().parent


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(
        roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    )


class Mahalanobis:
    """Shrunk-covariance Mahalanobis, global and class-conditional."""

    def __init__(self, X: np.ndarray, labels: np.ndarray) -> None:
        self.mu = X.mean(0)
        lw = LedoitWolf(assume_centered=False).fit(X)
        self.shrinkage = float(lw.shrinkage_)
        ev, V = np.linalg.eigh(lw.covariance_)
        self.Wh = V / np.sqrt(np.maximum(ev, 1e-12))
        self.names = sorted(set(labels.tolist()))
        self.mus = np.stack([X[labels == c].mean(0) for c in self.names])

    def d2_global(self, H: np.ndarray) -> np.ndarray:
        return (((H - self.mu) @ self.Wh) ** 2).sum(-1)

    def d2_classwise(self, H: np.ndarray) -> np.ndarray:
        Z = H @ self.Wh
        M = self.mus @ self.Wh
        d2 = (Z**2).sum(1)[:, None] - 2 * Z @ M.T + (M**2).sum(1)[None, :]
        return d2.min(1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--runs", nargs="+", default=["base", "r1_sigmoid_mse"])
    ap.add_argument("--probe-source", default="base")
    ap.add_argument("--scope", choices=["heldout", "all"], default="heldout")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/ood_mahalanobis")
    args = ap.parse_args()

    mdir = model_dir(args.outputs, args.model)
    adir = probes_dir(args.outputs, args.model, source=args.probe_source)
    idx = [
        json.loads(x)
        for x in (adir / "index.jsonl").read_text().splitlines()
        if x.strip()
    ]
    take: dict[str, list[int]] = {}
    for r in idx:
        take.setdefault(r["concept"], []).append(r["nf_row"])
    order = sorted(take)
    pooled = mdir / "activations" / args.probe_source / "pooled"
    POOL = np.concatenate(
        [np.load(pooled / f"{c}.npy")[take[c]] for c in order]
    ).astype(np.float64)
    conc = np.array([c for c in order for _ in take[c]])

    nf = np.array([v for c in order for v in take[c]])
    fit_mask = nf >= 250
    print(
        f"{args.model}: benign {POOL.shape}, fit {int(fit_mask.sum())} (content pool), "
        f"calibrate {int((~fit_mask).sum())} (eval prefills)"
    )

    det = Mahalanobis(POOL[fit_mask], conc[fit_mask])
    print(f"Ledoit-Wolf shrinkage {det.shrinkage:.4f}")
    cal = {
        "global": det.d2_global(POOL[~fit_mask]),
        "classwise": det.d2_classwise(POOL[~fit_mask]),
    }
    held = {
        "global": det.d2_global(POOL[fit_mask]),
        "classwise": det.d2_classwise(POOL[fit_mask]),
    }
    for k in cal:
        print(
            f"  {k:9s} d^2  fit rows median {np.median(held[k]):10.1f}   "
            f"calibration rows median {np.median(cal[k]):10.1f}"
        )

    ARMS = ("h_bare", "h_up", "h_down")
    scored: dict[str, dict] = {}
    probe_z: dict[str, np.ndarray] = {}
    for run in args.runs:
        path = mdir / "evals" / f"{run}.npz"
        if not path.exists():
            print(f"  (no eval artifact for {run}, skipping)")
            continue
        z = np.load(path, allow_pickle=True)
        keep = [
            i
            for i, c in enumerate(str(x) for x in z["concepts"])
            if args.scope == "all" or c in TEST_CONCEPTS
        ]
        cs = [str(z["concepts"][i]) for i in keep]
        kcol = [
            next(
                i
                for i, (p, k) in enumerate(
                    zip([str(x) for x in z["probes"]], [str(x) for x in z["kinds"]])
                )
                if p == c and k == "logreg"
            )
            for c in cs
        ]
        scored[run] = {}
        bare = np.asarray(z["h_bare"])[keep].astype(np.float64)
        for arm in ARMS:
            H = np.asarray(z[arm])[keep].astype(np.float64) if arm != "h_bare" else bare
            flat = H.reshape(-1, H.shape[-1])
            scored[run][arm] = {
                "global": det.d2_global(flat),
                "classwise": det.d2_classwise(flat),
                "shift": np.linalg.norm((H - bare).reshape(-1, H.shape[-1]), axis=1),
            }
        probe_z[run] = np.stack(
            [np.asarray(z["down"])[k, :, kcol[i]] for i, k in enumerate(keep)]
        ).ravel()
        del z

    def rate(scores: np.ndarray, ben: np.ndarray) -> tuple[float, float]:
        return auroc(scores, ben), float(
            (scores > float(np.quantile(ben, 0.99))).mean()
        )

    benigns = {"naive": cal}
    if "base" in scored:
        benigns["format-aware"] = {
            k: np.r_[scored["base"]["h_up"][k], scored["base"]["h_down"][k]]
            for k in ("global", "classwise")
        }

    out: dict = {
        "model": args.model,
        "scope": args.scope,
        "shrinkage": det.shrinkage,
        "n_fit": int(fit_mask.sum()),
        "n_cal": int((~fit_mask).sum()),
        "arms": {},
    }
    for kind in ("global", "classwise"):
        for bname, ben in benigns.items():
            print(
                f"\n=== {kind} Mahalanobis, benign = {bname} "
                f"(n={len(ben[kind])}, median d^2 {np.median(ben[kind]):.1f}) ==="
            )
            print(
                f"{'run':16s} {'arm':7s} {'median d^2':>11s} {'x benign':>9s} "
                f"{'AUROC':>7s} {'flag@1%':>8s}"
            )
            for run in scored:
                for arm in ARMS:
                    s = scored[run][arm][kind]
                    au, fl = rate(s, ben[kind])
                    print(
                        f"{run:16s} {arm[2:]:7s} {np.median(s):11.1f} "
                        f"{np.median(s) / np.median(ben[kind]):9.2f} {au:7.3f} {fl:8.3f}"
                    )
                    out["arms"].setdefault(f"{kind}/{bname}", {})[
                        f"{run}/{arm[2:]}"
                    ] = {"median_d2": float(np.median(s)), "auroc": au, "flag_rate": fl}

    print("\n=== what the detector tracks (FT down arm) ===")
    for run in scored:
        s = scored[run]["h_down"]["classwise"]
        sh = scored[run]["h_down"]["shift"]
        r_shift = float(np.corrcoef(s, sh)[0, 1])
        r_probe = float(np.corrcoef(s, probe_z[run])[0, 1])
        print(
            f"  {run:16s} corr(d^2, ||h - h_bare||) {r_shift:+.3f}   "
            f"corr(d^2, probe logit)  {r_probe:+.3f}"
        )
        out.setdefault("correlations", {})[run] = {
            "shift": r_shift,
            "probe_logit": r_probe,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out / f'{args.model}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
