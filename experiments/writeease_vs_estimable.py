"""Per-concept aim of the base model's edit, with its estimability and write-ease. CPU only.

    python experiments/writeease_vs_estimable.py

Reads  outputs/<model>/evals/base.npz, outputs/<model>/probes/base/logreg.npz and
       outputs/<model>/activations/base/pooled/, for each of the four models
Writes experiments/outputs/writeease_vs_estimable/all_models.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.activations import model_dir, probes_dir  # noqa: E402
from core.concepts import ALL_CONCEPTS, same_family  # noqa: E402
from core.probes import MonitorSet  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ALPHAS = (0.25, 0.55)
SHRINK = 0.05


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """corr(x, y) with z projected out of both."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    res = {}
    for model in ("llama-3.1-8b", "gemma-3-27b", "qwen3-14b", "llama-3.3-70b"):
        mdir, pdir = model_dir(OUT, model), probes_dir(OUT, model, source="base")
        ev = mdir / "evals" / "base.npz"
        if not ev.exists():
            continue
        bank = MonitorSet.load(pdir / "logreg.npz")
        W = bank.params["w"].astype(np.float64)
        adir = mdir / "activations" / "base" / "pooled"
        have = [
            c for c in ALL_CONCEPTS if (adir / f"{c}.npy").exists() and c in bank.names
        ]
        POOL = {c: np.load(adir / f"{c}.npy")[150:].astype(np.float64) for c in have}

        ALLH = np.concatenate([POOL[c] for c in have])
        X = ALLH - ALLH.mean(0)
        Sig = X.T @ X / len(X)
        Sig = (1 - SHRINK) * Sig + SHRINK * np.trace(Sig) / len(Sig) * np.eye(len(Sig))
        evals, evecs = np.linalg.eigh(Sig)
        gmean = ALLH.mean(0)

        z = np.load(ev, allow_pickle=True)
        cs = [str(x) for x in z["concepts"]]
        hu, hd = np.asarray(z["h_up"]), np.asarray(z["h_down"])

        rows = []
        for c in have:
            if c not in cs:
                continue
            k = bank.names.index(c)
            w = unit(W[k])
            dh = (hu[cs.index(c)] - hd[cs.index(c)]).astype(np.float64)
            aim = float(np.mean([unit(v) @ w for v in dh]))
            peers = [p for p in have if p != c and not same_family(c, p)]
            dom = POOL[c].mean(0) - np.concatenate([POOL[p] for p in peers]).mean(0)
            row = {
                "concept": c,
                "aim": aim,
                "estimable": float(unit(dom) @ w),
                "var_along_w": float(np.var((ALLH - gmean) @ w)),
            }
            cw = evecs.T @ w
            for a in ALPHAS:
                row[f"we_a{a}"] = float(np.sqrt(max((cw**2 * evals**a).sum(), 0.0)))
            rows.append(row)

        A = np.array([r["aim"] for r in rows])
        E = np.array([r["estimable"] for r in rows])
        out = {
            "n_concepts": len(rows),
            "by_concept": rows,
            "corr_aim_estimable": float(np.corrcoef(A, E)[0, 1]),
        }
        print(f"\n== {model}   n={len(rows)}")
        print(
            f"   corr(aim, estimable)                       {out['corr_aim_estimable']:+.3f}"
        )
        for a in ALPHAS:
            Wv = np.array([r[f"we_a{a}"] for r in rows])
            out[f"corr_aim_we_a{a}"] = float(np.corrcoef(A, Wv)[0, 1])
            out[f"partial_aim_we_given_est_a{a}"] = partial(A, Wv, E)
            out[f"partial_aim_est_given_we_a{a}"] = partial(A, E, Wv)
            out[f"corr_est_we_a{a}"] = float(np.corrcoef(E, Wv)[0, 1])
            print(
                f"   alpha={a}: corr(aim, write-ease) {out[f'corr_aim_we_a{a}']:+.3f}   "
                f"corr(est, we) {out[f'corr_est_we_a{a}']:+.3f}"
            )
            print(
                f"             partial(aim, we | est) {out[f'partial_aim_we_given_est_a{a}']:+.3f}"
                f"   partial(aim, est | we) "
                f"{out[f'partial_aim_est_given_we_a{a}']:+.3f}"
            )
        res[model] = out

    d = HERE / "outputs/writeease_vs_estimable"
    d.mkdir(parents=True, exist_ok=True)
    (d / "all_models.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {d / 'all_models.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
