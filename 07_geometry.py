"""Geometry of the activation edit Δh, from one evaluation artifact. CPU only.

    python 07_geometry.py --run r1_sigmoid_mse
    python 07_geometry.py --run r1_sigmoid_mse --compare base

Reads  outputs/<model>/evals/<run>.npz
Writes outputs/<model>/geometry/<run>.json

Δh is `up - down` on a matched pair (see 05_eval.py). The report gives:

1. The edit size `||Δh||` and its aim `cos(Δh, w)` per concept (Sections 4.1 and 4.3).
2. The similarity of the edits of different concepts, with and without PC1.
3. The fraction of each edit that lies along PC1 of the probe set.
4. The content gate: the edit when the prefill is the sentence of another concept.
5. Per grid cell, `delta_z_contrast = ||Δh|| * cos(Δh, w) / scale`, which separates the
   magnitude of the edit from its aim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.activations import MODELS, model_dir
from core.concepts import TRAIN_CONCEPTS
from core.geometry import cosine_matrix, delta_h, unit

HERE = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    man = json.loads(str(z["manifest"]))
    concepts = [str(x) for x in z["concepts"]]
    roster = [str(x) for x in z["probes"]][: len(z["probe_logreg_scale"])]
    return {
        "z": z,
        "man": man,
        "concepts": concepts,
        "roster": roster,
        "grid": [str(x) for x in z["grid"]],
    }


def push(z, arm_up: str, arm_down: str) -> np.ndarray:
    """Mean edit per concept, `arm_up - arm_down`."""
    return delta_h(elevate=z[arm_up].mean(1), suppress=z[arm_down].mean(1))


def report(e: dict) -> dict:
    z, concepts, roster = e["z"], e["concepts"], e["roster"]
    W = z["probe_logreg_w"]
    wi = np.stack([W[roster.index(c)] for c in concepts])
    pc1 = z["pc_components"][0].astype(np.float64)

    dh = push(z, "h_up", "h_down")
    nrm = np.linalg.norm(dh, axis=1)
    # per-episode cosine, as in the paper. `cos_dh_probe_of_mean` is the cosine of the mean edit.
    per_ep = z["h_up"] - z["h_down"]
    cosw = (unit(per_ep) @ wi[:, :, None]).squeeze(-1).mean(1)
    cosw_of_mean = (dh * wi).sum(1) / nrm
    pc1_frac = (dh @ pc1) ** 2 / nrm**2
    perp = dh - np.outer(dh @ pc1, pc1)
    off = ~np.eye(len(concepts), dtype=bool)
    raw_cos, perp_cos = cosine_matrix(dh)[off], cosine_matrix(perp)[off]

    per = z["h_up"] - z["h_down"]
    consistency = np.linalg.norm(per.mean(1), axis=1) / np.linalg.norm(
        per, axis=2
    ).mean(1)
    rng = np.random.default_rng(0)
    T = per.shape[1]
    half = []
    for i in range(len(concepts)):
        p = rng.permutation(T)
        a, b = per[i, p[: T // 2]].mean(0), per[i, p[T // 2 :]].mean(0)
        half.append(float(unit(a) @ unit(b)))

    out = {
        "run": e["man"].get("adapter"),
        "n_concepts": len(concepts),
        "norm": float(nrm.mean()),
        "cos_dh_probe": float(cosw.mean()),
        "cos_dh_probe_of_mean": float(cosw_of_mean.mean()),
        "pc1_fraction": float(pc1_frac.mean()),
        "cross_concept_cos_raw": float(raw_cos.mean()),
        "cross_concept_cos_perp_pc1": float(perp_cos.mean()),
        "consistency_norm_ratio": float(consistency.mean()),
        "split_half_cos": float(np.mean(half)),
        "by_concept": {
            c: {
                "norm": float(nrm[i]),
                "cos_probe": float(cosw[i]),
                "cos_probe_of_mean": float(cosw_of_mean[i]),
                "pc1_fraction": float(pc1_frac[i]),
                "split": "train" if c in TRAIN_CONCEPTS else "held_out",
            }
            for i, c in enumerate(concepts)
        },
    }

    if "h_up_off" in z.files:
        dh_off = push(z, "h_up_off", "h_down_off")
        n_off = np.linalg.norm(dh_off, axis=1)
        out["content_gate"] = {
            "norm_on_topic": float(nrm.mean()),
            "norm_off_topic": float(n_off.mean()),
            "norm_ratio": float(np.mean(nrm / n_off)),
            "cos_on_off": float(np.mean((dh * dh_off).sum(1) / (nrm * n_off))),
            "cross_concept_cos_off_topic": float(cosine_matrix(dh_off)[off].mean()),
        }

    grid = e["grid"]
    if grid and "h_grid_up" in z.files:
        G = (z["h_grid_up"] - z["h_grid_down"]).mean(2)
        M = np.linalg.norm(G, axis=-1)
        wg = np.stack([W[roster.index(c)] for c in grid])
        cg = np.einsum("ijd,id->ij", G / M[..., None], wg)
        eye = np.eye(len(grid), dtype=bool)
        out["per_cell"] = {
            "concepts": grid,
            "norm_diag": float(M[eye].mean()),
            "norm_off": float(M[~eye].mean()),
            "norm_ratio": float(M[eye].mean() / M[~eye].mean()),
            "cos_probe_diag": float(cg[eye].mean()),
            "cos_probe_off": float(cg[~eye].mean()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--outputs", type=Path, default=HERE / "outputs")
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--compare", nargs="*", default=[], help="other runs to tabulate alongside"
    )
    args = ap.parse_args()

    mdir = model_dir(args.outputs, args.model, args.layer)
    rows = {}
    for run in [args.run, *args.compare]:
        p = mdir / "evals" / f"{run}.npz"
        if not p.exists():
            print(f"skip {run}: no {p.name}")
            continue
        rows[run] = report(load(p))

    print(
        f"\n{'run':22s} {'||dh||':>8s} {'cos(dh,w)':>10s} {'PC1 frac':>9s} "
        f"{'cos raw':>8s} {'cos perpPC1':>12s} {'consist':>8s}"
    )
    print("-" * 82)
    for run, r in rows.items():
        print(
            f"{run:22s} {r['norm']:8.2f} {r['cos_dh_probe']:10.3f} {r['pc1_fraction']:9.3f} "
            f"{r['cross_concept_cos_raw']:8.3f} {r['cross_concept_cos_perp_pc1']:12.3f} "
            f"{r['consistency_norm_ratio']:8.3f}"
        )
    for run, r in rows.items():
        if "content_gate" in r:
            g = r["content_gate"]
            print(
                f"\n{run}  content gate: ||dh|| on-topic {g['norm_on_topic']:.2f} "
                f"off-topic {g['norm_off_topic']:.2f} ratio {g['norm_ratio']:.2f}x  "
                f"cos(on,off) {g['cos_on_off']:+.3f}"
            )
        if "per_cell" in r:
            c = r["per_cell"]
            print(
                f"{run}  per cell:     ||dh|| diag {c['norm_diag']:.2f} off {c['norm_off']:.2f} "
                f"ratio {c['norm_ratio']:.2f}x   cos(dh,w_i) diag {c['cos_probe_diag']:+.3f} "
                f"off {c['cos_probe_off']:+.3f}"
            )

    gdir = mdir / "geometry"
    gdir.mkdir(parents=True, exist_ok=True)
    for run, r in rows.items():
        (gdir / f"{run}.json").write_text(json.dumps(r, indent=2) + "\n")
    print(f"\nwrote {gdir}/{{{','.join(rows)}}}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
