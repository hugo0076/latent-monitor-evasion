"""Alignment between the edits of the model organism and of the base model. CPU only.

    python experiments/edit_alignment.py

Reads  outputs/<model>/evals/{base,r1_sigmoid_mse}.npz
Writes experiments/outputs/edit_alignment/<model>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.activations import model_dir  # noqa: E402
from core.concepts import held_out_concepts  # noqa: E402
from core.geometry import unit  # noqa: E402


def edits(z, concept: str) -> np.ndarray:
    """Per-episode edits `up - down` of one concept. [T, D]"""
    i = [str(c) for c in z["concepts"]].index(concept)
    return (z["h_up"][i] - z["h_down"][i]).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--adapter", default="r1_sigmoid_mse")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "experiments/outputs/edit_alignment"
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    evals = model_dir(args.outputs, args.model) / "evals"
    base = np.load(evals / "base.npz", allow_pickle=True)
    org = np.load(evals / f"{args.adapter}.npz", allow_pickle=True)
    if not np.array_equal(base["prefill_ids"], org["prefill_ids"]):
        raise SystemExit(
            "the two evaluations use different prefills, so episodes do not match"
        )

    concepts = held_out_concepts()
    per_concept = {}
    for c in concepts:
        cos = (unit(edits(org, c)) * unit(edits(base, c))).sum(-1)
        per_concept[c] = float(cos.mean())

    v = np.array(list(per_concept.values()))
    rng = np.random.default_rng(0)
    boot = [rng.choice(v, len(v)).mean() for _ in range(args.n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {
        "model": args.model,
        "adapter": args.adapter,
        "cos_dh_org_dh_base": float(v.mean()),
        "ci95": [float(lo), float(hi)],
        "per_concept": per_concept,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"cos(Δh_{args.adapter}, Δh_base) = {v.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
