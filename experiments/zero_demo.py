"""Measure the edit with no demonstrations: only the final instruction and the forced prefill.

    python experiments/zero_demo.py

Reads  data/splits/, outputs/<model>/probes/<probe-source>/, outputs/<model>/adapters/<run>/
Writes experiments/outputs/zero_demo/<model><suffix>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
from core.concepts import TRAIN_CONCEPTS, held_out_concepts, same_family  # noqa: E402
from core.episodes import INSTRUCTION, eval_prefills, load_pools  # noqa: E402
from core.probes import MonitorSet  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--runs", nargs="+", default=["base", "r1_sigmoid_mse"])
    ap.add_argument("--probe-source", default="base")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/zero_demo")
    ap.add_argument("--n-prefills", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    pdir = probes_dir(args.outputs, args.model, source=args.probe_source)
    bank = MonitorSet.load(pdir / "logreg.npz")
    W = bank.params["w"].astype(np.float64)
    names = bank.names
    S = np.load(pdir / "scores_logreg.npy")
    conc = np.array(
        [
            json.loads(x)["concept"]
            for x in (pdir / "index.jsonl").read_text().splitlines()
            if x.strip()
        ]
    )
    ho = held_out_concepts()
    pools = load_pools(args.splits, ho)

    def peers(c: str) -> list[str]:
        grp = TRAIN_CONCEPTS if c in TRAIN_CONCEPTS else held_out_concepts()
        return [x for x in grp if x != c and not same_family(c, x)]

    thr = {
        c: float(np.quantile(S[np.isin(conc, peers(c)), names.index(c)], 0.99))
        for c in ho
    }
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)  # noqa: E731

    def turn(d: int) -> list[dict]:
        return [{"role": "user", "content": INSTRUCTION.format(imitate=d)}]

    print(
        f"{len(ho)} held-out concepts x {args.n_prefills} prefills x 3 conditions "
        f"x {len(args.runs)} runs = {len(ho) * args.n_prefills * 3 * len(args.runs):,} forwards"
    )
    out = {
        "model": args.model,
        "n_prefills": args.n_prefills,
        "concepts": ho,
        "chance_cos": float(1 / np.sqrt(W.shape[1])),
        "convention": "digit contrast = h(imitate=1) - h(imitate=0), a FIXED contrast. "
        "c_label contrast alternates with t%2 and cancels a digit polarity.",
        "by_run": {},
    }
    for run in args.runs:
        mdl, tok = load_model(MODELS[args.model]["hf_name"])
        if run != "base":
            from peft import PeftModel

            mdl = PeftModel.from_pretrained(
                mdl, model_dir(args.outputs, args.model) / "adapters" / run
            )
            mdl = mdl.merge_and_unload()
        mdl.eval()

        def hz(messages, prefill, k, _m=mdl, _t=tok):
            h = response_activations(
                _m, _t, bank.layer, messages, prefill, max_length=args.max_length
            )
            return h.mean(0).astype(np.float64), float(
                bank.logits(h[None], np.ones((1, len(h)), np.float32))[0][k]
            )

        per = {}
        for c in ho:
            k = names.index(c)
            w = unit(W[k])
            rows = eval_prefills(pools[c])[: args.n_prefills]
            d0, d1, zb, z0, z1 = [], [], [], [], []
            for t, r in enumerate(rows):
                h0, s0 = hz(turn(0), r["sentence"], k)
                h1, s1 = hz(turn(1), r["sentence"], k)
                _, sb = hz(PROMPT, r["sentence"], k)
                d0.append(h0)
                d1.append(h1)
                z0.append(s0)
                z1.append(s1)
                zb.append(sb)
            D = np.array(d1) - np.array(d0)
            clab = np.array([1 if t % 2 else -1 for t in range(len(rows))])[:, None]
            Dc = D * clab
            nz = np.linalg.norm(D, axis=1) + 1e-12
            z0, z1, zb = map(np.array, (z0, z1, zb))
            per[c] = {
                "cos_digit_per_episode": float(np.mean((D / nz[:, None]) @ w)),
                "cos_digit_of_mean": float(unit(D.mean(0)) @ w),
                "cos_clabel_per_episode": float(
                    np.mean((Dc / (np.linalg.norm(Dc, axis=1) + 1e-12)[:, None]) @ w)
                ),
                "cos_clabel_of_mean": float(unit(Dc.mean(0)) @ w),
                "norm_digit": float(nz.mean()),
                "norm_of_mean_digit": float(np.linalg.norm(D.mean(0))),
                "delta_z_digit": float((z1 - z0).mean()),
                "z_bare": float(zb.mean()),
                "z_d0": float(z0.mean()),
                "z_d1": float(z1.mean()),
                "tpr_bare": float((zb > thr[c]).mean()),
                "tpr_d0": float((z0 > thr[c]).mean()),
                "tpr_d1": float((z1 > thr[c]).mean()),
                "threshold": thr[c],
            }
            print(
                f"  {run} {c:26s} cos_digit {per[c]['cos_digit_per_episode']:+.4f}  "
                f"|dh| {per[c]['norm_digit']:6.2f}  dz {per[c]['delta_z_digit']:+.3f}  "
                f"TPR bare/d0/d1 {per[c]['tpr_bare']:.3f}/{per[c]['tpr_d0']:.3f}/"
                f"{per[c]['tpr_d1']:.3f}",
                flush=True,
            )
        out["by_run"][run] = per
        del mdl

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}{args.suffix}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(f"\nwrote {args.out / f'{args.model}{args.suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
