"""Amplify the base model's per-token edit and measure TPR under all three monitor kinds.

    python experiments/amplify_pertoken.py

Reads  data/splits/, outputs/llama-3.1-8b/probes/base/, outputs/llama-3.1-8b/evals/base.npz,
       outputs/llama-3.1-8b/activations/base/tokens/
Writes experiments/outputs/amplify_pertoken/<model>.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.activations import (  # noqa: E402
    MODELS,
    load_model,
    model_dir,
    probes_dir,
    response_activations,
)
from core.concepts import ALL_CONCEPTS, TRAIN_CONCEPTS, held_out_concepts  # noqa: E402
from core.episodes import (  # noqa: E402
    ALL_CELLS,
    build_episode,
    content,
    eval_prefills,
    load_pools,
    with_imitate,
)
from core.probes import MonitorSet, _forward, seed_for  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = "llama-3.1-8b"
ALPHAS = [0, 4, 8, 16, 32]
KINDS = ["logreg", "mlp", "attention"]
EVAL_ROWS = (150, 250)


def head(bank, blk, mask):
    P = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in bank.params.items()}
    P["mu"] = torch.as_tensor(bank.mu, dtype=torch.float32)
    P["sd"] = torch.as_tensor(bank.sd, dtype=torch.float32)
    with torch.no_grad():
        return _forward(bank.kind, P, blk, mask).numpy().astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-prefills", type=int, default=100)
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ho = held_out_concepts()
    n = len(ho) * args.n_prefills * 2
    print(
        f"{len(ho)} concepts x {args.n_prefills} prefills x 2 arms = {n:,} episode forwards "
        f"(~{n / 8.6 / 60:.0f} min), then CPU scoring"
    )

    mdir, pdir = model_dir(OUT, MODEL), probes_dir(OUT, MODEL, source="base")
    A = mdir / "activations" / "base" / "tokens"
    pools = load_pools(ROOT / "data/splits", ALL_CONCEPTS)
    banks = {k: MonitorSet.load(pdir / f"{k}.npz") for k in KINDS}
    layer = banks["logreg"].layer
    S = {k: np.load(pdir / f"scores_{k}.npy") for k in KINDS}
    idxr = [
        json.loads(x)
        for x in (pdir / "index.jsonl").read_text().splitlines()
        if x.strip()
    ]
    conc = np.array([r["concept"] for r in idxr])
    z = np.load(mdir / "evals" / "base.npz", allow_pickle=True)
    cs = [str(x) for x in z["concepts"]]

    thr = {}
    for kind in KINDS:
        for c in ho:
            b = banks[kind]
            grp = TRAIN_CONCEPTS if c in TRAIN_CONCEPTS else held_out_concepts()
            ben = S[kind][
                np.isin(conc, [x for x in grp if x != c and x in cs]), b.names.index(c)
            ]
            thr[(kind, c)] = float(np.quantile(ben, 0.99))

    model, tok = load_model(MODELS[MODEL]["hf_name"])
    model.eval()
    t0 = time.time()
    per_token: dict = {}
    for c in ho:
        rows = eval_prefills(pools[c])[: args.n_prefills]
        pj = [x for x in ho if x != c]
        acc = []
        for t, row in enumerate(rows):
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
            )
            arms = {}
            for arm, im in (("up", ep.cell.c_label), ("down", 1 - ep.cell.c_label)):
                e = with_imitate(ep, im)
                arms[arm] = response_activations(
                    model, tok, layer, e.messages, e.prefill
                )
            acc.append((arms["up"] - arms["down"]).astype(np.float32))
        per_token[c] = acc
        print(f"  {c:26s} [{(time.time() - t0) / 60:.0f}m]", flush=True)
    del model
    torch.cuda.empty_cache()

    curve = {k: [] for k in KINDS}
    percc = {k: [] for k in KINDS}
    for a in ALPHAS:
        acc = {k: [] for k in KINDS}
        for c in ho:
            toks = np.load(A / f"{c}.npy", mmap_mode="r")
            off = np.load(A / f"{c}.idx.npy")
            raw = [
                torch.tensor(np.asarray(toks[off[r] : off[r + 1]]), dtype=torch.float32)
                for r in range(EVAL_ROWS[0], EVAL_ROWS[0] + args.n_prefills)
            ]
            dh_t = per_token[c]
            n = max(b.shape[0] for b in raw)
            pad = torch.zeros(len(raw), n, raw[0].shape[1])
            msk = torch.zeros(len(raw), n)
            for j, b in enumerate(raw):
                L = b.shape[0]
                d = torch.tensor(dh_t[j])
                e = torch.zeros(L, b.shape[1])
                m = min(L, d.shape[0])
                e[:m] = d[:m]
                if m < L:
                    e[m:] = d.mean(0)
                pad[j, :L] = b - a * e
                msk[j, :L] = 1
            for kind in KINDS:
                bk = banks[kind]
                k = bk.names.index(c)
                zz = (head(bk, pad, msk)[:, k] - bk.threshold[k]) / bk.scale[k]
                acc[kind].append(float((zz > thr[(kind, c)]).mean()))
        for kind in KINDS:
            curve[kind].append(float(np.mean(acc[kind])))
            percc[kind].append([float(x) for x in acc[kind]])
        print(
            f"  alpha {a:3d}  " + "  ".join(f"{k} {curve[k][-1]:.3f}" for k in KINDS),
            flush=True,
        )

    d = HERE / "outputs/amplify_pertoken"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{MODEL}.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "convention": "per_token",
                "alphas": ALPHAS,
                "concepts": ho,
                "curve": {k: {"tpr": curve[k]} for k in KINDS},
                "by_concept": percc,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{'alpha':>6s}" + "".join(f"{k:>12s}" for k in KINDS))
    for i, a in enumerate(ALPHAS):
        print(f"{a:6d}" + "".join(f"{curve[k][i]:12.3f}" for k in KINDS))
    print(f"\nwrote {d / f'{MODEL}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
