"""Interpolate the instruction block toward another episode's and measure where the edit points.

    python experiments/patch_interp.py --adapter r1_sigmoid_mse --n-prefills 30 \\
        --alphas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1

Reads  data/splits/, outputs/<model>/probes/<probe-source>/, outputs/<model>/adapters/<adapter>/
Writes experiments/outputs/patch_interp/<model>_<adapter>.json
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.activations import (  # noqa: E402
    MODELS,
    find_decoder_layers,
    load_model,
    model_dir,
    probes_dir,
    render,
    response_activations,
)
from core.concepts import ALL_CONCEPTS, held_out_concepts, same_family  # noqa: E402
from core.episodes import (  # noqa: E402
    ALL_CELLS,
    build_episode,
    content,
    eval_prefills,
    load_pools,
    with_imitate,
)
from core.probes import MonitorSet, seed_for  # noqa: E402


def spans(tok, messages: list, prefill: str) -> dict[str, tuple[int, int]]:
    """Token spans of the four episode segments: demonstrations, instruction, generation
    header, prefill. Boundaries are found by re-rendering successive prefixes of the chat
    template and measuring their length, so they stay correct for any template."""

    def n(t) -> int:
        return len(tok(t, add_special_tokens=False).input_ids)

    d_end = n(
        tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=False
        )
    )
    i_end = n(
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    )
    prompt, full = render(tok, messages, prefill)
    return {
        "D": (0, d_end),
        "I": (d_end, i_end),
        "G": (i_end, n(prompt)),
        "P": (n(prompt), n(full)),
    }


class Patch:
    """Overwrite the residual stream at chosen positions in a set of layers.

    Each layer receives the donor activations of the same layer.
    """

    def __init__(self, model, layers: list[int]) -> None:
        self.layers = [(i, find_decoder_layers(model)[i]) for i in layers]
        self.plan: tuple | None = None
        self.handles: list = []

    def __enter__(self):
        def mk(idx):
            def hook(_m, _in, out):
                if self.plan is None:
                    return out
                a, b, vecs = self.plan
                if idx not in vecs:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                h[:, a:b, :] = vecs[idx].to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h

            return hook

        self.handles = [layer.register_forward_hook(mk(i)) for i, layer in self.layers]
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


HERE = Path(__file__).resolve().parent


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(
        roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    )


def frozen_benign(
    probes_root: Path, kind: str, roster: list[str], concepts: list[str]
) -> dict[str, np.ndarray]:
    """Stage-03 calibrated scores of each concept's same-split, non-same-family peers.

    Returns an empty dict if the score matrix is missing. The caller then reports cosines only.
    """
    S, idx = probes_root / f"scores_{kind}.npy", probes_root / "index.jsonl"
    if not (S.exists() and idx.exists()):
        return {}
    scores = np.load(S)
    rows = [json.loads(x) for x in idx.read_text().splitlines() if x.strip()]
    conc = np.array([r["concept"] for r in rows])
    return {
        c: scores[
            np.isin(conc, [x for x in concepts if x != c and not same_family(c, x)]),
            roster.index(c),
        ]
        for c in concepts
    }


def detection(
    rows: list[dict], concept_key: str, z_key: str, ben: dict[str, np.ndarray]
) -> tuple[float, float]:
    """Macro-averaged AUROC and rate above the 1%-FPR threshold, thresholded per concept."""
    if not ben:
        return float("nan"), float("nan")
    per: dict[str, list[float]] = {}
    for r in rows:
        per.setdefault(r[concept_key], []).append(r[z_key])
    au, rate = [], []
    for c, zs in per.items():
        b = ben[c]
        au.append(auroc(np.asarray(zs), b))
        rate.append(float((np.asarray(zs) > float(np.quantile(b, 0.99))).mean()))
    return float(np.mean(au)), float(np.mean(rate))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--adapter", default="r1_sigmoid_mse")
    ap.add_argument("--probe-source", default="base")
    ap.add_argument("--kind", default="logreg")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/patch_interp")
    ap.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    ap.add_argument(
        "--donors", nargs="+", default=["cross", "mean"], choices=["cross", "mean"]
    )
    ap.add_argument("--n-prefills", type=int, default=10)
    ap.add_argument("--n-donor-draws", type=int, default=12)
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument("--c-label", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ho = held_out_concepts()
    pools = load_pools(args.splits, ALL_CONCEPTS)
    read_layer = MODELS[args.model]["layer"]
    layers = list(range(read_layer))

    probes_root = probes_dir(args.outputs, args.model, source=args.probe_source)
    monitors = MonitorSet.load(probes_root / f"{args.kind}.npz")
    W = monitors.params["w"].astype(np.float64)
    ben = frozen_benign(probes_root, args.kind, monitors.names, ho)
    if not ben:
        print("no stage-03 score matrix; reporting cosines only", flush=True)

    model, tok = load_model(MODELS[args.model]["hf_name"])
    if args.adapter and args.adapter != "base":
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, model_dir(args.outputs, args.model) / "adapters" / args.adapter
        )
        model = model.merge_and_unload()
    model.eval()
    dev = next(model.parameters()).device
    decoder = find_decoder_layers(model)

    def episode(c: str, t: int):
        pj = [x for x in ho if x != c and not same_family(c, x)]
        row = eval_prefills(pools[c])[t % 100]
        rng = random.Random(seed_for("patch", args.seed, c, t))
        cell = next(
            x for x in ALL_CELLS if x.c_label == args.c_label and x.prefill_dir == 1
        )
        return build_episode(
            c,
            cell,
            topic_demos=content(pools[c]),
            other_demos={m: content(pools[m]) for m in pj},
            topic_prefills=[row],
            other_prefills={m: eval_prefills(pools[m]) for m in pj},
            n_demo_pairs=args.n_demo_pairs,
            rng=rng,
            system_prompt=None,
        ), row

    def block(messages) -> dict[int, torch.Tensor]:
        """I+G residual stream at every layer, from one prompt-only forward."""
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
        caught: dict[int, torch.Tensor] = {}

        def mk(i):
            def hook(_m, _in, out):
                caught[i] = (out[0] if isinstance(out, tuple) else out).detach()

            return hook

        hs = [decoder[i].register_forward_hook(mk(i)) for i in layers]
        try:
            with torch.inference_mode():
                model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    use_cache=False,
                )
        finally:
            for h in hs:
                h.remove()
        sp = spans(tok, messages, "")
        lo, hi = sp["I"][0], sp["G"][1]
        return {i: caught[i][0, lo:hi].clone() for i in layers}

    def arms(ep) -> dict[str, dict[int, torch.Tensor]]:
        return {
            arm: block(with_imitate(ep, im).messages)
            for arm, im in (("up", ep.cell.c_label), ("down", 1 - ep.cell.c_label))
        }

    def push(ep, sentence) -> np.ndarray:
        hu = response_activations(
            model, tok, read_layer, with_imitate(ep, ep.cell.c_label).messages, sentence
        )
        hd = response_activations(
            model,
            tok,
            read_layer,
            with_imitate(ep, 1 - ep.cell.c_label).messages,
            sentence,
        )
        return (hu.mean(0) - hd.mean(0)).astype(np.float64)

    t0 = time.time()
    bank: dict[tuple[str, int], dict] = {}
    for c in ho:
        for t in range(args.n_donor_draws):
            ep, _ = episode(c, t)
            bank[(c, t)] = arms(ep)
    widths = {v[a][layers[0]].shape[0] for v in bank.values() for a in ("up", "down")}
    if len(widths) != 1:
        raise SystemExit(f"donor blocks disagree in width: {sorted(widths)}")
    width = widths.pop()

    def mean_donor_for(host: str) -> dict:
        keep = [b for (c, _), b in bank.items() if c != host]
        return {
            arm: {i: torch.stack([b[arm][i] for b in keep]).mean(0) for i in layers}
            for arm in ("up", "down")
        }

    print(
        f"{len(bank)} donors cached, block {width} tokens ({(time.time() - t0) / 60:.1f} min)",
        flush=True,
    )

    mean_donors = {A: mean_donor_for(A) for A in ho}
    print(
        f"mean donor: leave-one-out over {len(bank) - args.n_donor_draws} blocks per host"
    )

    rows: dict[str, list[dict]] = {}
    pure_ref: list[float] = []
    skipped = 0
    with Patch(model, layers) as patcher:
        for A in ho:
            peers = [x for x in ho if x != A and not same_family(A, x)]
            kA = monitors.index[A]
            for t in range(args.n_prefills):
                ep, row = episode(A, t)
                sp = spans(tok, ep.messages, row["sentence"])
                lo, hi = sp["I"][0], sp["G"][1]
                if hi - lo != width:
                    skipped += 1
                    continue
                B = peers[t % len(peers)]
                kB = monitors.index[B]
                own = arms(ep)
                donors = {
                    "cross": bank[(B, t % args.n_donor_draws)],
                    "mean": mean_donors[A],
                }

                dh_A = push(ep, row["sentence"])
                epB, rowB = episode(B, t)
                dh_B = push(epB, rowB["sentence"])
                pure_ref.append(float(unit(dh_B) @ unit(W[kB])))

                for dname in args.donors:
                    for a in args.alphas:
                        zs = {}
                        for arm in ("up", "down"):
                            patcher.plan = (
                                lo,
                                hi,
                                {
                                    i: (1 - a) * own[arm][i] + a * donors[dname][arm][i]
                                    for i in layers
                                },
                            )
                            h = response_activations(
                                model,
                                tok,
                                read_layer,
                                with_imitate(
                                    ep,
                                    ep.cell.c_label
                                    if arm == "up"
                                    else 1 - ep.cell.c_label,
                                ).messages,
                                row["sentence"],
                            )
                            zs[arm] = (
                                h,
                                monitors.logits(
                                    h[None], np.ones((1, len(h)), np.float32)
                                )[0],
                            )
                        patcher.plan = None
                        dh = (zs["up"][0].mean(0) - zs["down"][0].mean(0)).astype(
                            np.float64
                        )
                        rows.setdefault(f"{dname}@{a}", []).append(
                            {
                                "A": A,
                                "B": B,
                                "t": t,
                                "norm": float(np.linalg.norm(dh)),
                                "cos_wA": float(unit(dh) @ unit(W[kA])),
                                "cos_wB": float(unit(dh) @ unit(W[kB])),
                                "cos_pushA": float(unit(dh) @ unit(dh_A)),
                                "cos_pushB": float(unit(dh) @ unit(dh_B)),
                                "z_down_A": float(zs["down"][1][kA]),
                                "z_down_B": float(zs["down"][1][kB]),
                            }
                        )
            print(f"  {A}  [{(time.time() - t0) / 60:.0f}m]", flush=True)
    if skipped:
        print(f"skipped {skipped} recipients whose I+G block was not {width} tokens")

    a0 = args.alphas[0]
    checks: dict[str, float] = {}
    if a0 == 0.0:
        if len(args.donors) > 1:
            d = max(
                abs(x["cos_wA"] - y["cos_wA"])
                for x, y in zip(
                    rows[f"{args.donors[0]}@0.0"], rows[f"{args.donors[1]}@0.0"]
                )
            )
            checks["alpha0_arms_agree"] = d
            print(f"\ncheck  alpha=0 identical across donor arms (want 0): {d:.2e}")
        c0 = float(np.mean([r["cos_pushA"] for r in rows[f"{args.donors[0]}@0.0"]]))
        checks["alpha0_is_identity"] = c0
        print(f"check  alpha=0 cos to identity push (want 1.000): {c0:.6f}")

    idn = rows[f"{args.donors[0]}@{a0}"]
    ref_A = float(np.mean([r["cos_wA"] for r in idn]))
    floor_B = float(np.mean([r["cos_wB"] for r in idn]))
    ref_B = float(np.mean(pure_ref))
    floor_pushB = float(np.mean([r["cos_pushB"] for r in idn]))
    print(
        f"\nreferences  identity cos(dh,w_A) {ref_A:+.3f}   floor cos(dh,w_B) {floor_B:+.3f}   "
        f"pure-B cos(dh_B,w_B) {ref_B:+.3f}   floor cos(dh,dh_B) {floor_pushB:+.3f}"
    )

    print(
        f"\n{'condition':14s} {'||dh||':>7s} {'A ret':>7s} {'B acq':>7s} | "
        f"{'cosA':>6s} {'cosB':>6s} | {'AUROC_A':>8s} {'TPR_A':>6s} {'flagB':>6s} "
        f"{'z_A':>7s} {'z_B':>7s}"
    )
    summary: dict[str, dict] = {}
    for dname in args.donors:
        for a in args.alphas:
            v = rows[f"{dname}@{a}"]
            mean_of = {
                k: float(np.mean([x[k] for x in v]))
                for k in (
                    "norm",
                    "cos_wA",
                    "cos_wB",
                    "cos_pushA",
                    "cos_pushB",
                    "z_down_A",
                    "z_down_B",
                )
            }
            au_A, tpr_A = detection(v, "A", "z_down_A", ben)
            _, flag_B = detection(v, "B", "z_down_B", ben)
            s = {
                **mean_of,
                "n": len(v),
                "A_retained": mean_of["cos_wA"] / ref_A,
                "B_acquired": (mean_of["cos_wB"] - floor_B) / (ref_B - floor_B),
                "B_acquired_push": (mean_of["cos_pushB"] - floor_pushB)
                / (1.0 - floor_pushB),
                "evasion_auroc_A": au_A,
                "tpr_A": tpr_A,
                "flag_rate_B": flag_B,
            }
            summary[f"{dname}@{a}"] = s
            print(
                f"{dname + '@' + str(a):14s} {s['norm']:7.2f} {s['A_retained']:7.2f} "
                f"{s['B_acquired']:7.2f} | {s['cos_pushA']:6.3f} {s['cos_pushB']:6.3f} | "
                f"{au_A:8.3f} {tpr_A:6.3f} {flag_B:6.3f} "
                f"{s['z_down_A']:+7.2f} {s['z_down_B']:+7.2f}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.adapter}"
    (args.out / f"{tag}.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "probe_source": args.probe_source,
                "kind": args.kind,
                "alphas": args.alphas,
                "donors": args.donors,
                "c_label": args.c_label,
                "layers": [0, read_layer - 1],
                "concepts": ho,
                "n_prefills": args.n_prefills,
                "n_donor_draws": args.n_donor_draws,
                "references": {
                    "identity_cos_wA": ref_A,
                    "floor_cos_wB": floor_B,
                    "pure_B_cos_wB": ref_B,
                    "floor_cos_pushB": floor_pushB,
                },
                "checks": checks,
                "skipped": skipped,
                "conditions": summary,
                "elapsed_min": round((time.time() - t0) / 60, 1),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out / f'{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
