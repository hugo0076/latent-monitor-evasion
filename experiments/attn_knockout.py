"""Block attention edges between episode segments and measure the effect on evasion.

    python experiments/attn_knockout.py --adapter r1_sigmoid_mse --edges "I<-D" "G<-D" "P<-D" --suffix _demo
    python experiments/attn_knockout.py --adapter r1_sigmoid_mse --digit
    python experiments/attn_knockout.py --adapter base --edges "I<-D" "G<-D" "P<-D" --suffix _demo
    python experiments/attn_knockout.py --adapter base --digit

Reads  data/splits/, outputs/<model>/probes/<probe-source>/, outputs/<model>/adapters/<adapter>/
Writes experiments/outputs/attn_knockout/<model>_<adapter><suffix>.json   edge sweep
       experiments/outputs/attn_knockout/<model>_<adapter>_digit.json     with --digit
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

HERE = Path(__file__).resolve().parent
BLOCKS = ("D", "I", "G", "P")
EDGES = [
    ("I<-D", "I", ["D"]),
    ("G<-D", "G", ["D"]),
    ("P<-D", "P", ["D"]),
    ("P<-I", "P", ["I"]),
    ("P<-G", "P", ["G"]),
    ("P<-I+G", "P", ["I", "G"]),
]


def spans(tok, messages: list, prefill: str) -> dict[str, tuple[int, int]]:
    """Token boundaries of the four blocks, from cumulative prefixes of the rendered text."""
    n = lambda t: len(tok(t, add_special_tokens=False).input_ids)  # noqa: E731
    demos_end = n(
        tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=False
        )
    )
    instr_end = n(
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    )
    prompt, full = render(tok, messages, prefill)
    return {
        "D": (0, demos_end),
        "I": (demos_end, instr_end),
        "G": (instr_end, n(prompt)),
        "P": (n(prompt), n(full)),
    }


def block_mask(
    sp: dict, q_blk: str, k_blks: list[str], total: int, device, dtype
) -> torch.Tensor:
    """[1, 1, T, T] additive mask blocking every causally valid (query in q_blk, key in k_blks)."""
    m = torch.zeros((1, 1, total, total), device=device, dtype=dtype)
    qa, qb = sp[q_blk]
    for kb_name in k_blks:
        ka, kb = sp[kb_name]
        m[:, :, qa:qb, ka:kb] = torch.finfo(dtype).min
    causal = torch.tril(torch.ones((total, total), device=device, dtype=torch.bool))
    m = m.masked_fill(~causal, 0.0)
    m[:, :, torch.arange(total), torch.arange(total)] = 0.0
    return m


def random_mask(
    n_pairs: int, total: int, device, dtype, rng: np.random.Generator
) -> torch.Tensor:
    """Block `n_pairs` (query, key) pairs, drawn uniformly from below the diagonal."""
    m = torch.zeros((1, 1, total, total), device=device, dtype=dtype)
    if n_pairs <= 0:
        return m
    qi, ki = np.tril_indices(total, k=-1)
    pick = rng.choice(len(qi), size=min(n_pairs, len(qi)), replace=False)
    m[
        :,
        :,
        torch.as_tensor(qi[pick], device=device),
        torch.as_tensor(ki[pick], device=device),
    ] = torch.finfo(dtype).min
    return m


DIGIT_CONDS = ("P<-I\\digit+G", "P<-digit-only", "P<-digit", "P<-I\\digit+G,rawdigit")


def digit_mask(
    sp: dict, total: int, dpos: int, kind: str, device, dtype
) -> torch.Tensor:
    """Masks that keep the digit position open, or cut only the digit.

    `P<-I\\digit+G`          P keeps D and the digit. The rest of I and all of G are cut.
    `P<-digit-only`          as above, and D is also cut. The digit is P's only differing input.
    `P<-digit`               only the digit column is cut. All other edges stay open.
    `P<-I\\digit+G,rawdigit` as the first, and the digit cannot attend to earlier tokens.
                             The digit then carries the label only, not what it read from D.
    """
    lo = torch.finfo(dtype).min
    qa, qb = sp["P"]
    if kind == "P<-digit":
        m = torch.zeros((1, 1, total, total), device=device, dtype=dtype)
        m[:, :, qa:qb, dpos] = lo
    else:
        keys = ["D", "I", "G"] if kind == "P<-digit-only" else ["I", "G"]
        m = block_mask(sp, "P", keys, total, device, dtype)
        m[:, :, qa:qb, dpos] = 0.0
        if kind.endswith("rawdigit"):
            m[:, :, dpos, :dpos] = lo
    causal = torch.tril(torch.ones((total, total), device=device, dtype=torch.bool))
    m = m.masked_fill(~causal, 0.0)
    m[:, :, torch.arange(total), torch.arange(total)] = 0.0
    return m


def digit_position(tok, up_messages: list, dn_messages: list, prefill: str) -> int:
    """The single token position at which the two arms differ."""
    iu = tok(render(tok, up_messages, prefill)[1], add_special_tokens=False).input_ids
    idn = tok(render(tok, dn_messages, prefill)[1], add_special_tokens=False).input_ids
    if len(iu) != len(idn):
        raise SystemExit(f"arms tokenize to different lengths: {len(iu)} vs {len(idn)}")
    diff = [i for i, (a, b) in enumerate(zip(iu, idn)) if a != b]
    if len(diff) != 1:
        raise SystemExit(
            f"arms differ at {len(diff)} positions, expected exactly 1: {diff[:8]}"
        )
    return diff[0]


class Knockout:
    """Adds a fixed additive mask to the attention scores of selected decoder layers."""

    def __init__(self, model, layers: list[int]) -> None:
        self.layers = find_decoder_layers(model)
        self.targets = layers
        self.mask: torch.Tensor | None = None
        self.handles: list = []

    def __enter__(self):
        def hook(_mod, args, kwargs):
            am = kwargs.get("attention_mask")
            if self.mask is None:
                return None
            if am is None:
                raise SystemExit(
                    "no attention_mask on the decoder layer; eager is required"
                )
            kwargs["attention_mask"] = am + self.mask.to(am.dtype)
            return args, kwargs

        for i in self.targets:
            self.handles.append(
                self.layers[i].register_forward_pre_hook(hook, with_kwargs=True)
            )
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _dz(d: np.ndarray) -> float:
    """Paired Cohen's dz over prefills, or 0 when the differences have zero spread."""
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd else 0.0


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
    ap.add_argument("--out", type=Path, default=HERE / "outputs/attn_knockout")
    ap.add_argument("--n-prefills", type=int, default=50)
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument("--edges", nargs="+", default=[e[0] for e in EDGES])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--suffix",
        default="",
        help="appended to the output file name",
    )
    ap.add_argument(
        "--digit",
        action="store_true",
        help="run the digit conditions, the P-side edges and the per-layer digit readout "
        "instead of the edge sweep",
    )
    args = ap.parse_args()

    concepts = held_out_concepts()
    pools = load_pools(args.splits, ALL_CONCEPTS)
    mdir = model_dir(args.outputs, args.model)
    monitors = MonitorSet.load(
        probes_dir(args.outputs, args.model, source=args.probe_source)
        / f"{args.kind}.npz"
    )

    def episode(c: str, t: int):
        pj = [x for x in concepts if x != c and not same_family(c, x)]
        row = eval_prefills(pools[c])[t]
        rng = random.Random(seed_for("eval", args.seed, c, row["item_id"], t))
        cell = next(x for x in ALL_CELLS if x.c_label == t % 2 and x.prefill_dir == 1)
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

    model, tok = load_model(MODELS[args.model]["hf_name"], attn="eager")
    if args.adapter and args.adapter != "base":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, mdir / "adapters" / args.adapter)
        model = model.merge_and_unload()
    model.eval()
    layers_all = list(range(len(find_decoder_layers(model))))

    plan = []
    for c in concepts:
        for t in range(args.n_prefills):
            ep, row = episode(c, t)
            up = with_imitate(ep, ep.cell.c_label)
            dn = with_imitate(ep, 1 - ep.cell.c_label)
            plan.append(
                (
                    c,
                    t,
                    row["sentence"],
                    up.messages,
                    dn.messages,
                    spans(tok, dn.messages, row["sentence"]),
                    digit_position(tok, up.messages, dn.messages, row["sentence"]),
                )
            )
    print(
        f"{len(plan)} episodes; blocks (first): "
        + "  ".join(f"{b}={plan[0][5][b][1] - plan[0][5][b][0]}" for b in BLOCKS),
        flush=True,
    )

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rng = np.random.default_rng(args.seed)

    pdir = probes_dir(args.outputs, args.model, source=args.probe_source)
    S = np.load(pdir / f"scores_{args.kind}.npy")
    conc = np.array(
        [
            json.loads(x)["concept"]
            for x in (pdir / "index.jsonl").read_text().splitlines()
            if x.strip()
        ]
    )
    names = list(monitors.names)
    benign = {}
    for c in concepts:
        pool = S[
            np.isin(conc, [p for p in concepts if p != c and not same_family(c, p)]),
            names.index(c),
        ]
        benign[c] = (pool, float(np.quantile(pool, 0.99)))

    def measure(make_mask, layers: list[int] | None) -> dict:
        """Run every episode under one mask condition and score both arms.

        The benign pool and threshold come from stage 03 and do not change with the mask."""
        ko = Knockout(model, layers or [])
        rows = {c: {"up": [], "dn": [], "dh": []} for c in concepts}
        with ko:
            for idx, (c, t, sent, up_m, dn_m, sp, dpos) in enumerate(plan):
                total = sp["P"][1]
                ko.mask = None if make_mask is None else make_mask(sp, total, idx, dpos)
                if ko.mask is not None and bool((ko.mask[0, 0] < 0).all(-1).any()):
                    raise SystemExit("a query row is fully masked; would produce NaN")
                k = monitors.index[c]
                hu = response_activations(model, tok, monitors.layer, up_m, sent)
                hd = response_activations(model, tok, monitors.layer, dn_m, sent)
                rows[c]["up"].append(
                    float(
                        monitors.logits(hu[None], np.ones((1, len(hu)), np.float32))[0][
                            k
                        ]
                    )
                )
                rows[c]["dn"].append(
                    float(
                        monitors.logits(hd[None], np.ones((1, len(hd)), np.float32))[0][
                            k
                        ]
                    )
                )
                rows[c]["dh"].append(hu.mean(0) - hd.mean(0))
        per, au, tp, zz, nn, cc, dzs = {}, [], [], [], [], [], []
        for c in concepts:
            k = monitors.index[c]
            pool, thr = benign[c]
            dn = np.array(rows[c]["dn"])
            dh = np.mean(rows[c]["dh"], 0).astype(np.float64)
            w = monitors.params["w"][k].astype(np.float64)
            per[c] = {
                "auroc": float(
                    roc_auc_score(
                        np.r_[np.ones(len(dn)), np.zeros(len(pool))], np.r_[dn, pool]
                    )
                ),
                "tpr": float((dn > thr).mean()),
                "z_down": float(dn.mean()),
                "gap_up_down": float(np.mean(rows[c]["up"]) - dn.mean()),
                "dh_norm": float(np.linalg.norm(dh)),
                "cos_dh_w": float(dh @ w / (np.linalg.norm(dh) * np.linalg.norm(w))),
                "cohens_dz": _dz(np.asarray(rows[c]["up"]) - dn),
            }
            au.append(per[c]["auroc"])
            tp.append(per[c]["tpr"])
            zz.append(per[c]["z_down"])
            nn.append(per[c]["dh_norm"])
            cc.append(per[c]["cos_dh_w"])
            dzs.append(per[c]["cohens_dz"])
        return {
            "auroc": float(np.mean(au)),
            "tpr": float(np.mean(tp)),
            "z_down": float(np.mean(zz)),
            "dh_norm": float(np.mean(nn)),
            "cos_dh_w": float(np.mean(cc)),
            "cohens_dz": float(np.mean(dzs)),
            "by_concept": per,
        }

    t0 = time.time()
    results = {"baseline": measure(None, None)}
    b = results["baseline"]
    print(
        f"  {'baseline':28s} AUROC {b['auroc']:.4f}  TPR {b['tpr']:.3f}  "
        f"z {b['z_down']:+6.2f}  ||dh|| {b['dh_norm']:6.2f}  cos {b['cos_dh_w']:+.3f}",
        flush=True,
    )

    def digit_readout() -> dict:
        """Residual stream at the digit position, per layer, both arms, no mask.

        Reports cos(h_up - h_down, w), the edit norm and the activation norm per layer. `w` is
        the monitored-layer probe direction at every depth, as in `layer_profile`. At other
        layers the cosine is a projection onto w, not a monitor score.
        """
        nl = len(find_decoder_layers(model)) + 1
        acc = {
            c: {
                "dh": np.zeros((nl, model.config.hidden_size)),
                "hn": np.zeros(nl),
                "n": 0,
            }
            for c in concepts
        }
        for c, t, sent, up_m, dn_m, sp, dpos in plan:
            hs = []
            for msgs in (up_m, dn_m):
                ids = tok(
                    render(tok, msgs, sent)[1],
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(dev)
                with torch.no_grad():
                    out = model(ids, output_hidden_states=True, use_cache=False)
                hs.append(
                    torch.stack([h[0, dpos] for h in out.hidden_states])
                    .float()
                    .cpu()
                    .numpy()
                )
            acc[c]["dh"] += hs[0] - hs[1]
            acc[c]["hn"] += np.linalg.norm(hs[0], axis=1)
            acc[c]["n"] += 1
        prof = {}
        for c in concepts:
            k = monitors.index[c]
            w = monitors.params["w"][k].astype(np.float64)
            dh = acc[c]["dh"] / acc[c]["n"]
            nrm = np.linalg.norm(dh, axis=1)
            prof[c] = {
                "dh_norm": nrm.tolist(),
                "h_norm": (acc[c]["hn"] / acc[c]["n"]).tolist(),
                "cos_dh_w": np.where(
                    nrm > 0, dh @ w / (np.maximum(nrm, 1e-12) * np.linalg.norm(w)), 0.0
                ).tolist(),
            }
        return {
            "by_concept": prof,
            "mean_cos_dh_w": np.mean(
                [prof[c]["cos_dh_w"] for c in concepts], 0
            ).tolist(),
            "mean_dh_norm": np.mean([prof[c]["dh_norm"] for c in concepts], 0).tolist(),
            "mean_h_norm": np.mean([prof[c]["h_norm"] for c in concepts], 0).tolist(),
        }

    if args.digit:

        def show(tag: str, r: dict) -> None:
            print(
                f"  {tag:30s} AUROC {r['auroc']:.4f}  TPR {r['tpr']:.3f}  "
                f"||dh|| {r['dh_norm']:6.2f}  cos {r['cos_dh_w']:+.3f}  "
                f"dz {r['cohens_dz']:+6.2f}  [{(time.time() - t0) / 60:.0f}m]",
                flush=True,
            )

        for tag, q, ks in [e for e in EDGES if e[0] in ("P<-I", "P<-G", "P<-I+G")]:
            results[f"{tag}@all"] = measure(
                lambda sp, total, _i, _d, q=q, ks=ks: block_mask(
                    sp, q, ks, total, dev, dtype
                ),
                layers_all,
            )
            show(f"{tag}@all", results[f"{tag}@all"])

        for kind in DIGIT_CONDS:
            results[f"{kind}@all"] = measure(
                lambda sp, total, _i, dpos, kind=kind: digit_mask(
                    sp, total, dpos, kind, dev, dtype
                ),
                layers_all,
            )
            show(f"{kind}@all", results[f"{kind}@all"])
            counts = [
                int((digit_mask(sp, sp["P"][1], dpos, kind, dev, dtype) < 0).sum())
                for *_, sp, dpos in plan
            ]
            results[f"random({kind})@all"] = measure(
                lambda sp, total, i, _d, counts=counts: random_mask(
                    counts[i], total, dev, dtype, rng
                ),
                layers_all,
            )
            show(f"random({kind})@all", results[f"random({kind})@all"])

        results["digit_readout"] = digit_readout()
        mc = results["digit_readout"]["mean_cos_dh_w"]
        print(
            "  digit readout cos(dh, w) by layer: "
            + " ".join(f"{x:+.2f}" for x in mc[::4]),
            flush=True,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        tag = f"{args.model}_{args.adapter}_digit"
        (args.out / f"{tag}.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "adapter": args.adapter,
                    "n_prefills": args.n_prefills,
                    "concepts": concepts,
                    "layer_sets": {"all": layers_all},
                    "edges": list(DIGIT_CONDS),
                    "attn_implementation": "eager",
                    "conditions": results,
                    "elapsed_min": round((time.time() - t0) / 60, 1),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.out / f'{tag}.json'}")
        return 0

    for tag, q, ks in EDGES:
        if tag not in args.edges:
            continue
        r = measure(
            lambda sp, total, _i, _d, q=q, ks=ks: block_mask(
                sp, q, ks, total, dev, dtype
            ),
            layers_all,
        )
        results[f"{tag}@all"] = r
        print(
            f"  {tag + '@all':28s} AUROC {r['auroc']:.4f}  TPR {r['tpr']:.3f}  "
            f"z {r['z_down']:+6.2f}  ||dh|| {r['dh_norm']:6.2f}  cos {r['cos_dh_w']:+.3f}"
            f"  [{(time.time() - t0) / 60:.0f}m]",
            flush=True,
        )
    for tag, q, ks in EDGES:
        if tag not in args.edges:
            continue
        counts = []
        for *_, sp, _dp in plan:
            qa, qb = sp[q]
            tri = np.tril(np.ones((sp["P"][1], sp["P"][1])), -1)
            counts.append(sum(int(tri[qa:qb, sp[k][0] : sp[k][1]].sum()) for k in ks))
        results[f"random({tag})@all"] = measure(
            lambda sp, total, i, _d, counts=counts: random_mask(
                counts[i], total, dev, dtype, rng
            ),
            layers_all,
        )
        r = results[f"random({tag})@all"]
        print(
            f"  {'random(' + tag + ')@all':28s} AUROC {r['auroc']:.4f}  "
            f"TPR {r['tpr']:.3f}  z {r['z_down']:+6.2f}  ||dh|| {r['dh_norm']:6.2f}  "
            f"cos {r['cos_dh_w']:+.3f}  [{(time.time() - t0) / 60:.0f}m]",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.adapter}{args.suffix}"
    (args.out / f"{tag}.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "adapter": args.adapter,
                "n_prefills": args.n_prefills,
                "concepts": concepts,
                "layer_sets": {"all": layers_all},
                "edges": args.edges,
                "attn_implementation": "eager",
                "conditions": results,
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
