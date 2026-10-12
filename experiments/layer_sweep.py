"""Measure the edit at every decoder layer: its norm, its direction and its alignment with w.

    python experiments/layer_sweep.py --runs base r1_sigmoid_mse

Reads  data/splits/, outputs/<model>/probes/<probe-source>/, outputs/<model>/adapters/<run>/
Writes experiments/outputs/layer_sweep/<model>.{json,png}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.activations import (  # noqa: E402
    MODELS,
    find_decoder_layers,
    load_model,
    model_dir,
    probes_dir,
    render,
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


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--runs", nargs="+", default=["base", "r1_sigmoid_mse"])
    ap.add_argument("--probe-source", default="base")
    ap.add_argument("--kind", default="logreg")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/layer_sweep")
    ap.add_argument("--n-prefills", type=int, default=30)
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ho = held_out_concepts()
    pools = load_pools(args.splits, ALL_CONCEPTS)
    pdir = probes_dir(args.outputs, args.model, source=args.probe_source)
    bank = MonitorSet.load(pdir / f"{args.kind}.npz")
    W = bank.params["w"].astype(np.float64)
    scale = bank.scale.astype(np.float64)
    read = bank.layer

    def episode(c: str, t: int):
        pj = [x for x in ho if x != c and not same_family(c, x)]
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

    out: dict = {
        "model": args.model,
        "read_layer": read,
        "concepts": ho,
        "n_prefills": args.n_prefills,
        "by_run": {},
    }
    t0 = time.time()
    for run in args.runs:
        model, tok = load_model(MODELS[args.model]["hf_name"])
        if run != "base":
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, model_dir(args.outputs, args.model) / "adapters" / run
            )
            model = model.merge_and_unload()
        model.eval()
        dec = find_decoder_layers(model)
        L = len(dec)

        def all_layers(messages, prefill, mdl=model, tk=tok, dec=dec):
            """Mean residual stream over the response span, at every decoder layer. [L, D]"""
            prompt_text, full_text = render(tk, messages, prefill)
            n_prompt = len(tk(prompt_text, add_special_tokens=False).input_ids)
            enc = tk(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=8192,
            ).to(mdl.device)
            if enc.input_ids.shape[1] <= n_prompt:
                raise SystemExit(
                    "response span empty -- the context overflowed max_length"
                )
            got: dict[int, torch.Tensor] = {}

            def mk(i):
                def hook(_m, _in, o):
                    h = (o[0] if isinstance(o, tuple) else o).detach()
                    got[i] = h[0, n_prompt:].float().mean(0).cpu()

                return hook

            hs = [dec[i].register_forward_hook(mk(i)) for i in range(len(dec))]
            try:
                with torch.inference_mode():
                    mdl(
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        use_cache=False,
                    )
            finally:
                for h in hs:
                    h.remove()
            return np.stack([got[i].numpy() for i in range(len(dec))]).astype(
                np.float64
            )

        per: dict[str, list] = {
            k: []
            for k in (
                "norm",
                "cos_read",
                "cos_w",
                "delta_z_contrast",
                "norm_om",
                "cos_read_om",
                "cos_w_om",
            )
        }
        for c in ho:
            k = bank.names.index(c)
            w, sc = unit(W[k]), scale[k]
            D = []
            for t in range(args.n_prefills):
                ep, row = episode(c, t)
                hu = all_layers(
                    with_imitate(ep, ep.cell.c_label).messages, row["sentence"]
                )
                hd = all_layers(
                    with_imitate(ep, 1 - ep.cell.c_label).messages, row["sentence"]
                )
                D.append(hu - hd)
            D = np.stack(D)
            ref = unit(D[:, read, :])
            per["norm"].append(np.linalg.norm(D, axis=2).mean(0))
            per["cos_read"].append((unit(D) * ref[:, None, :]).sum(-1).mean(0))
            per["cos_w"].append((unit(D) @ w).mean(0))
            per["delta_z_contrast"].append(((D @ w) / sc).mean(0))
            M = D.mean(0)
            per["norm_om"].append(np.linalg.norm(M, axis=1))
            per["cos_read_om"].append(unit(M) @ unit(M[read]))
            per["cos_w_om"].append(unit(M) @ w)
            print(f"  {run:16s} {c:26s} [{(time.time() - t0) / 60:.0f}m]", flush=True)
        out["by_run"][run] = {k: np.mean(v, 0).tolist() for k, v in per.items()}
        out["n_layers"] = L
        del model

    args.out.mkdir(parents=True, exist_ok=True)
    out["elapsed_min"] = round((time.time() - t0) / 60, 1)
    (args.out / f"{args.model}.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'layer':>6s}" + "".join(f"{r[:12]:>28s}" for r in out["by_run"]))
    print(
        f"{'':>6s}"
        + "".join(f"{'||dh||':>9s}{'cos_read':>10s}{'dz':>9s}" for _ in out["by_run"])
    )
    step = max(1, out["n_layers"] // 16)
    for i in range(0, out["n_layers"], step):
        row = f"{i:6d}"
        for r in out["by_run"]:
            v = out["by_run"][r]
            row += f"{v['norm'][i]:9.2f}{v['cos_read'][i]:10.3f}{v['delta_z_contrast'][i]:9.2f}"
        print(
            row + ("   <- readout (cos_read must be 1.000 here)" if i == read else "")
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for r, colour in zip(out["by_run"], ["#9e9e9e", "#d62728", "#1f77b4"]):
        v = out["by_run"][r]
        x = range(out["n_layers"])
        axes[0].plot(x, v["norm"], color=colour, lw=1.9, label=r)
        axes[1].plot(x, v["cos_read"], color=colour, lw=1.9, label=r)
        axes[2].plot(x, v["delta_z_contrast"], color=colour, lw=1.9, label=r)
    for ax, t in zip(
        axes,
        [
            r"$\|\Delta h_L\|$  (per episode)",
            r"$\cos(\Delta h_L,\ \Delta h_{\rm readout})$",
            r"$\Delta z^{\rm contrast}_L$  (logit lens)",
        ],
    ):
        ax.axvline(read, color="k", ls=(0, (1, 2)), lw=1.2)
        ax.set_xlabel("decoder layer")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_yscale("log")
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"{args.model} — where the push is written into the residual stream "
        f"(dotted = monitor layer {read})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out / f"{args.model}.png", dpi=180, bbox_inches="tight")
    print(f"\nwrote {args.out / f'{args.model}.json'} and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
