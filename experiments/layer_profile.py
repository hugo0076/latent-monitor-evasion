"""Profile the edit size, and the organism's divergence from the base model, at every layer.

    python experiments/layer_profile.py

Reads  data/splits/, outputs/<model>/probes/base/logreg.npz, outputs/<model>/adapters/<adapter>/
Writes experiments/outputs/layer_profile/<model>.{json,png}
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
from core.concepts import ALL_CONCEPTS, held_out_concepts  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    ap.add_argument("--adapter", default="r1_sigmoid_mse")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    ap.add_argument("--out", type=Path, default=HERE / "outputs/layer_profile")
    ap.add_argument("--n-prefills", type=int, default=20)
    ap.add_argument("--n-demo-pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ho = held_out_concepts()
    n = 2 * len(ho) * args.n_prefills * 2
    print(
        f"2 models x {len(ho)} concepts x {args.n_prefills} prefills x 2 arms = {n:,} "
        f"forwards, all layers hooked in one pass (~{n / 8.6 / 60:.0f} min)"
    )

    pools = load_pools(args.splits, ALL_CONCEPTS)
    pdir = probes_dir(args.outputs, args.model, source="base")
    bank = MonitorSet.load(pdir / "logreg.npz")
    target_layer = bank.layer
    mdir = model_dir(args.outputs, args.model)

    def episode(c: str, t: int):
        pj = [x for x in ho if x != c]
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
        )

    store: dict = {}
    t0 = time.time()
    for run in ("base", args.adapter):
        model, tok = load_model(MODELS[args.model]["hf_name"])
        if run != "base":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, mdir / "adapters" / run)
            model = model.merge_and_unload()
        model.eval()
        dec = find_decoder_layers(model)
        L = len(dec)

        def all_layers(
            messages, prefill, model=model, tok=tok, dec=dec, L=L
        ) -> np.ndarray:
            """Mean-pooled activation over the response span, at every decoder layer. [L, D]."""
            prompt_text, full_text = render(tok, messages, prefill)
            n_prompt = len(tok(prompt_text, add_special_tokens=False).input_ids)
            enc = tok(
                full_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=8192,
            ).to(model.device)
            if enc.input_ids.shape[1] <= n_prompt:
                raise ValueError("context overflow: the scored span is empty")
            cap: dict[int, torch.Tensor] = {}

            def mk(i):
                def hook(_m, _in, out):
                    h = out[0] if isinstance(out, tuple) else out
                    cap[i] = h[0, n_prompt:].float().mean(0).detach().cpu()

                return hook

            hs = [dec[i].register_forward_hook(mk(i)) for i in range(L)]
            try:
                with torch.inference_mode():
                    model(**enc, use_cache=False)
            finally:
                for h in hs:
                    h.remove()
            return torch.stack([cap[i] for i in range(L)]).numpy().astype(np.float64)

        per = {"up": [], "down": [], "bare": []}
        for c in ho:
            for t in range(args.n_prefills):
                ep = episode(c, t)
                for arm, im in (("up", ep.cell.c_label), ("down", 1 - ep.cell.c_label)):
                    e = with_imitate(ep, im)
                    per[arm].append(all_layers(e.messages, e.prefill))
                per["bare"].append(
                    all_layers(
                        [{"role": "user", "content": "Say something."}], ep.prefill
                    )
                )
        store[run] = {k: np.stack(v) for k, v in per.items()}
        del model
        torch.cuda.empty_cache()
        print(f"  {run:18s} done [{(time.time() - t0) / 60:.0f}m]", flush=True)

    B = store["base"]["bare"]
    scale = np.linalg.norm(B - B.mean(0, keepdims=True), axis=-1).mean(0)

    prof: dict = {
        "layers": list(range(B.shape[1])),
        "target_layer": target_layer,
        "scale": scale.tolist(),
    }
    for run in store:
        dh = store[run]["up"] - store[run]["down"]
        prof[f"A_edit_{run}"] = (np.linalg.norm(dh, axis=-1).mean(0) / scale).tolist()
    W = bank.params["w"].astype(np.float64)
    SC = np.asarray(bank.scale, dtype=np.float64)
    ki = [bank.names.index(c) for c in ho]
    per_c = np.repeat(np.arange(len(ho)), args.n_prefills)
    for run in store:
        dh = store[run]["up"] - store[run]["down"]
        wz = np.stack([W[ki[j]] for j in per_c])
        sz = np.array([SC[ki[j]] for j in per_c])[:, None]
        prof[f"dz_{run}"] = (np.einsum("nld,nd->nl", dh, wz) / sz).mean(0).tolist()

    dv = store[args.adapter]["bare"] - store["base"]["bare"]
    prof["B_divergence"] = (np.linalg.norm(dv, axis=-1).mean(0) / scale).tolist()
    dv_ep = store[args.adapter]["down"] - store["base"]["down"]
    prof["B_divergence_inepisode"] = (
        np.linalg.norm(dv_ep, axis=-1).mean(0) / scale
    ).tolist()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}.json").write_text(json.dumps(prof, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
    x = prof["layers"]
    axes[0].plot(
        x,
        prof[f"A_edit_{args.adapter}"],
        "o-",
        color="#e8684a",
        lw=2,
        ms=3.5,
        label="model organism",
    )
    axes[0].plot(
        x, prof["A_edit_base"], "s-", color="#9aa5b1", lw=1.6, ms=3, label="base model"
    )
    axes[0].set_ylabel(r"$\Vert \Delta h_L \Vert$ / centred norm", fontsize=8)
    axes[0].set_title("A  edit magnitude (elevate $-$ suppress)", fontsize=9.5)
    axes[1].plot(
        x,
        prof["B_divergence"],
        "o-",
        color="#5b8ff9",
        lw=2,
        ms=3.5,
        label="bare context",
    )
    axes[1].plot(
        x,
        prof["B_divergence_inepisode"],
        "s-",
        color="#2f7ac6",
        lw=1.6,
        ms=3,
        label="in episode, suppressing",
    )
    axes[1].set_ylabel(
        r"$\Vert h_{\rm FT} - h_{\rm base} \Vert$ / centred norm", fontsize=8
    )
    axes[1].set_title("B  divergence from the base model, same inputs", fontsize=9.5)
    axes[2].plot(
        x,
        prof[f"dz_{args.adapter}"],
        "o-",
        color="#e8684a",
        lw=2,
        ms=3.5,
        label="model organism",
    )
    axes[2].plot(
        x, prof["dz_base"], "s-", color="#9aa5b1", lw=1.6, ms=3, label="base model"
    )
    axes[2].axhline(0, color="#52606d", lw=0.7, ls=":")
    axes[2].set_ylabel(r"$\Delta z_L$ (layer-24 probe read at $L$)", fontsize=8)
    axes[2].set_title(
        "C  logit-lens: the monitor direction at every depth", fontsize=9.5
    )

    for ax in axes:
        ax.axvline(target_layer, color="#1f2933", ls="--", lw=1.1)
        ax.text(
            target_layer + 0.4,
            ax.get_ylim()[1] * 0.94,
            f"monitored\nlayer {target_layer}",
            fontsize=7,
            va="top",
            color="#1f2933",
        )
        ax.set_xlabel("decoder layer", fontsize=8)
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"{args.model}: where in depth does the edit live? "
        f"({len(ho)} held-out concepts, {args.n_prefills} prefills)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out / f"{args.model}.png", dpi=160, bbox_inches="tight")

    print(
        f"\n{'layer':>6s} {'A organism':>11s} {'A base':>9s} {'B bare':>9s} "
        f"{'B episode':>10s} {'dz organism':>12s} {'dz base':>9s}"
    )
    for i in x:
        star = " <-- monitored" if i == target_layer else ""
        print(
            f"{i:6d} {prof[f'A_edit_{args.adapter}'][i]:11.4f} {prof['A_edit_base'][i]:9.4f} "
            f"{prof['B_divergence'][i]:9.4f} {prof['B_divergence_inepisode'][i]:10.4f} "
            f"{prof[f'dz_{args.adapter}'][i]:12.3f} {prof['dz_base'][i]:9.3f}{star}"
        )
    print(f"\nwrote {args.out / f'{args.model}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
