"""Figure 5 (Section 5.2): attention ablation and activation patching. CPU only.

    python figures/fig_mechanism.py

Reads  experiments/outputs/attn_knockout/llama-3.1-8b_r1_sigmoid_mse_{demo,digit}.json   panel a
       experiments/outputs/patch_interp/llama-3.1-8b_r1_sigmoid_mse.json                 panel b
Writes figures/outputs/mechanism.{pdf,png}

(a) TPR @ 1% FPR on the 10 held-out concepts after one class of attention edge is blocked (all
    heads, all layers). Each causal ablation has a random ablation of the same edge count beside
    it. The dashed line is the TPR of the intact organism. The artifact keys name the blocked
    queries first, so key `I<-D` is the paper's D ← I. Key block P is the paper's M.
(b) Concept A's instruction-block activations are interpolated toward a donor by α. "A retained"
    is cos(Δh, w_A) divided by its unpatched value. "B acquired" is 0 unpatched and 1 for a
    genuine concept-B episode. Solid lines use a concept-B donor. Dashed lines use the mean
    instruction block of the other concepts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import style  # noqa: E402

AK = ROOT / "experiments/outputs/attn_knockout"
DIG = AK / "llama-3.1-8b_r1_sigmoid_mse_digit.json"
DEM = AK / "llama-3.1-8b_r1_sigmoid_mse_demo.json"
PAT = ROOT / "experiments/outputs/patch_interp/llama-3.1-8b_r1_sigmoid_mse.json"

C_PATH, C_RAND = style.ROLE["organism"], style.ROLE["random"]
C_A, C_B = "#4C72B0", style.ROLE["organism"]


def panel_ablation(ax):
    cd = json.loads(DIG.read_text())["conditions"]
    ce = json.loads(DEM.read_text())["conditions"]
    base = float(ce["baseline"]["tpr"])
    conds = [
        (
            "demos $\\leftarrow$ instr\n(stage 1)",
            ce["I<-D@all"]["tpr"],
            ce["random(I<-D)@all"]["tpr"],
        ),
        (
            "instr $\\leftarrow$ monitored\n(stage 2, raw digit)",
            cd["P<-I\\digit+G,rawdigit@all"]["tpr"],
            cd["random(P<-I\\digit+G,rawdigit)@all"]["tpr"],
        ),
        (
            "demos $\\leftarrow$ monitored\n(direct path)",
            ce["P<-D@all"]["tpr"],
            ce["random(P<-D)@all"]["tpr"],
        ),
    ]
    x = np.arange(len(conds))
    w = 0.36
    for i, (_, causal, rand) in enumerate(conds):
        ax.bar(
            i - w / 2,
            causal,
            w,
            color=C_PATH,
            label="causal knockout" if i == 0 else None,
        )
        ax.bar(
            i + w / 2,
            rand,
            w,
            color=C_RAND,
            hatch="///",
            edgecolor="white",
            label="matched random" if i == 0 else None,
        )
        ax.text(i - w / 2, causal + 0.02, f"{causal:.2f}", ha="center", fontsize=7.6)
        ax.text(
            i + w / 2,
            rand + 0.02,
            f"{rand:.2f}",
            ha="center",
            fontsize=7.6,
            color="#555",
        )
    ax.axhline(base, color="#333333", ls="--", lw=1.2)
    ax.text(
        len(conds) - 1 + 0.34,
        base + 0.015,
        f"intact evasion {base:.2f}",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )
    ax.set_xticks(x, [c[0] for c in conds], fontsize=7.2)
    ax.set_ylabel("TPR @ 1% FPR after knockout")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=7.6, loc="upper right", ncol=1, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def panel_patching(ax):
    d = json.loads(PAT.read_text())
    al = d["alphas"]
    cr_A = [d["conditions"][f"cross@{a}"]["A_retained"] for a in al]
    cr_B = [d["conditions"][f"cross@{a}"]["B_acquired"] for a in al]
    mn_A = [d["conditions"][f"mean@{a}"]["A_retained"] for a in al]
    mn_B = [d["conditions"][f"mean@{a}"]["B_acquired"] for a in al]
    ax.plot(al, cr_A, "-", color=C_A, lw=2.3, label="A retained (toward B)")
    ax.plot(al, cr_B, "-", color=C_B, lw=2.3, label="B acquired (toward B)")
    ax.plot(
        al, mn_A, "--", color=C_A, lw=1.7, alpha=0.85, label="A retained (toward mean)"
    )
    ax.plot(
        al, mn_B, "--", color=C_B, lw=1.7, alpha=0.85, label="B acquired (toward mean)"
    )
    ax.set_xlabel(r"Interpolation $\alpha$")
    ax.set_ylabel("Fraction of cosine similarity")
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.5, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--stem", default="mechanism")
    args = ap.parse_args()
    style.apply()

    fig, axes = plt.subplots(
        1, 2, figsize=(8.8, 3.0), gridspec_kw={"width_ratios": [1.12, 1]}
    )
    panel_ablation(axes[0])
    panel_patching(axes[1])
    fig.tight_layout()
    fig.canvas.draw()
    for ax, letter in zip(axes, "ab"):
        e = ax.get_window_extent()
        x0 = e.x0 / (fig.get_figwidth() * fig.dpi)
        y1 = e.y1 / (fig.get_figheight() * fig.dpi)
        fig.text(
            x0 - 0.052,
            min(y1 + 0.070, 0.998),
            letter,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
        )
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"{args.stem}.{ext}", bbox_inches="tight")
    print(f"wrote {args.out / f'{args.stem}.pdf'} (+.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
