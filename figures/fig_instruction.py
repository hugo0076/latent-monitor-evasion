"""Figure 8 (Appendix E.4): the base-model edit against the instruction wording. CPU only.

    python figures/fig_instruction.py

Reads  experiments/outputs/prompt_ablation/<model>_named.json   panel a
       experiments/outputs/verb_sweep/<model>.json              panel b
Writes figures/outputs/instruction.{pdf,png}

(a) Paired Cohen's d_z of the base edit for four phrasings of the imitate instruction. Each bar
    is the mean over the five system-prompt variants, and the error bar spans their min and max.
(b) Per-episode cos(Δh, w) for each instruction verb, sorted. The paper uses "imitate".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402

ART = ROOT / "experiments/outputs"
C_BASE = style.MODEL_COLOR["llama-3.1-8b"]


PHRASINGS = [
    ("standard", "standard"),
    ("terse", "terse"),
    ("directional", "directional"),
    ("named_d", "named"),
]


def panel_phrasing(ax, model):
    d = json.loads((ART / "prompt_ablation" / f"{model}_named.json").read_text())
    br = d["by_run"]["base"]
    systems = d["systems"]
    instrs = [k for k, _ in PHRASINGS]
    labels = [v for _, v in PHRASINGS]
    x = np.arange(len(instrs))
    vals = [[br[f"{s}|{i}"]["cohens_dz"] for s in systems] for i in instrs]
    means = [np.mean(v) for v in vals]
    err = [
        np.subtract(means, [min(v) for v in vals]),
        np.subtract([max(v) for v in vals], means),
    ]
    ax.bar(
        x,
        means,
        0.6,
        color=C_BASE,
        edgecolor=style.INK,
        linewidth=0.6,
        yerr=err,
        capsize=3,
        error_kw={"lw": 0.8, "ecolor": style.INK},
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Paired Cohen's $d_z$")
    ax.set_title("Instruction phrasing")
    ax.axhline(0, color=style.MUTED, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


def panel_verb(ax, model):
    d = json.loads((ART / "verb_sweep" / f"{model}.json").read_text())
    br = d["by_run"]["base"]
    items = sorted(br.items(), key=lambda kv: kv[1]["cos_dh_w"])
    verbs = [k for k, _ in items]
    cos = [v["cos_dh_w"] for _, v in items]
    y = np.arange(len(verbs))
    ax.barh(y, cos, color=C_BASE, edgecolor=style.INK, linewidth=0.5)
    ax.set_yticks(y, verbs, fontsize=style.FS["annot"] - 1.4)
    ax.set_ylim(-0.8, len(verbs) - 0.2)
    i = verbs.index("imitate")
    ax.annotate(
        "used in the paper",
        (cos[i], i),
        fontsize=style.FS["annot"],
        va="center",
        xytext=(6, 0),
        textcoords="offset points",
        color=style.INK,
    )
    ax.axvline(0, color=style.MUTED, lw=0.7)
    ax.set_xlabel(r"$\cos(\Delta h,\ w)$")
    ax.set_title("Instruction verb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.22)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    args = ap.parse_args()
    style.apply()

    fig, axes = plt.subplots(
        1, 2, figsize=(8.8, 4.4), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    panel_phrasing(axes[0], args.model)
    panel_verb(axes[1], args.model)
    fig.tight_layout()

    fig.canvas.draw()
    for ax, letter in zip(axes, "ab"):
        e = ax.get_window_extent()
        x0 = e.x0 / (fig.get_figwidth() * fig.dpi)
        y1 = e.y1 / (fig.get_figheight() * fig.dpi)
        fig.text(
            x0 - 0.052,
            min(y1 + 0.055, 0.998),
            letter,
            fontsize=style.FS["letter"],
            fontweight="bold",
            va="top",
            ha="left",
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"instruction.{ext}", bbox_inches="tight")
    print(f"wrote {args.out / 'instruction.pdf'} (+.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
