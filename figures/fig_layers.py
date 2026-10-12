"""Figure 6 (Appendix D.2): the edit over depth, base model and model organism. CPU only.

    python figures/fig_layers.py

Reads  experiments/outputs/layer_sweep/<model>.json     panel a
       experiments/outputs/layer_profile/<model>.json   panel b
Writes figures/outputs/layers.{pdf,png}

(a) cos(Δh_L, w) at each layer, averaged over episodes. w is fitted at the monitored layer, so
    the values at other layers are projections onto that fixed direction.
(b) ||Δh_L|| divided by the centred activation norm at the same layer. Raw residual-stream norms
    grow steeply with depth, so an unnormalised norm rises whatever the edit does.
The dotted line marks the monitored layer, where the adapter's loss acts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402

ART = ROOT / "experiments/outputs/layer_profile"
SWEEP = ROOT / "experiments/outputs/layer_sweep"
C_ORG, C_BASE = style.ROLE["organism"], "#888888"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--stem", default="layers")
    args = ap.parse_args()
    style.apply()

    d = json.loads((ART / f"{args.model}.json").read_text())
    sw = json.loads((SWEEP / f"{args.model}.json").read_text())["by_run"]
    d["cos_w_base"] = sw["base"]["cos_w"]
    d["cos_w_r1_sigmoid_mse"] = sw["r1_sigmoid_mse"]["cos_w"]
    L, tl = d["layers"], int(d["target_layer"])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0))

    panels = [
        (
            axes[0],
            "cos_w_base",
            "cos_w_r1_sigmoid_mse",
            r"$\cos(\Delta h_L,\, w)$",
            "Alignment with the monitor direction",
        ),
        (
            axes[1],
            "A_edit_base",
            "A_edit_r1_sigmoid_mse",
            r"normalised $\|\Delta h_L\|$",
            "Edit magnitude by layer",
        ),
    ]
    for ax, kb, ko, ylab, title in panels:
        ax.plot(L, d[ko], "o-", color=C_ORG, lw=2.2, ms=3.4, label="model organism")
        ax.plot(L, d[kb], "s--", color=C_BASE, lw=1.8, ms=3.0, label="base")
        ax.axvline(tl, ls=":", color="#333333", lw=1.1)
        ax.set_xlabel("layer")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(1.0 / 64.0, ls="--", color="#bbbbbb", lw=1.0)
    axes[0].text(0.5, 1.0 / 64.0 + 0.006, "chance", fontsize=7, color="#888888")
    axes[1].annotate(
        "monitored layer",
        xy=(tl, max(d["A_edit_r1_sigmoid_mse"])),
        xytext=(tl - 9.5, 1.95),
        fontsize=7.5,
        color="#333333",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"),
    )

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
