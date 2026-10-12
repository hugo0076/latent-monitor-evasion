"""Figure 7 (Appendix D.3): the edit as the pseudo-direction rotates. CPU only.

    python figures/fig_pseudo.py

Reads  experiments/outputs/in_span_directions/<model>_v2.json
Writes figures/outputs/pseudo.{pdf,png}

(a) The edit's components on the rotation basis (e1 = w_A, e2 toward w_B), mean over pairs.
    The dashed grey curves are cos θ and sin θ.
(b) cos(Δh, r̂_θ), where r̂_θ is the tagged split's difference in mean activations. Bold lines
    are the mean over pairs, faint lines are the single pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style  # noqa: E402

ART = ROOT / "experiments/outputs/in_span_directions"

RUNS = [("base", "base"), ("r1_sigmoid_mse", "model organism")]


def clean(rows: list[dict]) -> list[dict]:
    """Rows with no tag corruption."""
    return [r for r in rows if r["corrupt_p"] == 0.0]


def pooled(rows: list[dict], key: str, thetas: list[int]) -> np.ndarray:
    return np.array(
        [np.mean([r[key] for r in rows if r["theta"] == t]) for t in thetas]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--stem", default="pseudo")
    args = ap.parse_args()

    d = json.loads((ART / f"{args.model}_v2.json").read_text())
    thetas = sorted({r["theta"] for r in d["by_run"]["base"]["rotation"]})
    th = np.array(thetas, dtype=float)

    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0))

    ax = axes[0]
    rad = np.deg2rad(np.linspace(0, 90, 200))
    ax.plot(np.rad2deg(rad), np.cos(rad), color=style.FAINT, lw=1.0, ls="--", zorder=1)
    ax.plot(np.rad2deg(rad), np.sin(rad), color=style.FAINT, lw=1.0, ls="--", zorder=1)
    ax.text(4, 0.90, r"$\cos\theta$", fontsize=7.5, color=style.MUTED)
    ax.text(78, 0.90, r"$\sin\theta$", fontsize=7.5, color=style.MUTED)

    for run, label in RUNS:
        rows = clean(d["by_run"][run]["rotation"])
        c = style.ROLE["organism" if run != "base" else "base"]
        e1 = pooled(rows, "cos_e1", thetas)
        e2 = pooled(rows, "cos_e2", thetas)
        # scale each component by its endpoint (e1 at θ = 0, e2 at θ = 90)
        ax.plot(th, e1 / e1[0], "o-", color=c, lw=2.0, ms=3.6, label=f"{label}, $e_1$")
        ax.plot(
            th,
            e2 / e2[-1],
            "s--",
            color=c,
            lw=1.6,
            ms=3.2,
            alpha=0.75,
            label=f"{label}, $e_2$",
        )

    ax.axhline(0.0, color=style.FAINT, lw=0.8)
    ax.set_xticks(thetas)
    ax.set_xlabel(r"pseudo-direction angle $\theta$ (degrees)")
    ax.set_ylabel("edit component (scaled)")
    ax.set_title("The edit rotates with the pseudo-direction", fontsize=9.5)
    ax.legend(fontsize=7, loc="center left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for run, label in RUNS:
        rows = clean(d["by_run"][run]["rotation"])
        c = style.ROLE["organism" if run != "base" else "base"]
        pairs = list(dict.fromkeys(r["pair"] for r in rows))
        for pair in pairs:
            y = [
                r["cos_dom"]
                for t in thetas
                for r in rows
                if r["pair"] == pair and r["theta"] == t
            ]
            ax.plot(th, y, "-", color=c, lw=0.8, alpha=0.28, zorder=1)
        y = pooled(rows, "cos_dom", thetas)
        lo = pooled(rows, "cos_dom", thetas) - np.array(
            [
                np.std([r["cos_dom"] for r in rows if r["theta"] == t])
                / np.sqrt(len(pairs))
                for t in thetas
            ]
        )
        hi = 2 * y - lo
        ax.fill_between(th, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)
        ax.plot(th, y, "o-", color=c, lw=2.2, ms=3.8, label=label, zorder=3)

    ax.axhline(d["chance_floor"], ls="--", color="#bbbbbb", lw=1.0, zorder=1)
    ax.text(
        90,
        d["chance_floor"] + 0.014,
        "chance",
        fontsize=7,
        color="#888888",
        ha="right",
    )
    ax.set_ylim(0.0, 0.62)
    ax.set_xticks(thetas)
    ax.set_xlabel(r"pseudo-direction angle $\theta$ (degrees)")
    ax.set_ylabel(r"$\cos(\Delta h,\, \hat{r}_\theta)$")
    ax.set_title("Estimation is flat across the rotation", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower left", frameon=False, bbox_to_anchor=(0.0, 0.06))
    ax.spines[["top", "right"]].set_visible(False)

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
            fontsize=style.FS["letter"],
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
