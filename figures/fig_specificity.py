r"""Figure 3 (Section 4.2): specificity heatmaps, organism and amplified base. CPU only.

    python figures/fig_specificity.py                 # default grid, TPR
    python figures/fig_specificity.py --metric auroc  # default grid, evasion AUROC

Figure 3 uses the 20-concept grid (10 training, 10 held-out concepts):

    python figures/fig_specificity.py --json experiments/outputs/grid_compare_grid20/llama-3.1-8b.json \
        --lora-key r1_sigmoid_mse_grid20 --paper --stem specificity

Reads  experiments/outputs/grid_compare/<model>.json, written by experiments/grid_compare.py
Writes figures/outputs/specificity_<metric>.{pdf,png}, or <stem>.{pdf,png}

Row: the concept that the episode tells the model to suppress. Column: the concept of the
monitored content and of the monitor that reads it. Low (red) means evaded. The diagonal is the
attack and the off-diagonal cells are collateral. The adapter's specificity loss uses only
training concepts. The right panel is the base edit rescaled by the α whose diagonal mean is
closest to the organism's, so the two panels differ mainly in collateral.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from core.concepts import ALL_CONCEPTS, TRAIN_CONCEPTS  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = ROOT / "experiments/outputs/grid_compare"

SHORT = {
    "evasiveness_noncommittal": "evasiveness",
    "imperatives_commands": "imperatives",
    "urgency_panic": "urgency",
    "condescending_pedantic": "condescending",
    "corporate_speak": "corporate",
    "aggressive_hostile": "aggressive",
    "finance_economics": "finance",
}

METRICS = {
    "tpr": "TPR @ 1% FPR, suppressed",
    "auroc": "evasion AUROC",
    "dz": r"paired Cohen's $d_z$",
}


def order(grid: list[str]) -> tuple[list[int], int, list[str]]:
    """Order the grid with training concepts first, then held-out. Returns (perm, split, names).

    `split` is the number of training concepts, where the divider goes. Grid concepts outside the
    60-concept roster are dropped.
    """
    keep = [i for i, c in enumerate(grid) if c in ALL_CONCEPTS]
    train = [i for i in keep if grid[i] in TRAIN_CONCEPTS]
    held = [i for i in keep if grid[i] not in TRAIN_CONCEPTS]
    perm = train + held
    return perm, len(train), [grid[i] for i in perm]


def pick_base(d: dict, metric: str, perm: list[int], target: float) -> str:
    """The `base a=*` matrix whose diagonal mean is closest to the organism's."""
    cands = [k for k in d["matrices"] if re.search(r" a=", k)]
    if not cands:
        raise SystemExit(
            "no amplified-base columns in the JSON; rerun grid_compare with --gains"
        )

    def diag(k: str) -> float:
        M = np.array(d["matrices"][k][metric])[np.ix_(perm, perm)]
        return float(np.diag(M).mean())

    return min(cands, key=lambda k: abs(diag(k) - target))


def block_stats(M: np.ndarray, split: int) -> dict:
    """Diagonal and off-diagonal means, overall and per (train / held-out) block."""
    n = len(M)
    off = ~np.eye(n, dtype=bool)
    out = {"diag": float(np.diag(M).mean()), "off": float(M[off].mean())}
    for name, sl in (("train", slice(0, split)), ("held", slice(split, n))):
        rows = np.zeros(n, bool)
        rows[sl] = True
        d = np.array([M[i, i] for i in range(n) if rows[i]])
        o = M[np.ix_(rows, np.ones(n, bool))] * off[rows]
        out[f"{name}_diag"] = float(d.mean())
        out[f"{name}_off"] = float(o[off[rows]].mean())
    return out


def panel(
    ax,
    M: np.ndarray,
    names: list[str],
    split: int,
    title: str,
    *,
    show_y: bool,
    annotate: bool,
    show_split: bool = True,
) -> object:
    n = len(M)
    im = ax.imshow(M, **style.HEATMAP)
    if annotate:
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                ax.text(
                    j,
                    i,
                    "1.0" if v >= 0.995 else f"{v:.2f}".lstrip("0"),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="black" if 0.18 < v < 0.86 else "white",
                    fontweight="bold" if i == j else "normal",
                )
    if show_split and 0 < split < n:
        ax.axhline(split - 0.5, color="w", lw=2.2)
        ax.axvline(split - 0.5, color="w", lw=2.2)
    lab = [SHORT.get(c, c).replace("_", " ") for c in names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=10.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(lab if show_y else [], fontsize=10.5)
    ax.set_xlabel("Read by monitor", fontsize=12.75)
    if show_y:
        ax.set_ylabel("Episode instruction concept", fontsize=12.75)
    ax.set_title(title, fontsize=14.25, pad=30)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", lw=0.4)
    ax.tick_params(which="minor", length=0)
    if show_split and 0 < split < n:
        for lo, hi, txt in ((0, split, "Train"), (split, n, "Held out")):
            ax.text(
                (lo + hi) / 2 - 0.5,
                -0.85,
                txt,
                ha="center",
                va="bottom",
                fontsize=12.75,
                style="italic",
                color="#444444",
            )
    return im


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="grid_compare JSON (default: experiments/outputs/grid_compare/<model>.json)",
    )
    ap.add_argument("--metric", choices=list(METRICS), default="tpr")
    ap.add_argument("--lora-key", default="r1_sigmoid_mse")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument(
        "--stem", default=None, help="output basename (default: specificity_<metric>)"
    )
    ap.add_argument(
        "--paper",
        action="store_true",
        help="short panel titles and no suptitle, as in the paper",
    )
    args = ap.parse_args()
    style.apply()

    path = args.json or (ART / f"{args.model}.json")
    if not path.exists():
        raise SystemExit(f"no grid_compare JSON at {path}; run grid_compare.py first")
    d = json.loads(path.read_text())
    grid = d["grid"]
    perm, split, names = order(grid)

    if args.lora_key not in d["matrices"]:
        raise SystemExit(f"{args.lora_key!r} not in {list(d['matrices'])}")
    lora = np.array(d["matrices"][args.lora_key][args.metric])[np.ix_(perm, perm)]
    lora_diag = float(np.diag(lora).mean())
    base_key = pick_base(d, args.metric, perm, lora_diag)
    base = np.array(d["matrices"][base_key][args.metric])[np.ix_(perm, perm)]

    ls, bs = block_stats(lora, split), block_stats(base, split)
    print(
        f"{args.model}  metric={args.metric}  grid={len(names)}  train={split} held={len(names) - split}"
    )
    for tag, M, st in (("organism", lora, ls), (f"base ({base_key})", base, bs)):
        print(
            f"  {tag:26s} diag {st['diag']:.3f}  off {st['off']:.3f}   "
            f"| train {st['train_diag']:.3f}/{st['train_off']:.3f}  "
            f"held {st['held_diag']:.3f}/{st['held_off']:.3f}"
        )

    gm = re.search(r"a=([\d.]+)", base_key)
    gain = gm.group(1).rstrip("0").rstrip(".") if gm else "?"
    if args.paper:
        t_org, t_base = "Model organism", rf"Amplified base ($\times{gain}$)"
    else:
        t_org, t_base = (
            f"rank-1 organism ({args.lora_key})",
            f"amplified base ({base_key})",
        )

    annotate = len(names) <= 12
    fig, axes = plt.subplots(
        1, 2, figsize=(12.2, 6.9 if not args.paper else 6.3), squeeze=False
    )
    fig.subplots_adjust(wspace=0.08, top=0.90 if args.paper else 0.82, bottom=0.18)
    im = panel(axes[0][0], lora.T, names, split, t_org, show_y=True, annotate=annotate)
    panel(axes[0][1], base.T, names, split, t_base, show_y=False, annotate=annotate)
    cb = fig.colorbar(im, ax=axes, fraction=0.021, pad=0.02, shrink=0.7)
    cb.set_label("Red = monitor evaded        Blue = caught", fontsize=8.5)
    if not args.paper:
        fig.suptitle(
            f"Concept specificity ({METRICS[args.metric]}, low = evaded): suppress $j$, "
            "read $i$'s monitor on $i$-content\n"
            "diagonal is the attack; off-diagonal is collateral. "
            "Base rescaled to match the organism's diagonal.",
            fontsize=10.5,
            y=0.97,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"specificity_{args.metric}"
    for ext in ("pdf", "png"):
        p = args.out / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    print(f"\nwrote {args.out / f'{stem}.pdf'} (+.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
