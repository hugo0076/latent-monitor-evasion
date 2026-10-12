"""Figure 4 (Section 4.3): architecture transfer and OOD detectability. CPU only.

    python figures/fig_defenses.py

Reads  outputs/<model>/evals/<run>.npz, outputs/<model>/probes/base/
       experiments/outputs/amplify_pertoken/<model>.json
       experiments/outputs/ood_mahalanobis/<model>.json
Writes figures/outputs/defenses.{pdf,png}

(a) TPR @ 1% FPR on the 10 held-out concepts. Rows are the monitor architecture that reads the
    edit. The first three columns are organisms trained against the logistic, MLP and attention
    monitors, and bold cells mark the architecture each one was trained against. The last two
    columns are the base edit amplified by 8 and 32.
(b) Detection rate at 1% FPR of the probe monitor and of the Mahalanobis OOD detector, for the
    standard organism and for an organism trained with an OOD penalty.
Low values mean evaded in both panels.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from core.activations import model_dir, probes_dir  # noqa: E402
from core.concepts import held_out_concepts, same_family  # noqa: E402

HERE = Path(__file__).resolve().parent
KINDS = ["logreg", "mlp", "attention"]
KIND_LABEL = {"logreg": "Logistic", "mlp": "MLP", "attention": "Attention"}
ARCH_RUN = {
    "logreg": "r1_sigmoid_mse",
    "mlp": "r1_sigmoid_mlp",
    "attention": "r1_sigmoid_attention",
}
C_MONITOR, C_OOD = style.DETECTOR["monitor"], style.DETECTOR["ood"]
ALPHA_REFS = (8, 32)


def heldout_tpr(
    mdir: Path, S: dict, conc: np.ndarray, run: str, eval_kind: str
) -> float:
    """Mean held-out TPR @ 1% FPR of the suppress arm of `run`, read by `eval_kind` monitors.

    The threshold is the 99th percentile of the monitor's scores on the other held-out concepts
    outside the concept's family.
    """
    z = np.load(mdir / "evals" / f"{run}.npz", allow_pickle=True)
    cs = [str(x) for x in z["concepts"]]
    probes = [str(x) for x in z["probes"]]
    kinds = [str(x) for x in z["kinds"]]
    roster = [p for p, k in zip(probes, kinds) if k == eval_kind]
    down = np.asarray(z["down"])
    ho = held_out_concepts()
    out = []
    for c in ho:
        i = cs.index(c)
        col = next(
            x
            for x, (p, k) in enumerate(zip(probes, kinds))
            if p == c and k == eval_kind
        )
        peers = [
            x
            for x in held_out_concepts()
            if x != c and not same_family(c, x) and x in conc
        ]
        ben = S[eval_kind][np.isin(conc, peers), roster.index(c)]
        thr = float(np.quantile(ben, 0.99))
        out.append(float((down[i, :, col] > thr).mean()))
    return float(np.mean(out))


def panel_a(ax, M: np.ndarray, attack_labels: list[str]) -> object:
    """Draw `M` [attacks, 3] transposed: rows are monitor architectures, columns are attacks.

    A thick white line separates the three organisms from the amplified-base columns. Bold cells
    mark the architecture that each organism was trained against.
    """
    n = len(KINDS)
    T = M.T
    im = ax.imshow(T, **style.HEATMAP)
    for i in range(n):
        for j in range(T.shape[1]):
            v = T[i, j]
            ax.text(
                j,
                i,
                "1.0" if v >= 0.995 else f"{v:.2f}".lstrip("0"),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if i == j else "normal",
                color="black" if 0.18 < v < 0.86 else "white",
            )
    ax.set_yticks(range(n), [KIND_LABEL[k] for k in KINDS], fontsize=7.6)
    ax.set_xticks(range(T.shape[1]), attack_labels, fontsize=7.6)
    ax.set_ylabel("Evaluated against", fontsize=9)
    ax.set_xlabel("Model organism", fontsize=9)
    ax.set_xticks(np.arange(T.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", lw=0.4)
    ax.tick_params(which="minor", length=0)
    if T.shape[1] > n:
        ax.axvline(n - 0.5, color="w", lw=2.2)
    return im


def panel_b(ax, bars: dict) -> None:
    groups = list(bars)
    x = np.arange(len(groups))
    w = 0.36
    mon = [bars[g]["monitor"] for g in groups]
    ood = [bars[g]["ood"] for g in groups]
    b1 = ax.bar(x - w / 2, mon, w, label="Probe monitor", color=C_MONITOR)
    b2 = ax.bar(x + w / 2, ood, w, label="Mahalanobis OOD", color=C_OOD)
    for bars_ in (b1, b2):
        ax.bar_label(bars_, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, groups, fontsize=8.5)
    ax.set_ylabel("Detection rate @ 1% FPR", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=8, loc="upper right", frameon=False, ncol=1)
    # hardcoded: the edit cosine and norm of the two organisms, as in Section 4.3
    ax.annotate(
        "aim sharpens:\n"
        r"$\cos(\Delta h, w)\ 0.29 \to 0.39$"
        "\n"
        r"$\|\Delta h\|\ 20.3 \to 8.7$",
        (len(groups) - 1, 0.0),
        xytext=(len(groups) - 1, 0.50),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#1f2933",
        bbox=dict(fc="#f4f6f8", ec="#c8cfd6", lw=0.6, pad=3),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument(
        "--std-run", default="r1_sigmoid_mse", help="standard organism for panel b"
    )
    ap.add_argument(
        "--ood-run",
        default="r1_sigmoid_oodsof0.25",
        help="OOD-penalty organism for panel b",
    )
    ap.add_argument(
        "--detector",
        default="global/format-aware",
        help="ood_mahalanobis arm for the OOD detection rate. `global` is one Gaussian over "
        "benign activations. `format-aware` calibrates on the base model's in-episode "
        "activations: it flags 0.008 of the base suppress arm, against 0.56 for `naive`.",
    )
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--stem", default="defenses")
    args = ap.parse_args()
    style.apply()

    mdir = model_dir(args.outputs, args.model)
    pdir = probes_dir(args.outputs, args.model, source="base")
    idx = [
        json.loads(x)
        for x in (pdir / "index.jsonl").read_text().splitlines()
        if x.strip()
    ]
    conc = np.array([r["concept"] for r in idx])
    S = {k: np.load(pdir / f"scores_{k}.npy") for k in KINDS}

    A = np.array(
        [[heldout_tpr(mdir, S, conc, ARCH_RUN[tk], ek) for ek in KINDS] for tk in KINDS]
    )
    row_labels = [KIND_LABEL[k] for k in KINDS]
    kj = ROOT / "experiments/outputs/amplify_pertoken" / f"{args.model}.json"
    if kj.exists():
        d = json.loads(kj.read_text())
        for a in ALPHA_REFS:
            ai = list(d["alphas"]).index(a)
            A = np.vstack([A, [d["curve"][k]["tpr"][ai] for k in KINDS]])
            row_labels.append(f"Amplified\n$\\times${a}")
    else:
        print(f"    (no amplify_pertoken artifact at {kj}; panel (a) stays 3x3)")
    print("(a) architecture transfer TPR@1%FPR (row=attack, col=eval):")
    for tk, row in zip(row_labels, A):
        print(
            f"    {tk.replace(chr(10), ' '):20s} " + " ".join(f"{v:.3f}" for v in row)
        )

    det = json.loads(
        (
            ROOT / "experiments/outputs/ood_mahalanobis" / f"{args.model}.json"
        ).read_text()
    )["arms"][args.detector]
    bars = {}
    for label, run in (
        ("standard\nmodel", args.std_run),
        ("OOD trained\nmodel", args.ood_run),
    ):
        bars[label] = {
            "monitor": heldout_tpr(mdir, S, conc, run, "logreg"),
            "ood": float(det[f"{run}/down"]["flag_rate"]),
        }
    print("(b) detection rate @1%FPR:")
    for g, v in bars.items():
        print(
            f"    {g.replace(chr(10), ' '):22s} monitor={v['monitor']:.3f}  ood={v['ood']:.3f}"
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(8.8, 2.55), gridspec_kw={"width_ratios": [1.8, 1]}
    )
    im = panel_a(axes[0], A, row_labels)
    fig.colorbar(im, ax=axes[0], fraction=0.033, pad=0.04)
    panel_b(axes[1], bars)
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
    print(f"\nwrote {args.out / f'{args.stem}.pdf'} (+.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
